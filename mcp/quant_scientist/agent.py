"""QuantDataScientist — offline alpha-mining MCP agent.

The agent's job (per ARCHITECTURE.md §13.1):

1. Pull the last 30 days of closed trades from the trade repository.
2. Pull 6 months of OHLCV per symbol via :class:`ParquetDataLoader`.
3. Build a structured prompt and ask the LLM to write Python analysis
   code. The prompt mandates BH-FDR, out-of-sample validation, and
   minimum effect-size gates.
4. Run the LLM-generated code in a restricted sandbox (`sandbox.py`).
   The LLM never touches the filesystem or the database directly.
5. Parse the sandbox `result` into :class:`AlphaFactor` records,
   apply confidence decay to the previous run's factors, merge, and
   write `alpha_factors.json` (unless `dry_run=True`).
6. Return an :class:`AlphaFactorsReport` summarising the run.

The workflow is fail-safe at every step: data fetch errors, LLM parse
failures, sandbox crashes, and write errors all log + propagate as a
report with `error` set, NEVER as a half-written `alpha_factors.json`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.quant_scientist.decay import apply_decay, merge_factors
from mcp.quant_scientist.factor import (
    AlphaFactor,
    AlphaFactorsReport,
    factors_to_nested_json,
    nested_json_to_factors,
)
from mcp.quant_scientist.prompts import SYSTEM_PROMPT, build_analysis_prompt
from mcp.quant_scientist.sandbox import (
    SandboxExecutionError,
    SandboxRejected,
    run_analysis,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from llm.base import LLMProvider
    from storage.repositories.base import TradeRepository


_DEFAULT_OUTPUT_PATH = Path("alpha_factors.json")
_DEFAULT_LOOKBACK_DAYS = 30
_DEFAULT_OHLCV_LOOKBACK_DAYS = 180  # 6 months
_DEFAULT_TIMEFRAMES = ("1h", "4h")
_LLM_MAX_TOKENS = 4096
_LLM_TEMPERATURE = 0.3


class QuantScientistError(Exception):
    """Top-level marker for any QuantDataScientist failure."""


class AnalysisCodeError(QuantScientistError):
    """LLM produced unusable analysis code (parse, screen, or exec)."""


class QuantDataScientist:
    """Offline alpha-mining agent. Run via cron, NOT inside the live pipeline.

    This class is intentionally I/O-heavy and side-effecting — it
    mutates `alpha_factors.json` on disk. It is NOT used by the live
    trading pipeline; ConvictionAgent only *reads* the file the agent
    produces.

    Construction is cheap (no DB connection, no LLM call). All real
    work happens inside :meth:`run`.
    """

    def __init__(
        self,
        llm_provider: "LLMProvider",
        trade_repository: "TradeRepository | None" = None,
        data_loader: Any = None,
        output_path: Path | str = _DEFAULT_OUTPUT_PATH,
        db_url: str | None = None,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
        ohlcv_lookback_days: int = _DEFAULT_OHLCV_LOOKBACK_DAYS,
        timeframes: tuple[str, ...] = _DEFAULT_TIMEFRAMES,
        bot_ids: list[str] | None = None,
    ) -> None:
        """
        Args:
            llm_provider: Anything implementing :class:`LLMProvider`.
                The agent calls ``generate_text`` exactly once per run.
            trade_repository: Optional pre-built TradeRepository. If
                None, the agent will lazily build one from ``db_url``
                via ``storage.repositories.get_repositories``.
            data_loader: Optional pre-built ParquetDataLoader (or any
                duck-typed object with the same ``load`` signature).
                The agent uses it to pull 6 months of OHLCV per symbol.
                When ``None`` the run still works but the LLM gets an
                empty ``ohlcv`` dict — useful for tests.
            output_path: Where to write ``alpha_factors.json``. The
                directory must exist (or be creatable). The agent will
                NEVER write to any other path — that's the §13.1.6
                safety contract.
            db_url: PostgreSQL DSN. Used only if ``trade_repository``
                is None and the runner needs to construct one.
            lookback_days: Trade history window (default 30).
            ohlcv_lookback_days: OHLCV history window (default 180).
            timeframes: Tuple of timeframes to load OHLCV for. Default
                ("1h", "4h").
            bot_ids: Optional explicit list of bot ids. When None,
                the agent uses the first bot it can find via the bot
                repo. Tests pass an explicit list.
        """
        self._llm = llm_provider
        self._trade_repo = trade_repository
        self._data_loader = data_loader
        self._output_path = Path(output_path)
        self._db_url = db_url
        self._lookback_days = int(lookback_days)
        self._ohlcv_lookback_days = int(ohlcv_lookback_days)
        self._timeframes = tuple(timeframes)
        self._bot_ids = list(bot_ids) if bot_ids else None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, dry_run: bool = False) -> AlphaFactorsReport:
        """Execute one full alpha-mining cycle.

        Args:
            dry_run: When True the agent skips the final write to
                ``alpha_factors.json`` and reports ``dry_run=True``.

        Returns:
            An :class:`AlphaFactorsReport` even on failure. Callers
            should check ``report.error`` rather than catching.
        """
        report = AlphaFactorsReport(
            output_path=str(self._output_path),
            dry_run=dry_run,
        )

        # 1. Pull recent trades
        try:
            trades = await self._fetch_recent_trades()
        except Exception as e:
            logger.exception("QuantDataScientist: trade fetch failed")
            report.error = f"trade_fetch_failed: {e}"
            return report

        report.trades_analyzed = len(trades)
        report.symbols_analyzed = len({t.get("symbol") for t in trades if t.get("symbol")})

        if not trades:
            logger.info("QuantDataScientist: no trades in lookback window — nothing to mine")
            # Still apply decay to the existing factors so weights age
            # naturally even on quiet weeks.
            report.factors, _, decay_counts = self._merge_with_existing(
                new_factors=[],
            )
            report.pruned_count = decay_counts.get("pruned", 0)
            self._maybe_write(report)
            return report

        # 2. Load 6 months of OHLCV per symbol
        ohlcv = self._load_ohlcv(trades)

        # 3. Build the prompt
        trade_summary = self._summarise_trades(trades)
        symbols = sorted({t["symbol"] for t in trades if t.get("symbol")})
        user_prompt = build_analysis_prompt(
            trade_summary=trade_summary,
            available_symbols=symbols,
            timeframes=self._timeframes,
        )

        # 4. Ask the LLM for analysis code
        try:
            llm_response = await self._llm.generate_text(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name="quant_data_scientist",
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                cache_system_prompt=True,
            )
        except Exception as e:
            logger.exception("QuantDataScientist: LLM call failed")
            report.error = f"llm_call_failed: {e}"
            return report

        code = self._extract_code(llm_response.content)
        if not code:
            logger.warning(
                "QuantDataScientist: could not extract Python code from LLM response"
            )
            report.error = "llm_no_code_block"
            return report

        # 5. Run in sandbox
        trades_df = self._trades_to_dataframe(trades)
        try:
            raw_result = run_analysis(code, trades_df=trades_df, ohlcv=ohlcv)
        except SandboxRejected as e:
            logger.warning(f"QuantDataScientist: sandbox rejected LLM code: {e}")
            report.error = f"sandbox_rejected: {e}"
            return report
        except SandboxExecutionError as e:
            logger.warning(f"QuantDataScientist: sandbox execution failed: {e}")
            report.error = f"sandbox_execution_failed: {e}"
            return report

        # 6. Parse + validate every result row → AlphaFactor
        new_factors = self._parse_result(raw_result)

        # 7. Decay-aware merge with the previous run's factors
        merged, _, counts = self._merge_with_existing(new_factors=new_factors)
        report.factors = merged
        report.new_count = counts.get("new", 0)
        report.confirmed_count = counts.get("confirmed", 0)
        report.pruned_count = counts.get("pruned", 0)

        # 8. Write (unless dry-run)
        self._maybe_write(report)
        return report

    # ------------------------------------------------------------------
    # Trade fetch
    # ------------------------------------------------------------------

    async def _fetch_recent_trades(self) -> list[dict]:
        repo = await self._resolve_trade_repo()
        if repo is None:
            return []
        bot_ids = self._bot_ids or await self._discover_bot_ids(repo)
        if not bot_ids:
            return []

        cutoff = datetime.now(tz=timezone.utc).timestamp() - (
            self._lookback_days * 86_400
        )
        all_trades: list[dict] = []
        for bot_id in bot_ids:
            try:
                rows = await repo.get_trades_by_bot(bot_id, limit=500)
            except Exception as e:
                logger.warning(
                    f"QuantDataScientist: get_trades_by_bot({bot_id}) failed: {e}"
                )
                continue
            for row in rows:
                if not _is_closed(row):
                    continue
                if not _within_lookback(row, cutoff):
                    continue
                all_trades.append(row)
        logger.info(
            f"QuantDataScientist: pulled {len(all_trades)} closed trade(s) "
            f"from {len(bot_ids)} bot(s) over {self._lookback_days} day(s)"
        )
        return all_trades

    async def _resolve_trade_repo(self):
        if self._trade_repo is not None:
            return self._trade_repo
        # Lazy construction via the repositories factory.
        try:
            from storage.repositories import get_repositories

            saved = os.environ.get("DATABASE_URL")
            if self._db_url:
                os.environ["DATABASE_URL"] = self._db_url
            try:
                repos = await get_repositories(
                    "postgresql" if self._db_url else None
                )
            finally:
                if self._db_url and saved is None:
                    os.environ.pop("DATABASE_URL", None)
                elif self._db_url and saved is not None:
                    os.environ["DATABASE_URL"] = saved
            self._trade_repo = repos.trades
            return self._trade_repo
        except Exception as e:
            logger.warning(f"QuantDataScientist: trade repo construction failed: {e}")
            return None

    @staticmethod
    async def _discover_bot_ids(repo) -> list[str]:
        # The TradeRepository ABC has no get_all_bots/list_bot_ids method —
        # tests + production should pass `bot_ids` explicitly. Returning
        # an empty list here causes the run to short-circuit cleanly.
        return []

    # ------------------------------------------------------------------
    # OHLCV fetch
    # ------------------------------------------------------------------

    def _load_ohlcv(self, trades: list[dict]) -> dict:
        if self._data_loader is None:
            logger.info(
                "QuantDataScientist: no data_loader provided — passing empty ohlcv"
            )
            return {}

        symbols = sorted({t["symbol"] for t in trades if t.get("symbol")})
        end_dt = datetime.now(tz=timezone.utc)
        start_dt = end_dt - _safe_timedelta(self._ohlcv_lookback_days)

        ohlcv: dict[str, dict] = {}
        for symbol in symbols:
            per_tf: dict = {}
            for timeframe in self._timeframes:
                try:
                    df = self._data_loader.load(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_dt,
                        end_date=end_dt,
                    )
                except FileNotFoundError as e:
                    logger.warning(
                        f"QuantDataScientist: no parquet for {symbol} {timeframe}: {e}"
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        f"QuantDataScientist: ohlcv load failed for "
                        f"{symbol} {timeframe}: {e}"
                    )
                    continue
                per_tf[timeframe] = df
            if per_tf:
                ohlcv[symbol] = per_tf
        return ohlcv

    # ------------------------------------------------------------------
    # Trade summary + dataframe
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_trades(trades: list[dict]) -> dict:
        if not trades:
            return {"trade_count": 0, "unique_symbols": 0, "win_rate": 0.0, "avg_r": 0.0}
        wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
        rs = [t.get("r_multiple") for t in trades if t.get("r_multiple") is not None]
        return {
            "trade_count": len(trades),
            "unique_symbols": len({t.get("symbol") for t in trades if t.get("symbol")}),
            "win_rate": wins / len(trades),
            "avg_r": sum(rs) / len(rs) if rs else 0.0,
        }

    @staticmethod
    def _trades_to_dataframe(trades: list[dict]):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return trades  # last-resort fallback for envs without pandas
        return pd.DataFrame(trades)

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    _CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    @classmethod
    def _extract_code(cls, content: str) -> str | None:
        """Pull the first ```python block out of the LLM response.

        Tolerates the LLM forgetting the language tag (``` ... ```)
        and the LLM dumping raw code with no fences at all.
        """
        if not content:
            return None
        match = cls._CODE_BLOCK_RE.search(content)
        if match:
            return match.group(1).strip()
        # No fences? If the response looks like Python source (has
        # `result =` somewhere), use it as-is. Otherwise give up.
        if "result" in content and "=" in content:
            return content.strip()
        return None

    @staticmethod
    def _parse_result(rows: list[dict]) -> list[AlphaFactor]:
        """Coerce sandbox `result` rows into AlphaFactor records.

        Invalid rows are dropped with a warning rather than crashing
        the whole run — partial output is better than nothing.
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        out: list[AlphaFactor] = []
        for i, row in enumerate(rows):
            try:
                pattern = row.get("pattern") or ""
                symbol = row.get("symbol") or ""
                timeframe = row.get("timeframe") or ""
                avg_r = row.get("avg_r")
                if avg_r is None:
                    avg_r = row.get("avg_R")
                factor = AlphaFactor(
                    pattern=str(pattern),
                    symbol=str(symbol),
                    timeframe=str(timeframe),
                    win_rate=float(row["win_rate"]),
                    avg_r=float(avg_r),
                    n=int(row["n"]),
                    confidence=str(row.get("confidence", "low")).lower(),
                    discovered_at=str(row.get("discovered_at") or now_iso),
                    last_confirmed=str(row.get("last_confirmed") or now_iso),
                    decay_weight=float(row.get("decay_weight", 1.0)),
                    note=row.get("note"),
                )
                factor.validate()
                out.append(factor)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"QuantDataScientist: dropping invalid result row {i}: {e}"
                )
        return out

    # ------------------------------------------------------------------
    # Existing-factor merge
    # ------------------------------------------------------------------

    def _merge_with_existing(
        self, new_factors: list[AlphaFactor]
    ) -> tuple[list[AlphaFactor], list[AlphaFactor], dict[str, int]]:
        """Load → decay → merge → return (final, decayed, counts)."""
        existing = self._load_existing_factors()
        decayed = apply_decay(existing)
        # The "pruned" count from apply_decay covers factors aged out
        # before the merge step. The "pruned" count from merge covers
        # factors that survived decay but ARE_NOT in the new batch and
        # have weight below threshold AFTER ageing. Add the two so the
        # report's pruned_count is the total disappearance count.
        pre_merge_pruned = len(existing) - len(decayed)
        merged, counts = merge_factors(
            new_factors=new_factors,
            existing_factors=decayed,
        )
        counts["pruned"] = counts.get("pruned", 0) + pre_merge_pruned
        return merged, decayed, counts

    def _load_existing_factors(self) -> list[AlphaFactor]:
        if not self._output_path.exists():
            return []
        try:
            payload = json.loads(self._output_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"QuantDataScientist: could not read existing {self._output_path}: {e}"
            )
            return []
        return nested_json_to_factors(payload)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _maybe_write(self, report: AlphaFactorsReport) -> None:
        if report.dry_run:
            logger.info(
                f"QuantDataScientist: dry-run, NOT writing "
                f"{len(report.factors)} factor(s)"
            )
            return
        try:
            self._write_alpha_factors(report.factors)
        except OSError as e:
            logger.exception("QuantDataScientist: write failed")
            report.error = f"write_failed: {e}"

    def _write_alpha_factors(self, factors: list[AlphaFactor]) -> None:
        """Atomically write the nested JSON to ``self._output_path``.

        Refuses to write to any path other than ``self._output_path``
        per ARCHITECTURE §13.1.6 — even if a downstream caller passes
        a file path argument that points elsewhere, this method only
        ever touches the configured path.
        """
        nested = factors_to_nested_json(factors)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tmp + rename so a crashed write never leaves
        # the file half-populated.
        tmp = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
        tmp.write_text(json.dumps(nested, indent=2, sort_keys=True))
        tmp.replace(self._output_path)
        logger.info(
            f"QuantDataScientist: wrote {len(factors)} factor(s) to {self._output_path}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_closed(trade: dict) -> bool:
    status = (trade.get("status") or "").lower()
    if status == "closed":
        return True
    # Tolerate trades without an explicit status field by checking for
    # an exit_time / exit_reason — both indicate a closed lifecycle.
    return bool(trade.get("exit_time") or trade.get("exit_reason"))


def _within_lookback(trade: dict, cutoff_seconds: float) -> bool:
    """Return True if trade.entry_time is more recent than the cutoff."""
    entry_time = trade.get("entry_time") or trade.get("exit_time")
    if entry_time is None:
        return True  # don't drop trades that have no timestamp
    try:
        if isinstance(entry_time, (int, float)):
            ts = float(entry_time)
            # Heuristic: ms vs s
            if ts > 1e12:
                ts /= 1000.0
            return ts >= cutoff_seconds
        cleaned = str(entry_time).replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).timestamp() >= cutoff_seconds
    except (TypeError, ValueError):
        return True


def _safe_timedelta(days: int):
    from datetime import timedelta
    return timedelta(days=days)
