"""Tests for `quantagent.main.main()` CLI dispatcher.

Specifically the global mode flags (`--shadow`, `--paper`) — they
mutate process-level environment variables and select which subset
of bots the runner loads from the DB. Both flags are mutually
exclusive at the CLI layer; the test suite pins all the documented
behaviours so a future refactor can't silently regress them.

We do NOT actually invoke `_run_server()` here — that would spin up
the full FastAPI stack and hit the database. The acceptance criterion
"paper mode loads paper bots only" is covered by a repository-level
test that exercises `get_active_bots_by_mode("paper")` directly,
which is the only DB-touching code path the CLI dispatcher controls.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import pytest_asyncio

from quantagent.main import main


# ---------------------------------------------------------------------------
# Test fixtures + isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_mode_environment():
    """Snapshot mode env vars; CLEAR them at test start; restore after.

    Two-phase isolation: snapshot the inherited value at fixture
    setup, CLEAR the env so the test starts from a known empty
    state regardless of what previous tests in the same pytest
    process left behind, then restore the snapshot at teardown so
    no pollution leaks forward either. This is stronger than the
    typical "save / yield / restore" pattern because it also
    defends against UPSTREAM pollution — e.g. another test file
    setting ``QUANTAGENT_SHADOW=1`` and forgetting to clean it up.

    The autouse marker means every test in this file gets a clean
    slate automatically.
    """
    keys = (
        "QUANTAGENT_SHADOW",
        "QUANTAGENT_PAPER",
        "HYPERLIQUID_TESTNET",
        "LANGCHAIN_PROJECT",
        "PORT",
    )
    saved = {k: os.environ.get(k) for k in keys}

    # Clear at start so the test starts from a known empty state
    for k in keys:
        os.environ.pop(k, None)

    yield

    # Restore the original snapshot (whatever was there before)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _run_main(*argv: str) -> None:
    """Invoke ``main()`` with ``argv`` mocked into ``sys.argv[1:]``.

    The ``run``/``migrate``/``seed`` subcommands are patched out to
    no-ops so the dispatcher returns immediately after parsing the
    flags — these tests are about argv parsing + env-var mutation,
    NOT about actually starting the server.
    """
    with patch("sys.argv", ["quantagent.main", *argv]), \
         patch("quantagent.main.run") as mock_run, \
         patch("quantagent.main.migrate") as mock_migrate, \
         patch("quantagent.main.seed") as mock_seed:
        main()


# ---------------------------------------------------------------------------
# --paper flag — env var mutation
# ---------------------------------------------------------------------------


class TestPaperFlagSetsEnvironment:
    """`--paper` must set every env var the downstream layers depend on."""

    def test_paper_sets_quantagent_paper_env_var(self):
        _run_main("run", "--paper")
        assert os.environ.get("QUANTAGENT_PAPER") == "1"

    def test_paper_sets_hyperliquid_testnet_true(self):
        """The HyperliquidAdapter reads `HYPERLIQUID_TESTNET` from
        the environment to branch credential lookup + endpoint URL.
        Paper mode must set it so every adapter built downstream
        (factory path AND any direct ctor in scripts) uses testnet
        regardless of how it was constructed."""
        _run_main("run", "--paper")
        assert os.environ.get("HYPERLIQUID_TESTNET") == "true"

    def test_paper_sets_langchain_project_for_observability_routing(self):
        """Paper-mode LLM traces should land in a separate LangSmith
        project so they don't pollute the live observability dashboards."""
        _run_main("run", "--paper")
        assert os.environ.get("LANGCHAIN_PROJECT") == "quantagent-paper"

    def test_paper_defaults_port_to_8001(self):
        """Default port lets paper run alongside shadow/live (port 8000)
        on the same host without colliding."""
        os.environ.pop("PORT", None)
        _run_main("run", "--paper")
        assert os.environ.get("PORT") == "8001"

    def test_paper_does_not_override_explicit_port(self):
        """If the operator explicitly sets `PORT=9000`, paper mode must
        respect it. ``setdefault`` is the right primitive for this —
        only fills the slot when nothing is set."""
        os.environ["PORT"] = "9000"
        _run_main("run", "--paper")
        assert os.environ.get("PORT") == "9000"

    def test_paper_strips_flag_from_argv_before_subcommand_dispatch(self):
        """The `--paper` flag must be stripped from argv before the
        positional subcommand dispatch so existing logic still
        recognises `run` as the command. Verified by patching out
        `run()` and asserting it WAS called (the dispatch reached it,
        proving `--paper` didn't confuse the positional parser)."""
        with patch("sys.argv", ["quantagent.main", "run", "--paper"]), \
             patch("quantagent.main.run") as mock_run, \
             patch("quantagent.main.migrate"), \
             patch("quantagent.main.seed"):
            main()
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# --shadow + --paper mutual exclusion
# ---------------------------------------------------------------------------


class TestShadowPaperMutualExclusion:
    """Both flags at once is almost certainly an operator typo. The
    safest response is to exit loudly so they pick the one they meant.
    """

    def test_both_flags_exits_with_error(self, capsys):
        with patch("sys.argv", ["quantagent.main", "run", "--shadow", "--paper"]), \
             patch("quantagent.main.run"), \
             patch("quantagent.main.migrate"), \
             patch("quantagent.main.seed"):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.out

    def test_both_flags_exits_BEFORE_setting_either_env_var(self):
        """Defense in depth: if the operator passes both flags, we
        must exit BEFORE mutating ANY env var, otherwise the next
        process in the same shell session inherits a stale flag."""
        # Pre-condition: neither var is set
        os.environ.pop("QUANTAGENT_SHADOW", None)
        os.environ.pop("QUANTAGENT_PAPER", None)

        with patch("sys.argv", ["quantagent.main", "run", "--shadow", "--paper"]), \
             patch("quantagent.main.run"), \
             patch("quantagent.main.migrate"), \
             patch("quantagent.main.seed"):
            with pytest.raises(SystemExit):
                main()

        # Neither env var should have been touched
        assert "QUANTAGENT_SHADOW" not in os.environ
        assert "QUANTAGENT_PAPER" not in os.environ

    def test_shadow_alone_does_not_set_paper_env_vars(self):
        """Negative control — shadow mode shouldn't accidentally set
        any of the paper-mode-specific env vars (PAPER marker, the
        testnet flag, the default port). The LangSmith project IS
        set by shadow mode (Paper Trading Task 4) — to
        ``quantagent-shadow``, NOT the paper-mode value — so shadow
        traces don't leak into the live observability dashboard.
        That's covered explicitly by
        ``TestShadowFlagSetsLangSmithProject`` in test_llm.py.
        """
        _run_main("run", "--shadow")
        assert os.environ.get("QUANTAGENT_SHADOW") == "1"
        assert os.environ.get("QUANTAGENT_PAPER") is None
        assert os.environ.get("HYPERLIQUID_TESTNET") is None
        # Shadow now DOES set LANGCHAIN_PROJECT, but to its OWN value
        # (not the paper one). Pin both halves of that contract.
        assert os.environ.get("LANGCHAIN_PROJECT") == "quantagent-shadow"
        assert os.environ.get("LANGCHAIN_PROJECT") != "quantagent-paper"

    def test_paper_alone_does_not_set_shadow_env_var(self):
        """Negative control — paper mode shouldn't accidentally set
        the shadow env var."""
        _run_main("run", "--paper")
        assert os.environ.get("QUANTAGENT_PAPER") == "1"
        assert os.environ.get("QUANTAGENT_SHADOW") is None


# ---------------------------------------------------------------------------
# --paper appears in --help
# ---------------------------------------------------------------------------


class TestHelpText:
    """The help text should advertise the new flag so operators
    discover it without reading the source."""

    def test_help_mentions_paper_flag(self, capsys):
        _run_main("--help")
        captured = capsys.readouterr()
        assert "--paper" in captured.out
        assert "testnet" in captured.out.lower()

    def test_bare_invocation_shows_help_with_paper_line(self, capsys):
        """Running with no args shows help, which must include --paper."""
        _run_main()
        captured = capsys.readouterr()
        assert "--paper" in captured.out


# ---------------------------------------------------------------------------
# Repo-level: paper mode loads paper bots only
# ---------------------------------------------------------------------------


class TestPaperModeBotLoading:
    """The CLI dispatcher's job is to set the env vars; ``_run_server``
    then resolves the env vars to a ``bot_mode`` string and calls
    ``repos.bots.get_active_bots_by_mode(bot_mode)``. We've already
    verified the dispatcher sets the env vars correctly above; here
    we verify the END of the chain — that asking the repo for
    "paper" bots returns ONLY paper bots, not shadow or live.

    This is the test that pins "paper mode loads paper bots only"
    without spinning up the full FastAPI stack. Combined with
    ``test_paper_sets_quantagent_paper_env_var`` above, the end-to-end
    contract is covered.
    """

    @pytest_asyncio.fixture
    async def repos(self, tmp_path):
        """Fresh on-disk SQLite repos for each test."""
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test_main_cli.db")
        r = SQLiteRepositories(db_path=db_path)
        await r.init_db()
        return r

    @pytest.mark.asyncio
    async def test_paper_mode_loads_paper_bots_only(self, repos):
        """Insert one bot per mode, fetch by 'paper', expect ONE row.

        This pins the second half of the dispatch chain: even though
        the dispatcher resolves `bot_mode = "paper"` correctly, if
        the repo accessor accidentally returned all bots OR returned
        live bots OR returned shadow bots, paper-mode operators would
        get the wrong workload. The repo-level test catches both
        regressions in one shot.
        """
        await repos.bots.save_bot({
            "id": "live-bot",
            "user_id": "u1", "symbol": "BTC-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "mode": "live",
        })
        await repos.bots.save_bot({
            "id": "shadow-bot",
            "user_id": "u1", "symbol": "ETH-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "mode": "shadow",
        })
        await repos.bots.save_bot({
            "id": "paper-bot-1",
            "user_id": "u1", "symbol": "SOL-USDC", "timeframe": "1h",
            "exchange": "hyperliquid", "mode": "paper",
        })
        await repos.bots.save_bot({
            "id": "paper-bot-2",
            "user_id": "u1", "symbol": "AVAX-USDC", "timeframe": "30m",
            "exchange": "hyperliquid", "mode": "paper",
        })

        loaded = await repos.bots.get_active_bots_by_mode("paper")
        assert {b["id"] for b in loaded} == {"paper-bot-1", "paper-bot-2"}

        # Cross-check the other modes still work — proves the filter
        # is exact-match, not substring or prefix
        live_loaded = await repos.bots.get_active_bots_by_mode("live")
        assert {b["id"] for b in live_loaded} == {"live-bot"}
        shadow_loaded = await repos.bots.get_active_bots_by_mode("shadow")
        assert {b["id"] for b in shadow_loaded} == {"shadow-bot"}
