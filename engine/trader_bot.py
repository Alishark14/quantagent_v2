"""TraderBot: ephemeral worker that runs one analysis cycle and executes.

Spawned by BotManager on SetupDetected or schedule. Runs the full
pipeline, executes the resulting action, registers position with
Sentinel, then dies. All data persists in the database; the process
does not.

On any error, returns an ERROR result dict — never crashes the parent.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from engine.execution.executor import Executor
from engine.pipeline import AnalysisPipeline
from engine.types import TradeAction
from sentinel.position_manager import PositionManager

logger = logging.getLogger(__name__)


class TraderBot:
    """Ephemeral bot: spawn, analyze, execute, die."""

    def __init__(
        self,
        bot_id: str,
        pipeline: AnalysisPipeline,
        executor: Executor,
        position_manager: PositionManager | None = None,
    ) -> None:
        self._bot_id = bot_id
        self._pipeline = pipeline
        self._executor = executor
        self._position_manager = position_manager

    @property
    def bot_id(self) -> str:
        return self._bot_id

    async def run(self) -> dict:
        """Run one complete lifecycle: analyze -> execute -> return.

        Returns:
            Result dict with bot_id, action, conviction, order_result,
            duration_ms, status ("OK" or "ERROR"), and error if any.
        """
        start = time.perf_counter()

        try:
            # 1. Run analysis pipeline
            action = await self._pipeline.run_cycle()

            # 2. Execute if action requires it
            order_result = None
            if action.action in ("LONG", "SHORT", "ADD_LONG", "ADD_SHORT", "CLOSE_ALL"):
                symbol = self._pipeline._config.symbol
                order_result = await self._executor.execute(action, symbol)

                # 3. Register position with Sentinel if opened
                if (
                    order_result.success
                    and action.action in ("LONG", "SHORT")
                    and self._position_manager is not None
                    and action.sl_price
                ):
                    atr = float(self._pipeline._config.account_balance * 0.01) if action.atr_multiplier is None else 0.0
                    # Get ATR from market data indicators if possible
                    direction = "long" if action.action == "LONG" else "short"
                    self._position_manager.register_position(
                        symbol=symbol,
                        direction=direction,
                        entry_price=order_result.fill_price or 0.0,
                        sl_price=action.sl_price,
                        atr=action.atr_multiplier or 0.0,
                    )

            elapsed = (time.perf_counter() - start) * 1000

            result = {
                "bot_id": self._bot_id,
                "status": "OK",
                "action": action.action,
                "conviction_score": action.conviction_score,
                "order_result": order_result.to_dict() if order_result else None,
                "reasoning": action.reasoning,
                "duration_ms": round(elapsed, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"TraderBot {self._bot_id}: {action.action} "
                f"(conviction={action.conviction_score:.2f}, {elapsed:.0f}ms)"
            )
            return result

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"TraderBot {self._bot_id}: ERROR — {e}", exc_info=True)
            return {
                "bot_id": self._bot_id,
                "status": "ERROR",
                "action": "SKIP",
                "conviction_score": 0.0,
                "order_result": None,
                "reasoning": f"TraderBot error: {e}",
                "duration_ms": round(elapsed, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
