"""Open positions API endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status

from api.auth import get_current_user
from api.dependencies import get_bot_repo, get_trade_repo
from api.schemas import ErrorResponse, PositionListResponse, PositionResponse
from storage.repositories.base import BotRepository, TradeRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/positions", tags=["positions"])


@router.get(
    "",
    response_model=PositionListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def list_positions(
    user_id: str = Depends(get_current_user),
    trade_repo: TradeRepository = Depends(get_trade_repo),
    bot_repo: BotRepository = Depends(get_bot_repo),
) -> PositionListResponse:
    """Get all open positions across all bots for the authenticated user."""
    # Get all user's bots, then collect open positions from each
    bots = await bot_repo.get_bots_by_user(user_id)

    positions: list[PositionResponse] = []
    for bot in bots:
        bot_id = bot["id"]
        open_trades = await trade_repo.get_open_positions(user_id, bot_id)
        for trade in open_trades:
            positions.append(PositionResponse(
                symbol=trade.get("symbol", ""),
                direction=trade.get("direction", ""),
                size=trade.get("size", 0.0) or 0.0,
                entry_price=trade.get("entry_price", 0.0) or 0.0,
                unrealized_pnl=trade.get("pnl", 0.0) or 0.0,
                leverage=None,
                bot_id=bot_id,
            ))

    return PositionListResponse(
        positions=positions,
        count=len(positions),
    )
