"""Trade listing API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.auth import get_current_user
from api.dependencies import get_trade_repo
from api.schemas import ErrorResponse, TradeListResponse, TradeResponse
from storage.repositories.base import TradeRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/trades", tags=["trades"])


def _trade_to_response(trade: dict) -> TradeResponse:
    """Convert a trade dict from the repository to a TradeResponse."""
    return TradeResponse(
        id=trade["id"],
        user_id=trade.get("user_id", ""),
        bot_id=trade.get("bot_id", ""),
        symbol=trade.get("symbol", ""),
        timeframe=trade.get("timeframe", ""),
        direction=trade.get("direction", ""),
        entry_price=trade.get("entry_price"),
        exit_price=trade.get("exit_price"),
        size=trade.get("size"),
        pnl=trade.get("pnl"),
        r_multiple=trade.get("r_multiple"),
        entry_time=trade.get("entry_time"),
        exit_time=trade.get("exit_time"),
        exit_reason=trade.get("exit_reason"),
        conviction_score=trade.get("conviction_score"),
        engine_version=trade.get("engine_version"),
        status=trade.get("status", "unknown"),
    )


@router.get(
    "",
    response_model=TradeListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def list_trades(
    bot_id: str | None = Query(default=None, description="Filter by bot ID"),
    symbol: str | None = Query(default=None, description="Filter by symbol"),
    limit: int = Query(default=50, ge=1, le=500),
    user_id: str = Depends(get_current_user),
    trade_repo: TradeRepository = Depends(get_trade_repo),
) -> TradeListResponse:
    """List recent trades with optional filters."""
    # Fetch trades — filter by bot_id if provided
    if bot_id:
        trades = await trade_repo.get_trades_by_bot(bot_id, limit=limit)
    else:
        # Get all trades for this user via open positions + closed
        # The repository doesn't have a get_all_by_user method,
        # so we get trades by bot from all user bots.
        # For now, use the trade_repo directly with bot_id filter.
        trades = await trade_repo.get_trades_by_bot("", limit=limit)

    # Apply user_id filter (multi-tenant isolation)
    trades = [t for t in trades if t.get("user_id") == user_id]

    # Apply symbol filter
    if symbol:
        trades = [t for t in trades if t.get("symbol") == symbol]

    # Apply limit
    trades = trades[:limit]

    return TradeListResponse(
        trades=[_trade_to_response(t) for t in trades],
        count=len(trades),
    )


@router.get(
    "/{trade_id}",
    response_model=TradeResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_trade(
    trade_id: str,
    user_id: str = Depends(get_current_user),
    trade_repo: TradeRepository = Depends(get_trade_repo),
) -> TradeResponse:
    """Get full trade detail including cycle data."""
    trade = await trade_repo.get_trade(trade_id)
    if trade is None or trade.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade {trade_id} not found.",
        )

    return _trade_to_response(trade)
