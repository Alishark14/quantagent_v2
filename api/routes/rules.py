"""Reflection rules API endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from api.auth import get_current_user
from api.dependencies import get_rule_repo
from api.schemas import ErrorResponse, RuleListResponse, RuleResponse
from storage.repositories.base import RuleRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/rules", tags=["rules"])


@router.get(
    "",
    response_model=RuleListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def list_rules(
    symbol: str = Query(default="BTC-USDC", description="Filter by symbol"),
    timeframe: str = Query(default="1h", description="Filter by timeframe"),
    user_id: str = Depends(get_current_user),
    rule_repo: RuleRepository = Depends(get_rule_repo),
) -> RuleListResponse:
    """Get active reflection rules filtered by symbol and timeframe."""
    rules = await rule_repo.get_rules(symbol, timeframe)

    rule_responses = []
    for r in rules:
        rule_responses.append(RuleResponse(
            id=r.get("id", ""),
            symbol=r.get("symbol", symbol),
            timeframe=r.get("timeframe", timeframe),
            rule_text=r.get("rule_text", ""),
            score=r.get("score", 0),
            active=r.get("active", True),
            created_at=r.get("created_at", ""),
        ))

    return RuleListResponse(
        rules=rule_responses,
        count=len(rule_responses),
    )
