"""Add cost-tracking columns to trades and cycles.

Trades gain ``raw_pnl`` (PnL before fees) and ``trading_fee`` (round-trip
exchange fee) so the engine can report gross vs net performance and
attribute slippage/cost drag per position.

Cycles gain ``llm_input_tokens``, ``llm_output_tokens``, and
``llm_cost_usd`` so every analysis cycle records its LLM spend.  This
feeds the per-bot and per-portfolio cost dashboards and lets the system
enforce per-cycle budget caps.

Existing rows backfill to NULL — downstream code treats NULL as
"cost data unavailable" and omits those rows from cost aggregations.

Revision ID: 008
Revises: 007
Create Date: 2026-04-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- trades: gross PnL and round-trip fee --
    op.add_column(
        "trades",
        sa.Column("raw_pnl", sa.Float(), nullable=True),
    )
    op.add_column(
        "trades",
        sa.Column("trading_fee", sa.Float(), nullable=True),
    )

    # -- cycles: LLM token usage and cost --
    op.add_column(
        "cycles",
        sa.Column("llm_input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "cycles",
        sa.Column("llm_output_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "cycles",
        sa.Column("llm_cost_usd", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cycles", "llm_cost_usd")
    op.drop_column("cycles", "llm_output_tokens")
    op.drop_column("cycles", "llm_input_tokens")
    op.drop_column("trades", "trading_fee")
    op.drop_column("trades", "raw_pnl")
