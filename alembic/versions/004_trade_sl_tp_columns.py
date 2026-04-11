"""Add sl_price + tp_price columns to trades.

Shadow data collection needs to monitor open positions for SL/TP hits
between TraderBot lifecycles — Sentinel scans every cycle, queries open
shadow trades, and closes them when the candle high/low breaches a
level. The Sentinel can only do that if it knows each trade's SL/TP
levels, which DecisionAgent already produces but the schema never
persisted.

Existing trades backfill to NULL — the Sentinel monitor treats NULL as
"no level set" and skips those rows, so live trades (which have native
exchange-side SL/TP orders) are unaffected.

Revision ID: 004
Revises: 003
Create Date: 2026-04-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("sl_price", sa.Float(), nullable=True),
    )
    op.add_column(
        "trades",
        sa.Column("tp_price", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trades", "tp_price")
    op.drop_column("trades", "sl_price")
