"""Add instrument type, exchange, leverage, and margin columns.

Future-proofs the schema for multi-exchange and multi-instrument support.
All columns have sensible defaults so existing rows are backfilled
automatically (no data migration needed).

Trades:
  - instrument_type TEXT DEFAULT 'perpetual'
  - exchange TEXT DEFAULT 'hyperliquid'
  - leverage FLOAT (nullable, no default — read from adapter when available)
  - margin_type TEXT DEFAULT 'cross'

Cycles:
  - exchange TEXT DEFAULT 'hyperliquid'

Bots:
  - instrument_type TEXT DEFAULT 'perpetual'

Revision ID: 011
Revises: 010
Create Date: 2026-04-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- trades --
    op.add_column(
        "trades",
        sa.Column("instrument_type", sa.Text(), server_default="perpetual", nullable=True),
    )
    op.add_column(
        "trades",
        sa.Column("exchange", sa.Text(), server_default="hyperliquid", nullable=True),
    )
    op.add_column(
        "trades",
        sa.Column("leverage", sa.Float(), nullable=True),
    )
    op.add_column(
        "trades",
        sa.Column("margin_type", sa.Text(), server_default="cross", nullable=True),
    )

    # -- cycles --
    op.add_column(
        "cycles",
        sa.Column("exchange", sa.Text(), server_default="hyperliquid", nullable=True),
    )

    # -- bots --
    op.add_column(
        "bots",
        sa.Column("instrument_type", sa.Text(), server_default="perpetual", nullable=True),
    )


def downgrade() -> None:
    op.drop_column("bots", "instrument_type")
    op.drop_column("cycles", "exchange")
    op.drop_column("trades", "margin_type")
    op.drop_column("trades", "leverage")
    op.drop_column("trades", "exchange")
    op.drop_column("trades", "instrument_type")
