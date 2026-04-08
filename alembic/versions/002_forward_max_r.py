"""Add forward_max_r column to trades.

The auto-miner needs the maximum favourable excursion (in R-multiples)
that the price would have hit in the N candles after entry. The
Tracking Module's `ForwardMaxRStamper` writes this value once per
closed trade by walking the high-resolution Forward Price Path from
Parquet. The column is nullable because (a) historical trades that
predate this migration won't have it, and (b) trades for symbols /
timeframes that don't have downloaded historical data won't be
backfillable until the data lands.

Revision ID: 002
Revises: 001
Create Date: 2026-04-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("forward_max_r", sa.Float, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trades", "forward_max_r")
