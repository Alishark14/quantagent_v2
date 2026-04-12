"""Persistent RegSHO short-volume cache for EquityFlowProvider.

FINRA drops a daily off-exchange short-volume file around 6 PM ET for
every US equity. ``EquityFlowProvider`` needs the 20 most recent days
per ticker to compute a Z-score, and a process restart would lose the
rolling window without persistence.

One row per (symbol, trade_date); application code handles retention
via ``cleanup_older_than_days`` so the retention window can be tuned
without a migration.

Revision ID: 007
Revises: 006
Create Date: 2026-04-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "regsho_cache",
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("short_volume", sa.BigInteger(), nullable=True),
        sa.Column("total_volume", sa.BigInteger(), nullable=True),
        sa.Column("short_volume_ratio", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "trade_date"),
    )
    op.create_index(
        "ix_regsho_cache_symbol_date",
        "regsho_cache",
        ["symbol", sa.text("trade_date DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_regsho_cache_symbol_date", table_name="regsho_cache")
    op.drop_table("regsho_cache")
