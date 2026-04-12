"""Persistent COT snapshot cache so the CommodityFlowProvider survives restarts.

The CFTC Commitment of Traders report drops once a week (Friday evening
for data as-of the prior Tuesday). ``CommodityFlowProvider`` computes
52-week percentiles / divergences off a rolling history, and without
persistence every process restart would lose the history — the
percentile math silently broken until 52 weeks of live uptime.

One row per (symbol, report_date); Alembic-managed cleanup is deferred
to application code (``cleanup_older_than_weeks``) so ops can tune the
retention window without a migration.

Revision ID: 006
Revises: 005
Create Date: 2026-04-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "cot_cache",
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("managed_money_net", sa.Float(), nullable=True),
        sa.Column("commercial_net", sa.Float(), nullable=True),
        sa.Column("total_oi", sa.Float(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("symbol", "report_date"),
    )
    op.create_index(
        "ix_cot_cache_symbol_date",
        "cot_cache",
        ["symbol", sa.text("report_date DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_cot_cache_symbol_date", table_name="cot_cache")
    op.drop_table("cot_cache")
