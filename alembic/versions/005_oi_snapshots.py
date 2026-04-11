"""Persist OI snapshots so the CryptoFlowProvider deque survives restarts.

The provider keeps an in-memory rolling deque of recent open-interest
snapshots per symbol so it can compute ``oi_change_*`` deltas without
asking the exchange for history (which most exchanges don't expose
anyway). The deque resets every time the process restarts, which means
every fresh boot pays a multi-hour cold-start penalty before the
divergence / accumulation rules in FlowSignalAgent can fire. With this
table the provider bulk-loads the recent window into its deques on
startup and writes every fresh snapshot back, so a restart is
effectively free after the first 24 hours of uptime.

Cleanup is the caller's responsibility — main.py runs an hourly task
that calls ``cleanup_older_than(86_400)`` to keep the table small.

Revision ID: 005
Revises: 004
Create Date: 2026-04-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "oi_snapshots",
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("oi_value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "timestamp"),
    )
    op.create_index(
        "ix_oi_snapshots_symbol_time",
        "oi_snapshots",
        ["symbol", sa.text("timestamp DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_oi_snapshots_symbol_time", table_name="oi_snapshots")
    op.drop_table("oi_snapshots")
