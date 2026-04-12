"""Add funding_cost to trades and duration_ms to cycles.

Extends the cost-tracking schema from migration 008:

* ``trades.funding_cost`` (Float, nullable) — estimated funding-rate
  cost accrued during the hold period. Computed as
  ``abs(funding_rate * notional * hold_hours)`` at trade close.

* ``cycles.duration_ms`` (Integer, nullable) — wall-clock milliseconds
  for the full analysis pipeline (data → signals → conviction →
  decision), measured via ``time.monotonic()`` in ``run_cycle()``.

Existing rows backfill to NULL.

Revision ID: 009
Revises: 008
Create Date: 2026-04-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("funding_cost", sa.Float(), nullable=True),
    )
    op.add_column(
        "cycles",
        sa.Column("duration_ms", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cycles", "duration_ms")
    op.drop_column("trades", "funding_cost")
