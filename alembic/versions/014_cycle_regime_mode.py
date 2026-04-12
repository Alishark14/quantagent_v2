"""Add regime and mode columns to cycles table.

* ``regime`` (Text, nullable) — the ConvictionAgent's detected market
  regime at cycle time (TRENDING_UP/DOWN, RANGING, HIGH_VOLATILITY,
  BREAKOUT).  Previously buried in ``conviction_json``; promoting to a
  standalone column enables ``GROUP BY regime`` analytics.

* ``mode`` (VARCHAR(10), DEFAULT 'live') — explicit 3-way mode
  (live/paper/shadow) mirroring the bots table pattern.  Keeps
  ``is_shadow`` for backward compatibility.

Backfills existing rows: shadow cycles get mode='shadow', live get
mode='live'. regime stays NULL for old rows (can't reconstruct).

CHECK constraint enforces mode/is_shadow consistency (PostgreSQL only;
SQLite ignores ALTER TABLE ADD CONSTRAINT).

Revision ID: 014
Revises: 013
Create Date: 2026-04-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "cycles",
        sa.Column("regime", sa.Text(), nullable=True),
    )
    op.add_column(
        "cycles",
        sa.Column("mode", sa.String(10), server_default="live", nullable=True),
    )

    # Backfill existing rows
    op.execute("UPDATE cycles SET mode = 'shadow' WHERE is_shadow = true AND mode IS NULL")
    op.execute("UPDATE cycles SET mode = 'live' WHERE is_shadow = false AND mode IS NULL")

    # CHECK constraint (PostgreSQL only)
    try:
        op.execute(
            """ALTER TABLE cycles ADD CONSTRAINT ck_cycles_mode_shadow_consistent
               CHECK (
                   (mode = 'shadow' AND is_shadow = true) OR
                   (mode = 'paper'  AND is_shadow = true) OR
                   (mode = 'live'   AND is_shadow = false)
               )"""
        )
    except Exception:
        pass  # SQLite doesn't support ALTER TABLE ADD CONSTRAINT


def downgrade() -> None:
    try:
        op.execute("ALTER TABLE cycles DROP CONSTRAINT ck_cycles_mode_shadow_consistent")
    except Exception:
        pass
    op.drop_column("cycles", "mode")
    op.drop_column("cycles", "regime")
