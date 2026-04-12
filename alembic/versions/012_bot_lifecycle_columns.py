"""Add bot lifecycle columns and mode/is_shadow CHECK constraint.

* ``is_active`` (Boolean, NOT NULL, DEFAULT true) — replaces the
  ``status = 'active'`` string check with a proper boolean for faster
  index scans and clearer semantics.  Existing rows default to true.

* ``deactivated_at`` (TimestampTZ, nullable) — set by
  ``deactivate_bot()`` so operators can see WHEN a bot was disabled.

* ``last_cycle_at`` (TimestampTZ, nullable) — stamped after every
  analysis cycle so the dedup logic can prefer the most-recently-active
  bot when multiple entries exist for the same symbol.

* Composite index ``ix_bots_active_mode`` on (is_active, mode) for the
  common ``get_active_bots_by_mode`` query path.

* CHECK constraint on mode/is_shadow consistency:
    (mode='shadow' AND is_shadow=true) OR
    (mode='paper'  AND is_shadow=true) OR
    (mode='live'   AND is_shadow=false)
  This catches accidental inconsistencies without removing the
  redundant ``is_shadow`` column (which existing queries still read).

Revision ID: 012
Revises: 011
Create Date: 2026-04-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "bots",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "bots",
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "bots",
        sa.Column("last_cycle_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_bots_active_mode", "bots", ["is_active", "mode"])

    # CHECK constraint for mode/is_shadow consistency.
    # Only PostgreSQL supports named CHECK constraints via ALTER TABLE;
    # SQLite ignores this (its CHECK lives in CREATE TABLE DDL).
    try:
        op.execute(
            """ALTER TABLE bots ADD CONSTRAINT ck_bots_mode_shadow_consistent
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
        op.execute("ALTER TABLE bots DROP CONSTRAINT ck_bots_mode_shadow_consistent")
    except Exception:
        pass
    op.drop_index("ix_bots_active_mode", table_name="bots")
    op.drop_column("bots", "last_cycle_at")
    op.drop_column("bots", "deactivated_at")
    op.drop_column("bots", "is_active")
