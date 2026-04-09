"""Add shadow mode columns to bots, trades, cycles + live_* views.

Replaces the separate `quantagent_shadow` database with a per-row shadow
flag on the shared schema. This lets a single BotRunner manage both live
and shadow bots from one database, and lets repository read methods
filter shadow data out of all production queries by default.

Schema changes:
  - bots: add `is_shadow BOOLEAN NOT NULL DEFAULT false`
          add `mode VARCHAR(10) NOT NULL DEFAULT 'live'`  ('live' | 'shadow')
  - trades: add `is_shadow BOOLEAN NOT NULL DEFAULT false`
  - cycles: add `is_shadow BOOLEAN NOT NULL DEFAULT false`
  - views:  CREATE VIEW live_trades / live_cycles for ergonomic
            production-only reads.

Existing rows backfill to `is_shadow=false` / `mode='live'` via the
column defaults — there is nothing to migrate inside this revision. The
9 production shadow bots living in the legacy `quantagent_shadow` DB are
copied across by `scripts/migrate_shadow_bots.py`, run once after this
upgrade lands on the server.

Revision ID: 003
Revises: 002
Create Date: 2026-04-09
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── bots ──
    op.add_column(
        "bots",
        sa.Column(
            "is_shadow",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "bots",
        sa.Column(
            "mode",
            sa.String(length=10),
            nullable=False,
            server_default="live",
        ),
    )

    # ── trades ──
    op.add_column(
        "trades",
        sa.Column(
            "is_shadow",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    # ── cycles ──
    op.add_column(
        "cycles",
        sa.Column(
            "is_shadow",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    # ── live_* views ──
    op.execute(
        "CREATE OR REPLACE VIEW live_trades AS "
        "SELECT * FROM trades WHERE is_shadow = false"
    )
    op.execute(
        "CREATE OR REPLACE VIEW live_cycles AS "
        "SELECT * FROM cycles WHERE is_shadow = false"
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS live_cycles")
    op.execute("DROP VIEW IF EXISTS live_trades")
    op.drop_column("cycles", "is_shadow")
    op.drop_column("trades", "is_shadow")
    op.drop_column("bots", "mode")
    op.drop_column("bots", "is_shadow")
