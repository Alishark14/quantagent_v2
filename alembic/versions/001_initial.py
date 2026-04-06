"""Initial schema: bots, trades, cycles, rules, cross_bot_signals.

Revision ID: 001
Revises: None
Create Date: 2026-04-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── bots ──
    op.create_table(
        "bots",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("exchange", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("config_json", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_health", sa.JSON, nullable=True),
    )
    op.create_index("ix_bots_user_id", "bots", ["user_id"])
    op.create_index("ix_bots_status", "bots", ["status"])
    op.create_index("ix_bots_user_status", "bots", ["user_id", "status"])

    # ── trades ──
    op.create_table(
        "trades",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("bot_id", sa.Text, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=True),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("size", sa.Float, nullable=True),
        sa.Column("pnl", sa.Float, nullable=True),
        sa.Column("r_multiple", sa.Float, nullable=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_reason", sa.Text, nullable=True),
        sa.Column("conviction_score", sa.Float, nullable=True),
        sa.Column("engine_version", sa.Text, nullable=True),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
    )
    op.create_index("ix_trades_bot_id", "trades", ["bot_id"])
    op.create_index("ix_trades_user_id", "trades", ["user_id"])
    op.create_index("ix_trades_symbol", "trades", ["symbol"])
    op.create_index("ix_trades_status", "trades", ["status"])
    op.create_index("ix_trades_bot_status", "trades", ["bot_id", "status"])
    op.create_index("ix_trades_entry_time", "trades", ["entry_time"])

    # ── cycles ──
    op.create_table(
        "cycles",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("bot_id", sa.Text, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("indicators_json", sa.JSON, nullable=True),
        sa.Column("signals_json", sa.JSON, nullable=True),
        sa.Column("conviction_json", sa.JSON, nullable=True),
        sa.Column("action", sa.Text, nullable=True),
        sa.Column("conviction_score", sa.Float, nullable=True),
    )
    op.create_index("ix_cycles_bot_id", "cycles", ["bot_id"])
    op.create_index("ix_cycles_timestamp", "cycles", ["timestamp"])
    op.create_index("ix_cycles_bot_timestamp", "cycles", ["bot_id", "timestamp"])

    # ── rules ──
    op.create_table(
        "rules",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("rule_text", sa.Text, nullable=False),
        sa.Column("score", sa.Integer, nullable=False, server_default="0"),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_rules_symbol_tf", "rules", ["symbol", "timeframe"])
    op.create_index("ix_rules_active", "rules", ["active"])

    # ── cross_bot_signals ──
    op.create_table(
        "cross_bot_signals",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("direction", sa.Text, nullable=False),
        sa.Column("conviction", sa.Float, nullable=False),
        sa.Column("bot_id", sa.Text, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cross_bot_user_symbol", "cross_bot_signals", ["user_id", "symbol"])
    op.create_index("ix_cross_bot_timestamp", "cross_bot_signals", ["timestamp"])


def downgrade() -> None:
    op.drop_table("cross_bot_signals")
    op.drop_table("rules")
    op.drop_table("cycles")
    op.drop_table("trades")
    op.drop_table("bots")
