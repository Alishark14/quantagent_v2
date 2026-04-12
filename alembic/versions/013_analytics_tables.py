"""Add sentinel_events and llm_calls analytics tables.

``sentinel_events`` persists every SetupDetected emission and SKIP
decision so Sentinel behavior is queryable after restart (previously
only in log files).

``llm_calls`` tracks every individual LLM API call with agent name,
model, tokens, latency, cost, and cache hit — enables per-agent cost
analysis and latency monitoring.

Both tables are write-heavy, read-rarely (analytics queries), so indexes
are on the most common query patterns only.

Revision ID: 013
Revises: 012
Create Date: 2026-04-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sentinel_events",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("timeframe", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("readiness_score", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("triggers_today", sa.Integer(), nullable=True),
        sa.Column("cooldown_remaining_s", sa.Float(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("is_shadow", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("ix_sentinel_events_symbol", "sentinel_events", ["symbol", sa.text("timestamp DESC")])
    op.create_index("ix_sentinel_events_type", "sentinel_events", ["event_type", sa.text("timestamp DESC")])

    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("cycle_id", sa.Text(), nullable=True),
        sa.Column("bot_id", sa.Text(), nullable=True),
        sa.Column("agent_name", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("is_shadow", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("ix_llm_calls_cycle", "llm_calls", ["cycle_id"])
    op.create_index("ix_llm_calls_agent", "llm_calls", ["agent_name", sa.text("timestamp DESC")])
    op.create_index("ix_llm_calls_model", "llm_calls", ["model"])


def downgrade() -> None:
    op.drop_table("llm_calls")
    op.drop_table("sentinel_events")
