"""Add trade quality columns for regime and sizing diagnostics.

* ``tp2_price`` — the full-RR target price (profile.rr_min × risk).
  Currently only ``tp_price`` is stored which collapses tp1/tp2 into
  one column; having both lets analysts compare 1:1 vs full-RR outcomes.

* ``atr_multiplier`` — the regime-adjusted ATR multiplier used to
  compute SL distance.  Diagnosis tool for "why was the stop so wide?"

* ``risk_weight`` — the conviction-tier sizing weight (0.75 / 1.0 /
  1.15 / 1.3) that PortfolioRiskManager multiplies the base risk by.

* ``regime`` — the ConvictionAgent's detected market regime at trade
  time (TRENDING_UP / TRENDING_DOWN / RANGING / HIGH_VOLATILITY /
  BREAKOUT).  Enables "win rate by regime" queries.

All nullable so existing trades backfill to NULL.

Revision ID: 010
Revises: 009
Create Date: 2026-04-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trades", sa.Column("tp2_price", sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("atr_multiplier", sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("risk_weight", sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("regime", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("trades", "regime")
    op.drop_column("trades", "risk_weight")
    op.drop_column("trades", "atr_multiplier")
    op.drop_column("trades", "tp2_price")
