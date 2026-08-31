"""Add per-chat web search preference and per-message generation time

Revision ID: 0004_web_search_and_timing
Revises: 0003_progress_and_usage
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_web_search_and_timing"
down_revision: Union[str, None] = "0003_progress_and_usage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("chats") as batch_op:
        batch_op.add_column(
            sa.Column(
                "web_search", sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )

    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("duration_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_column("duration_ms")

    with op.batch_alter_table("chats") as batch_op:
        batch_op.drop_column("web_search")
