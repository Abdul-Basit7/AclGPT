"""Add ingest progress and per-message token usage

Revision ID: 0003_progress_and_usage
Revises: 0002_oauth_accounts
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_progress_and_usage"
down_revision: Union[str, None] = "0002_oauth_accounts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column(
                "chunks_embedded", sa.Integer(), nullable=False, server_default="0"
            )
        )

    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("input_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("output_tokens", sa.Integer(), nullable=True))

    # Existing ready documents are fully embedded by definition.
    op.execute(
        "UPDATE documents SET chunks_embedded = chunk_count WHERE status = 'ready'"
    )


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_column("output_tokens")
        batch_op.drop_column("input_tokens")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("chunks_embedded")
