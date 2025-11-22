from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "20241121_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Text(), nullable=False, index=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_conversation_messages_session_created",
        "conversation_messages",
        ["session_id", "created_at"],
    )

    op.create_table(
        "session_summaries",
        sa.Column("session_id", sa.Text(), primary_key=True),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("summary_embedding", Vector(1536), nullable=True),
        sa.Column("last_message_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=False),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("session_summaries")
    op.drop_index("ix_conversation_messages_session_created", table_name="conversation_messages")
    op.drop_table("conversation_messages")

