from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(index=True)
    role: Mapped[str]
    content: Mapped[str] = mapped_column(Text)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        nullable=False,
    )


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    session_id: Mapped[str] = mapped_column(primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    last_message_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )