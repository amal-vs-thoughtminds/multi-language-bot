from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from openai import AsyncOpenAI
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.models import ConversationMessage, SessionSummary


@dataclass(slots=True)
class ConversationSnippet:
    role: str
    content: str


class ConversationMemory:
    """Session-scoped conversational memory leveraging pgvector."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        client: AsyncOpenAI,
        run_llm,
    ) -> None:
        self._sessions = session_factory
        self._client = client
        self._run_llm = run_llm
        self._embedding_model = settings.embedding_model
        self._recent_turn_limit = settings.conversation_recent_turns
        self._semantic_turn_limit = settings.conversation_semantic_turns
        self._summary_trigger_turns = settings.summary_trigger_turns
        self._summary_max_turns = settings.summary_max_turns

    async def record_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        trigger_summary: bool = False,
    ) -> None:
        embedding = await self._embed_text(content)
        async with self._sessions() as session:
            message = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content,
                extra=metadata,
                embedding=embedding,
            )
            session.add(message)
            await session.commit()
        if trigger_summary:
            await self._maybe_summarize(session_id)

    async def build_context(
        self,
        *,
        session_id: str,
        query_text: str,
    ) -> tuple[str | None, list[ConversationSnippet]]:
        async with self._sessions() as session:
            summary_row = await session.get(SessionSummary, session_id)
            summary_text = summary_row.summary if summary_row and summary_row.summary else None

            recent_stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(self._recent_turn_limit)
            )
            recent_result = await session.execute(recent_stmt)
            recent_messages = list(recent_result.scalars().all())
            recent_messages.reverse()

            ordered_messages: dict[int, ConversationSnippet] = {
                msg.id: ConversationSnippet(role=msg.role, content=msg.content)
                for msg in recent_messages
            }

            if query_text.strip() and self._semantic_turn_limit > 0:
                query_vec = await self._embed_text(query_text)
                semantic_stmt = (
                    select(ConversationMessage)
                    .where(
                        ConversationMessage.session_id == session_id,
                        ConversationMessage.embedding.is_not(None),
                    )
                    .order_by(ConversationMessage.embedding.cosine_distance(query_vec))
                    .limit(self._semantic_turn_limit)
                )
                semantic_result = await session.execute(semantic_stmt)
                for msg in semantic_result.scalars():
                    ordered_messages.setdefault(
                        msg.id,
                        ConversationSnippet(role=msg.role, content=msg.content),
                    )

        snippets = list(ordered_messages.values())
        return summary_text, snippets

    async def get_history(
        self,
        *,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve full conversation history for a session."""
        async with self._sessions() as session:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at.asc())
            )
            if limit:
                stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            messages = result.scalars().all()
            
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.extra,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                }
                for msg in messages
            ]

    async def _maybe_summarize(self, session_id: str) -> None:
        async with self._sessions() as session:
            summary_row = await session.get(SessionSummary, session_id)
            last_message_id = summary_row.last_message_id if summary_row else 0

            stmt: Select[tuple[ConversationMessage]] = (
                select(ConversationMessage)
                .where(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at)
            )
            if last_message_id:
                stmt = stmt.where(ConversationMessage.id > last_message_id)

            result = await session.execute(stmt)
            new_messages = result.scalars().all()
            if len(new_messages) < self._summary_trigger_turns:
                return

            trimmed_messages = new_messages[-self._summary_max_turns :]
            summary_prompt = self._format_summary_prompt(
                existing_summary=summary_row.summary if summary_row else "",
                new_messages=trimmed_messages,
            )

            updated_summary = await self._run_llm(
                [
                    {
                        "role": "system",
                        "content": (
                            "You maintain a running summary of a fish market conversation. "
                            "Preserve customer intent, requested fish types, negotiated prices, "
                            "and outstanding action items."
                        ),
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.2,
            )

            summary_embedding = await self._embed_text(updated_summary) if updated_summary else None
            last_id = trimmed_messages[-1].id

            now = datetime.utcnow()
            if summary_row:
                summary_row.summary = updated_summary
                summary_row.summary_embedding = summary_embedding
                summary_row.last_message_id = last_id
                summary_row.updated_at = now
            else:
                summary_row = SessionSummary(
                    session_id=session_id,
                    summary=updated_summary,
                    summary_embedding=summary_embedding,
                    last_message_id=last_id,
                    updated_at=now,
                )
                session.add(summary_row)

            await session.commit()

    def _format_summary_prompt(
        self,
        *,
        existing_summary: str,
        new_messages: Sequence[ConversationMessage],
    ) -> str:
        summary = existing_summary or "None"
        turns = "\n".join(f"{msg.role.title()}: {msg.content}" for msg in new_messages)
        return (
            f"Current summary:\n{summary}\n\n"
            f"New conversation turns:\n{turns}\n\n"
            "Update the summary to reflect the entire conversation so far. "
            "Include pending quantities, confirmed orders, or follow-up questions."
        )

    async def _embed_text(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding  # type: ignore[return-value]


