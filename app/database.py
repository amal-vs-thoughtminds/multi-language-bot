from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Base class for declarative SQLAlchemy models."""


def _create_engine() -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )


engine: AsyncEngine = _create_engine()
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


