from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from openai import AsyncOpenAI


@dataclass(slots=True)
class FishRecord:
    names: list[str]
    stock: int
    price_per_kg: float

    @property
    def primary_name(self) -> str:
        """Return the first (primary) name, typically English."""
        return self.names[0] if self.names else ""

    @property
    def dense_text(self) -> str:
        all_names = ", ".join(self.names) if len(self.names) > 1 else self.names[0]
        return (
            f"Fish: {all_names}. Available stock: {self.stock} kg. "
            f"Price per kg: ₹{self.price_per_kg}."
        )


class FishVectorStore:
    """In-memory semantic retriever for fish data."""

    def __init__(
        self,
        client: AsyncOpenAI,
        records: Sequence[FishRecord],
        embedding_model: str,
    ) -> None:
        self._client = client
        self._records = list(records)
        self._embedding_model = embedding_model
        self._embeddings: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    async def build(self) -> None:
        """Pre-compute embeddings for all fish records."""
        if not self._records:
            self._embeddings = np.zeros((0, 0))
            self._norms = np.zeros(0)
            return

        payload = [record.dense_text for record in self._records]
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=payload,
        )

        vectors = np.array([np.array(item.embedding, dtype=np.float32) for item in response.data])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._embeddings = vectors / norms
        self._norms = np.squeeze(norms, axis=1)

    def get_all_records(self) -> list[str]:
        """Return all fish descriptions."""
        return [record.dense_text for record in self._records]

    def get_records(self) -> list[FishRecord]:
        """Return the raw fish records."""
        return list(self._records)

    async def query(self, text: str, top_k: int = 3) -> list[str]:
        """Return top fish descriptions relevant to the query."""
        if self._embeddings is None or self._embeddings.size == 0:
            return []

        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm:
            query_vec = query_vec / query_norm

        similarities = np.dot(self._embeddings, query_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        context = []
        for idx in top_indices:
            record = self._records[idx]
            context.append(record.dense_text)
        return context


def load_fish_records(raw: list[dict]) -> list[FishRecord]:
    def sanitize(item: dict) -> FishRecord:
        # Support both old format (fish_name) and new format (fish_names)
        if "fish_names" in item:
            names = [str(n).strip() for n in item["fish_names"] if str(n).strip()]
        elif "fish_name" in item:
            names = [str(item["fish_name"]).strip()]
        else:
            names = []
        
        if not names:
            names = ["unknown"]
        
        return FishRecord(
            names=names,
            stock=int(item.get("fish_stock", 0)),
            price_per_kg=float(item.get("price per kg", 0)),
        )

    return [sanitize(entry) for entry in raw]

