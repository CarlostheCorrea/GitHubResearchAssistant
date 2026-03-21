from __future__ import annotations

import logging

from openai import OpenAI

from backend.config import Settings
from backend.models import ChunkRecord
from backend.utils import looks_like_placeholder_secret


logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[list[float]]:
        if not chunks:
            return []

        client = self._get_client()
        inputs = [chunk.embedding_input() for chunk in chunks]
        embeddings: list[list[float]] = []

        for index in range(0, len(inputs), self.settings.embedding_batch_size):
            batch = inputs[index : index + self.settings.embedding_batch_size]
            response = client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)

        logger.info("Generated embeddings for %s chunks", len(chunks))
        return embeddings

    def embed_query(self, question: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=[question],
        )
        return response.data[0].embedding

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for embeddings and question answering.")
            if looks_like_placeholder_secret(self.settings.openai_api_key):
                raise RuntimeError(
                    "OPENAI_API_KEY is still set to the placeholder value. Update .env with a real OpenAI API key and restart uvicorn."
                )
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
