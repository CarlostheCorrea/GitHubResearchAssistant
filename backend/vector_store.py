from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override

from backend.config import Settings
from backend.models import ChunkRecord
from backend.utils import safe_collection_name


class NoOpProductTelemetryClient(ProductTelemetryClient):
    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None


class ChromaVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(settings.chroma_dir),
                chroma_product_telemetry_impl="backend.vector_store.NoOpProductTelemetryClient",
                chroma_telemetry_impl="backend.vector_store.NoOpProductTelemetryClient",
            ),
        )

    def repo_has_data(self, repo_id: str) -> bool:
        return self._get_collection(repo_id).count() > 0

    def upsert_chunks(
        self,
        repo_id: str,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")

        collection = self._get_collection(repo_id)
        collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.chroma_metadata() for chunk in chunks],
            embeddings=embeddings,
        )

    def delete_repo(self, repo_id: str) -> bool:
        collection_name = safe_collection_name(repo_id)
        existing_names = set(self.client.list_collections())
        if collection_name not in existing_names:
            return False
        self.client.delete_collection(name=collection_name)
        return True

    def delete_all(self) -> int:
        collection_names = list(self.client.list_collections())
        deleted = 0
        for collection_name in collection_names:
            self.client.delete_collection(name=collection_name)
            deleted += 1
        return deleted

    def query(self, repo_id: str, query_embedding: list[float], top_k: int) -> list[tuple[ChunkRecord, float]]:
        collection = self._get_collection(repo_id)
        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        pairs: list[tuple[ChunkRecord, float]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            pairs.append((self._chunk_from_query_result(document, metadata), float(distance)))
        return pairs

    def _get_collection(self, repo_id: str) -> Collection:
        return self.client.get_or_create_collection(
            name=safe_collection_name(repo_id),
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_from_query_result(self, document: str, metadata: dict[str, Any]) -> ChunkRecord:
        start_line = int(metadata.get("start_line") or 0) or None
        end_line = int(metadata.get("end_line") or 0) or None
        symbol_name = metadata.get("symbol_name") or None
        short_summary = metadata.get("short_summary") or None
        file_role = metadata.get("file_role") or None

        return ChunkRecord(
            id=str(metadata.get("chunk_id") or "query-result"),
            repo_id=str(metadata.get("repo_id")),
            repo_name=str(metadata.get("repo_name")),
            file_path=str(metadata.get("file_path")),
            language=str(metadata.get("language")),
            chunk_type=str(metadata.get("chunk_type")),
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=end_line,
            short_summary=short_summary,
            file_role=file_role,
            text=document,
        )
