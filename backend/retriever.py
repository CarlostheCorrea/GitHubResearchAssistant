from __future__ import annotations

import logging
from pathlib import Path

from backend.config import Settings
from backend.embedder import OpenAIEmbedder
from backend.models import ChunkRecord, RepoDescriptor
from backend.utils import tokenize_for_matching
from backend.vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)

ROLE_KEYWORDS = {
    "training": {"train", "training", "trainer", "epoch", "loss", "optimizer", "fit", "backward"},
    "inference": {"infer", "inference", "predict", "generate", "forward", "serve", "endpoint"},
    "config": {"config", "configuration", "settings", "yaml", "json", "toml", "args", "parameters"},
    "data_loading": {"data", "dataset", "loader", "dataloader", "preprocess", "tokenizer"},
    "entrypoint": {"main", "entry", "start", "server", "app", "cli"},
}


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: OpenAIEmbedder,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        repo: RepoDescriptor,
        question: str,
        top_k: int | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        query_embedding = self.embedder.embed_query(question)
        candidates = self.vector_store.query(
            repo_id=repo.repo_id,
            query_embedding=query_embedding,
            top_k=self.settings.vector_query_k,
        )
        reranked = self._rerank(question, candidates)
        final_results = reranked[: top_k or self.settings.answer_context_k]
        logger.info(
            "Top retrieved chunks for '%s': %s",
            question,
            [
                {
                    "file": chunk.file_path,
                    "lines": (chunk.start_line, chunk.end_line),
                    "type": chunk.chunk_type,
                    "score": round(score, 4),
                }
                for chunk, score in final_results
            ],
        )
        return final_results

    def _rerank(
        self,
        question: str,
        candidates: list[tuple[ChunkRecord, float]],
    ) -> list[tuple[ChunkRecord, float]]:
        question_tokens = tokenize_for_matching(question)
        ranked: list[tuple[ChunkRecord, float]] = []

        for chunk, distance in candidates:
            semantic_score = 1 / (1 + max(distance, 0))
            lexical_score = self._keyword_boost(question_tokens, chunk)
            ranked.append((chunk, semantic_score + lexical_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _keyword_boost(self, question_tokens: set[str], chunk: ChunkRecord) -> float:
        path_tokens = tokenize_for_matching(chunk.file_path.replace(".", " "))
        basename_tokens = tokenize_for_matching(Path(chunk.file_path).name.replace(".", " "))
        symbol_tokens = tokenize_for_matching(chunk.symbol_name or "")
        summary_tokens = tokenize_for_matching((chunk.short_summary or "") + " " + (chunk.file_role or ""))
        body_tokens = tokenize_for_matching(chunk.text[:1200])

        score = 0.0

        overlap = len(question_tokens & path_tokens)
        score += min(overlap * 0.12, 0.36)

        basename_overlap = len(question_tokens & basename_tokens)
        score += min(basename_overlap * 0.18, 0.36)

        symbol_overlap = len(question_tokens & symbol_tokens)
        score += min(symbol_overlap * 0.2, 0.4)

        summary_overlap = len(question_tokens & summary_tokens)
        score += min(summary_overlap * 0.08, 0.24)

        body_overlap = len(question_tokens & body_tokens)
        score += min(body_overlap * 0.02, 0.16)

        for role, keywords in ROLE_KEYWORDS.items():
            if question_tokens & keywords and chunk.file_role == role:
                score += 0.28

        if chunk.chunk_type in {"python_function", "python_method", "python_class", "code_block"}:
            score += 0.04
        if chunk.chunk_type == "file_summary" and {"main", "components", "overview", "architecture"} & question_tokens:
            score += 0.12

        return score
