from __future__ import annotations

from openai import OpenAI

from backend.config import Settings
from backend.models import ChunkRecord, RepoSummary
from backend.utils import line_range_label, looks_like_placeholder_secret


SYSTEM_PROMPT = """You answer questions about a GitHub repository using only retrieved repository evidence.

Rules:
- Use only the provided repository context.
- If the evidence is insufficient or ambiguous, say so directly.
- Do not invent files, functions, classes, behavior, or architecture.
- Cite evidence inline with file paths and line ranges in the format [path:start-end].
- Explain relationships across files only when the retrieved context supports that connection.
- Prefer concise, technical answers over generic descriptions.
"""


class QAService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def answer_question(
        self,
        repo_summary: RepoSummary,
        question: str,
        retrieved_chunks: list[tuple[ChunkRecord, float]],
    ) -> str:
        if not retrieved_chunks:
            return "I do not have enough retrieved repository evidence to answer that question."

        client = self._get_client()
        user_prompt = self._build_user_prompt(repo_summary, question, retrieved_chunks)
        response = client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _build_user_prompt(
        self,
        repo_summary: RepoSummary,
        question: str,
        retrieved_chunks: list[tuple[ChunkRecord, float]],
    ) -> str:
        context_blocks = []
        for index, (chunk, score) in enumerate(retrieved_chunks, start=1):
            line_label = line_range_label(chunk.start_line, chunk.end_line)
            context_blocks.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"File: {chunk.file_path}",
                        f"Lines: {line_label}",
                        f"Chunk type: {chunk.chunk_type}",
                        f"Symbol: {chunk.symbol_name or 'n/a'}",
                        f"Role: {chunk.file_role or 'general'}",
                        f"Retriever score: {score:.4f}",
                        f"Summary: {chunk.short_summary or 'n/a'}",
                        "Content:",
                        chunk.text,
                    ]
                )
            )

        return "\n\n".join(
            [
                f"Repository: {repo_summary.repo_name}",
                f"Branch: {repo_summary.branch}",
                f"Detected languages: {', '.join(repo_summary.detected_languages)}",
                f"Key files: {', '.join(repo_summary.key_files[:8]) or 'n/a'}",
                f"High-level summary: {repo_summary.high_level_summary}",
                "",
                f"Question: {question}",
                "",
                "Retrieved repository context:",
                "\n\n".join(context_blocks),
                "",
                "Answer the question using only this context. If you are missing evidence, say exactly what is missing.",
            ]
        )

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
