from __future__ import annotations

from openai import OpenAI

from backend.config import Settings
from backend.models import ChunkRecord, RepoSummary
from backend.utils import line_range_label, looks_like_placeholder_secret


SYSTEM_PROMPT = """You answer questions about a GitHub repository using retrieved repository evidence plus optional repository-wide graph context.

Rules:
- Use the global graph context only as high-level structural guidance.
- Ground concrete claims in the retrieved repository chunks.
- If the evidence is insufficient or ambiguous, say so directly.
- Do not invent files, functions, classes, behavior, or architecture.
- Explain relationships across files only when the retrieved context supports that connection.
- Prefer concise, technical answers over generic descriptions.
- Return plain prose only.
- Do not use code fences, markdown bullets, numbered lists, or inline code formatting.
- Do not append bracketed file citations or line-range references in the answer body.
- Summarize behavior instead of reproducing implementation steps line by line.
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
                f"Global graph context: {repo_summary.global_context or 'n/a'}",
                "",
                f"Question: {question}",
                "",
                "Retrieved repository context:",
                "\n\n".join(context_blocks),
                "",
                "Use the graph context only to understand repo-wide structure. Support your answer with the retrieved chunks, but do not append bracketed file citations in the answer body because the UI shows sources separately. Respond in short plain prose paragraphs only, with no code formatting or markdown lists. If you are missing evidence, say exactly what is missing.",
            ]
        )

    def summarize_commits(self, repo_name: str, commits: list[object]) -> dict[str, str]:
        """
        Batch-generate a one-sentence explanation for each commit.
        Returns a dict mapping short_sha → explanation.
        Caps at 40 commits to keep the prompt manageable.
        """
        if not commits:
            return {}
        try:
            client = self._get_client()
        except RuntimeError:
            return {}

        batch = commits[:40]
        lines = []
        for c in batch:
            file_parts = []
            for fc in getattr(c, "file_changes", [])[:6]:
                file_parts.append(f"{fc['status']}: {fc['path']}")
            file_str = ", ".join(file_parts) if file_parts else "no files recorded"
            lines.append(f"[{c.short_sha}] {c.message}\n  Files: {file_str}")

        prompt = (
            f"Repository: {repo_name}\n\n"
            "For each commit below, write ONE concise sentence explaining what the change accomplishes. "
            "Focus on the effect, not just restating the commit message. Plain text only.\n\n"
            "Commits:\n" + "\n\n".join(lines) + "\n\n"
            "Reply ONLY with lines in this exact format (one per commit):\n"
            "[<7-char sha>]: <one sentence>\n"
            "Do not add any other text."
        )

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0.2,
                max_tokens=60 * len(batch),
                messages=[
                    {"role": "system", "content": "You explain git commits concisely."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
        except Exception:  # noqa: BLE001
            return {}

        # Parse "[sha]: explanation" lines
        import re as _re
        result: dict[str, str] = {}
        for match in _re.finditer(r"\[([a-f0-9]{7})\]:\s*(.+)", raw):
            result[match.group(1)] = match.group(2).strip()
        return result

    def summarize_activity(self, repo_name: str, commits: list[ChunkRecord | object]) -> str:
        """Generate a 1-2 sentence plain-English summary of recent commit activity."""
        if not commits:
            return ""
        try:
            client = self._get_client()
        except RuntimeError:
            return ""

        lines = []
        for c in commits[:20]:
            file_count = len(getattr(c, "file_changes", []))
            file_note = f" ({file_count} file{'s' if file_count != 1 else ''} changed)" if file_count else ""
            lines.append(f"- [{c.short_sha}] {c.message} — {c.author_name}{file_note}")

        prompt = (
            f"Repository: {repo_name}\n"
            f"Recent commits:\n" + "\n".join(lines) + "\n\n"
            "In 1-2 sentences, describe what kind of work is actively happening in this repository "
            "based on these commits. Be specific and technical. Plain text only, no markdown."
        )
        try:
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": "You summarize software development activity concisely."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:  # noqa: BLE001
            return ""

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
