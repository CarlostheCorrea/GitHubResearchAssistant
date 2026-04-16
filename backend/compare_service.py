from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from backend.config import Settings
from backend.embedder import OpenAIEmbedder
from backend.github_loader import GitHubLoader
from backend.judge_service import LLMJudgeService
from backend.models import (
    AskCompareResponse,
    AskVersionCompareResponse,
    BranchFileChange,
    BranchListResponse,
    CommitRecord,
    CompareBranchesResponse,
    CompareVersionsResponse,
    VersionFileChange,
)
from backend.utils import looks_like_placeholder_secret

logger = logging.getLogger(__name__)

COMPARE_SYSTEM_PROMPT = """You answer questions about differences between two repository versions.

Rules:
- Use only the supplied version comparison data.
- Explain practical code and project impact, not generic Git concepts.
- If the comparison data is insufficient for a specific claim, say so directly.
- Do not invent files, commits, authors, or behavior.
- Prefer concise plain prose.
- Do not use markdown tables, code fences, or numbered lists.
"""

_MAX_CONTEXT_CHARS = 42_000
_MAX_PATCH_CHARS = 8_000   # per-file patch truncation before storing
_RAG_TOP_K = 8             # diff chunks retrieved per question


# ── Lightweight in-memory RAG store for diff chunks ──────────────────────────

@dataclass
class DiffRAGStore:
    """Holds embedded diff chunks for one comparison. OpenAI embeddings are
    L2-normalised, so dot-product == cosine similarity."""
    chunks: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    def retrieve(self, query_embedding: list[float], top_k: int = _RAG_TOP_K) -> list[str]:
        if not self.embeddings:
            return self.chunks[:top_k]
        scores = [
            (sum(a * b for a, b in zip(query_embedding, emb)), i)
            for i, emb in enumerate(self.embeddings)
        ]
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k]]


class BranchCompareService:
    def __init__(
        self,
        settings: Settings,
        loader: GitHubLoader,
        judge: LLMJudgeService | None = None,
        embedder: OpenAIEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.judge = judge
        self.embedder = embedder
        self._client: OpenAI | None = None
        self._comparisons: dict[str, CompareVersionsResponse] = {}
        self._rag_stores: dict[str, DiffRAGStore] = {}

    def list_branches(self, repo_url: str) -> BranchListResponse:
        repo = self.loader.resolve_repo(repo_url)
        return BranchListResponse(
            repo_name=repo.repo_name,
            branches=self.loader.list_branches(repo),
        )

    def compare_branches(
        self,
        repo_url: str,
        base_branch: str,
        head_branch: str,
    ) -> CompareBranchesResponse:
        version_response = self.compare_versions(
            repo_url,
            base_branch,
            head_branch,
            base_label=base_branch,
            head_label=head_branch,
        )
        return CompareBranchesResponse(
            repo_name=version_response.repo_name,
            base_branch=version_response.base_ref,
            head_branch=version_response.head_ref,
            status=version_response.status,
            ahead_by=version_response.ahead_by,
            behind_by=version_response.behind_by,
            total_commits=version_response.total_commits,
            commits=version_response.commits,
            files_changed=version_response.files_changed,
            total_additions=version_response.total_additions,
            total_deletions=version_response.total_deletions,
            changed_files=[
                BranchFileChange.model_validate(file.model_dump())
                for file in version_response.changed_files
            ],
            compare_id=version_response.compare_id,
        )

    def compare_versions(
        self,
        repo_url: str,
        base_ref: str,
        head_ref: str,
        base_label: str | None = None,
        head_label: str | None = None,
    ) -> CompareVersionsResponse:
        repo = self.loader.resolve_repo(repo_url)
        payload = self.loader.compare_refs(repo, base_ref, head_ref)

        commits = [
            self._commit_from_compare_item(item)
            for item in payload.get("commits", [])
        ]
        changed_files = [
            self._file_change_from_compare_item(item)
            for item in payload.get("files", [])
        ]

        compare_id = self._build_compare_id(
            repo_name=repo.repo_name,
            base_ref=base_ref,
            head_ref=head_ref,
            status=str(payload.get("status") or ""),
            commits=commits,
            changed_files=changed_files,
        )
        response = CompareVersionsResponse(
            repo_name=repo.repo_name,
            base_ref=base_ref,
            head_ref=head_ref,
            base_label=base_label or self._format_ref_label(base_ref),
            head_label=head_label or self._format_ref_label(head_ref),
            status=str(payload.get("status") or "unknown"),
            ahead_by=int(payload.get("ahead_by") or 0),
            behind_by=int(payload.get("behind_by") or 0),
            total_commits=int(payload.get("total_commits") or len(commits)),
            commits=commits,
            files_changed=len(changed_files),
            total_additions=sum(file.additions for file in changed_files),
            total_deletions=sum(file.deletions for file in changed_files),
            changed_files=changed_files,
            compare_id=compare_id,
        )
        self._store_comparison(compare_id, response)
        if self.embedder is not None:
            try:
                self._build_rag_store(compare_id, response)
            except Exception:
                logger.warning("RAG store build failed for compare_id=%s; falling back to full-context prompt.", compare_id, exc_info=True)
        return response

    def ask_compare(self, compare_id: str, question: str) -> AskCompareResponse:
        response = self.ask_version_compare(compare_id, question)
        return AskCompareResponse(
            compare_id=response.compare_id,
            question=response.question,
            answer=response.answer,
        )

    def ask_version_compare(self, compare_id: str, question: str) -> AskVersionCompareResponse:
        comparison = self._comparisons.get(compare_id)
        if comparison is None:
            raise ValueError("Run a version comparison before asking about it.")

        client = self._get_client()

        # Use RAG-retrieved diff chunks when available; fall back to full-context prompt.
        rag_store = self._rag_stores.get(compare_id)
        if rag_store and self.embedder is not None:
            try:
                query_embedding = self.embedder.embed_query(question)
                retrieved_chunks = rag_store.retrieve(query_embedding)
                compare_prompt = self._build_rag_prompt(comparison, question, retrieved_chunks)
                logger.info(
                    "RAG retrieved %d diff chunks for compare_id=%s", len(retrieved_chunks), compare_id
                )
            except Exception:
                logger.warning("RAG retrieval failed; falling back to full-context prompt.", exc_info=True)
                compare_prompt = self._build_compare_prompt(comparison, question)
        else:
            compare_prompt = self._build_compare_prompt(comparison, question)

        response = client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": compare_prompt},
            ],
        )
        draft_answer = (response.choices[0].message.content or "").strip()

        # LLM-as-a-Judge pass — same quality gate as the main Q&A flow
        if self.judge is not None:
            final_answer = self.judge.review_and_revise_with_context(
                question=question,
                draft_answer=draft_answer,
                context_text=compare_prompt,
            )
        else:
            final_answer = draft_answer

        return AskVersionCompareResponse(
            compare_id=compare_id,
            question=question,
            answer=final_answer,
        )

    def _commit_from_compare_item(self, item: dict[str, Any]) -> CommitRecord:
        sha = str(item.get("sha") or "")
        commit = item.get("commit") or {}
        author = commit.get("author") or {}
        return CommitRecord(
            sha=sha,
            short_sha=sha[:7],
            author_name=str(author.get("name") or "Unknown"),
            date=str(author.get("date") or ""),
            message=str(commit.get("message") or "").splitlines()[0],
        )

    def _file_change_from_compare_item(self, item: dict[str, Any]) -> VersionFileChange:
        patch = item.get("patch")
        if isinstance(patch, str) and len(patch) > _MAX_PATCH_CHARS:
            patch = patch[:_MAX_PATCH_CHARS] + "\n... [patch truncated]"
        return VersionFileChange(
            filename=str(item.get("filename") or ""),
            status=str(item.get("status") or "modified"),
            additions=int(item.get("additions") or 0),
            deletions=int(item.get("deletions") or 0),
            changes=int(item.get("changes") or 0),
            patch=patch if isinstance(patch, str) else None,
        )

    def _build_compare_id(
        self,
        repo_name: str,
        base_ref: str,
        head_ref: str,
        status: str,
        commits: list[CommitRecord],
        changed_files: list[VersionFileChange],
    ) -> str:
        seed = "|".join(
            [
                repo_name,
                base_ref,
                head_ref,
                status,
                ",".join(commit.sha for commit in commits[:10]),
                ",".join(file.filename for file in changed_files[:40]),
            ]
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Drop all cached comparisons and RAG stores (e.g. on a global cache clear)."""
        self._comparisons.clear()
        self._rag_stores.clear()
        logger.info("Compare service cache cleared.")

    def _store_comparison(self, compare_id: str, response: CompareVersionsResponse) -> None:
        """Replace the cache with only this comparison — each run starts fresh."""
        self._comparisons.clear()
        self._rag_stores.clear()
        self._comparisons[compare_id] = response

    def _store_rag(self, compare_id: str, store: DiffRAGStore) -> None:
        """Store the RAG index for the current comparison."""
        self._rag_stores[compare_id] = store

    # ── RAG helpers ──────────────────────────────────────────────────────────

    def _build_rag_store(self, compare_id: str, comparison: CompareVersionsResponse) -> None:
        """Chunk the diff into per-file and per-commit pieces and embed them."""
        chunks: list[str] = []

        # One chunk per changed file (filename + status + full patch, no truncation here)
        for file in comparison.changed_files:
            parts = [
                f"File: {file.filename}",
                f"Status: {file.status}",
                f"Additions: {file.additions}  Deletions: {file.deletions}",
            ]
            if file.patch:
                parts.append(f"Patch:\n{file.patch}")
            else:
                parts.append("Patch: [binary or no textual diff]")
            chunks.append("\n".join(parts))

        # One chunk per commit (summary + which files it touches)
        file_names = [f.filename for f in comparison.changed_files]
        for commit in comparison.commits:
            chunk = (
                f"Commit: {commit.sha}\n"
                f"Short SHA: {commit.short_sha}\n"
                f"Author: {commit.author_name}\n"
                f"Date: {commit.date}\n"
                f"Message: {commit.message}"
            )
            chunks.append(chunk)

        # A single summary chunk with the overall stats
        summary_chunk = (
            f"Repository: {comparison.repo_name}\n"
            f"Base: {comparison.base_label} ({comparison.base_ref})\n"
            f"Head: {comparison.head_label} ({comparison.head_ref})\n"
            f"Status: {comparison.status}\n"
            f"Ahead by: {comparison.ahead_by}  Behind by: {comparison.behind_by}\n"
            f"Total commits: {comparison.total_commits}\n"
            f"Files changed: {comparison.files_changed}\n"
            f"Total additions: {comparison.total_additions}  Total deletions: {comparison.total_deletions}\n"
            f"All changed files: {', '.join(file_names[:80])}"
        )
        chunks.insert(0, summary_chunk)

        assert self.embedder is not None  # guarded by caller
        embeddings = self.embedder.embed_texts(chunks)
        store = DiffRAGStore(chunks=chunks, embeddings=embeddings)
        self._store_rag(compare_id, store)
        logger.info("Built RAG store with %d chunks for compare_id=%s", len(chunks), compare_id)

    def _build_rag_prompt(
        self,
        comparison: CompareVersionsResponse,
        question: str,
        retrieved_chunks: list[str],
    ) -> str:
        """Build a focused prompt from RAG-retrieved diff chunks."""
        header = "\n".join([
            f"Repository: {comparison.repo_name}",
            f"Base: {comparison.base_label} ({comparison.base_ref})",
            f"Head: {comparison.head_label} ({comparison.head_ref})",
            f"Status: {comparison.status}  "
            f"Ahead by: {comparison.ahead_by}  Behind by: {comparison.behind_by}",
            f"Total commits: {comparison.total_commits}  "
            f"Files changed: {comparison.files_changed}  "
            f"Additions: {comparison.total_additions}  Deletions: {comparison.total_deletions}",
            "",
            f"Question: {question}",
            "",
            "Most relevant diff evidence (retrieved by semantic similarity):",
        ])
        body = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"{header}\n\n{body}"
        if len(prompt) > _MAX_CONTEXT_CHARS:
            prompt = prompt[:_MAX_CONTEXT_CHARS] + "\n... [context truncated]"
        return prompt

    def _build_compare_prompt(self, comparison: CompareVersionsResponse, question: str) -> str:
        lines = [
            f"Repository: {comparison.repo_name}",
            f"Base version: {comparison.base_label} ({comparison.base_ref})",
            f"Head version: {comparison.head_label} ({comparison.head_ref})",
            f"Compare status: {comparison.status}",
            f"Ahead by: {comparison.ahead_by}",
            f"Behind by: {comparison.behind_by}",
            f"Total commits: {comparison.total_commits}",
            f"Files changed: {comparison.files_changed}",
            f"Total additions: {comparison.total_additions}",
            f"Total deletions: {comparison.total_deletions}",
            "",
            f"Question: {question}",
            "",
            "Commits:",
        ]
        for commit in comparison.commits[:25]:
            lines.append(f"- {commit.short_sha} {commit.message} by {commit.author_name} on {commit.date}")

        lines.extend(["", "Changed files and patches:"])
        for file in comparison.changed_files[:60]:
            lines.append(
                f"\nFile: {file.filename}\n"
                f"Status: {file.status}\n"
                f"Additions: {file.additions}\n"
                f"Deletions: {file.deletions}\n"
                f"Patch:\n{file.patch or '[No textual patch available]'}"
            )

        prompt = "\n".join(lines)
        if len(prompt) > _MAX_CONTEXT_CHARS:
            prompt = prompt[:_MAX_CONTEXT_CHARS] + "\n... [comparison context truncated]"
        return prompt

    def _format_ref_label(self, ref: str) -> str:
        return ref[:7] if len(ref) >= 12 and all(char in "0123456789abcdefABCDEF" for char in ref) else ref

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for version comparison questions.")
            if looks_like_placeholder_secret(self.settings.openai_api_key):
                raise RuntimeError(
                    "OPENAI_API_KEY is still set to the placeholder value. Update .env with a real OpenAI API key and restart."
                )
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
