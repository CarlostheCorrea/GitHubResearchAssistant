from __future__ import annotations

from collections import Counter
import logging
import re

from openai import OpenAI

from backend.config import Settings
from backend.models import ChunkRecord, RepoDescriptor, RepoFile, RepoSummary
from backend.utils import compact_whitespace, first_nonempty_lines, looks_like_placeholder_secret, truncate_text, unique_preserve_order

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are writing a concise high-level summary of a GitHub repository.

Use only the structured repository signals provided.

Rules:
- Write a short, human-readable summary in 2 to 4 sentences.
- Mention what the repository appears to do, the main architecture signals, and the most relevant implementation areas.
- Do not invent files, functions, components, workflows, or behavior not present in the provided signals.
- Do not dump long lists of file paths into the prose.
- Prefer plain technical language over marketing language.
"""


class RepoSummarizer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def summarize(
        self,
        repo: RepoDescriptor,
        files: list[RepoFile],
        chunks: list[ChunkRecord],
        global_context: str = "",
        critical_paths: list[list[str]] | None = None,
        dependency_links: list[list[str]] | None = None,
        graph_hubs: list[str] | None = None,
    ) -> RepoSummary:
        language_counts = Counter(file.language for file in files)
        detected_languages = [language for language, _count in language_counts.most_common()]

        key_files = self._select_key_files(files)
        readme_excerpt = self._readme_excerpt(files)
        components = self._main_components(chunks, key_files)
        probable_entry_points = self._paths_for_role(files, "entrypoint")
        probable_training_files = self._paths_for_role(files, "training")
        probable_inference_files = self._paths_for_role(files, "inference")
        probable_config_files = self._paths_for_role(files, "config")
        probable_data_files = self._paths_for_role(files, "data_loading")

        heuristic_summary = self._build_heuristic_summary(
            repo=repo,
            detected_languages=detected_languages,
            files=files,
            chunks=chunks,
            readme_excerpt=readme_excerpt,
            components=components,
            probable_entry_points=probable_entry_points,
            probable_training_files=probable_training_files,
            probable_inference_files=probable_inference_files,
            probable_config_files=probable_config_files,
            probable_data_files=probable_data_files,
        )
        high_level_summary = self._build_llm_summary(
            repo=repo,
            detected_languages=detected_languages,
            files=files,
            chunks=chunks,
            readme_excerpt=readme_excerpt,
            components=components,
            probable_entry_points=probable_entry_points,
            probable_training_files=probable_training_files,
            probable_inference_files=probable_inference_files,
            probable_config_files=probable_config_files,
            probable_data_files=probable_data_files,
            fallback=heuristic_summary,
        )

        return RepoSummary(
            repo_name=repo.repo_name,
            owner=repo.owner,
            branch=repo.branch,
            normalized_repo_url=repo.normalized_repo_url,
            detected_languages=detected_languages,
            language_distribution=dict(language_counts),
            key_files=key_files,
            high_level_summary=high_level_summary,
            global_context=global_context,
            critical_paths=critical_paths or [],
            dependency_links=dependency_links or [],
            graph_hubs=graph_hubs or [],
            readme_excerpt=readme_excerpt,
            probable_entry_points=probable_entry_points,
            probable_training_files=probable_training_files,
            probable_inference_files=probable_inference_files,
            probable_config_files=probable_config_files,
            probable_data_files=probable_data_files,
            files_indexed=len(files),
            chunks_indexed=len(chunks),
        )

    def _build_heuristic_summary(
        self,
        repo: RepoDescriptor,
        detected_languages: list[str],
        files: list[RepoFile],
        chunks: list[ChunkRecord],
        readme_excerpt: str | None,
        components: list[str],
        probable_entry_points: list[str],
        probable_training_files: list[str],
        probable_inference_files: list[str],
        probable_config_files: list[str],
        probable_data_files: list[str],
    ) -> str:
        lead_language = detected_languages[0] if detected_languages else "mixed-language"
        summary_parts = [
            f"{repo.repo_name} appears to be a {lead_language} repository.",
            f"The analysis indexed {len(files)} supported files and {len(chunks)} retrieval chunks.",
        ]
        if readme_excerpt:
            summary_parts.append(f"Based on the README, the project is described as {readme_excerpt}")
        if components:
            summary_parts.append(f"Key surfaced components include {', '.join(components[:5])}.")
        if probable_entry_points:
            summary_parts.append(f"Likely entry points are {self._format_paths(probable_entry_points, 2)}.")
        if probable_training_files:
            summary_parts.append(f"Training-related code is most likely in {self._format_paths(probable_training_files)}.")
        if probable_inference_files:
            summary_parts.append(f"Inference or serving logic is most likely in {self._format_paths(probable_inference_files)}.")
        if probable_config_files:
            summary_parts.append(f"Configuration appears to be defined in {self._format_paths(probable_config_files)}.")
        if probable_data_files:
            summary_parts.append(f"Data-loading logic appears in {self._format_paths(probable_data_files)}.")
        return " ".join(summary_parts)

    def _build_llm_summary(
        self,
        repo: RepoDescriptor,
        detected_languages: list[str],
        files: list[RepoFile],
        chunks: list[ChunkRecord],
        readme_excerpt: str | None,
        components: list[str],
        probable_entry_points: list[str],
        probable_training_files: list[str],
        probable_inference_files: list[str],
        probable_config_files: list[str],
        probable_data_files: list[str],
        fallback: str,
    ) -> str:
        client = self._get_client()
        if client is None or self.settings is None:
            return fallback

        prompt = "\n".join(
            [
                f"Repository: {repo.repo_name}",
                f"Branch: {repo.branch}",
                f"Detected languages: {', '.join(detected_languages) or 'unknown'}",
                f"Files indexed: {len(files)}",
                f"Chunks indexed: {len(chunks)}",
                f"README signal: {readme_excerpt or 'n/a'}",
                f"Surfaced components: {', '.join(components[:8]) or 'n/a'}",
                f"Likely entry points: {', '.join(probable_entry_points[:4]) or 'n/a'}",
                f"Likely training files: {', '.join(probable_training_files[:4]) or 'n/a'}",
                f"Likely inference files: {', '.join(probable_inference_files[:4]) or 'n/a'}",
                f"Likely config files: {', '.join(probable_config_files[:4]) or 'n/a'}",
                f"Likely data-loading files: {', '.join(probable_data_files[:4]) or 'n/a'}",
                "",
                "Write the repository high-level summary now.",
            ]
        )

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            return compact_whitespace(content) or fallback
        except Exception:  # noqa: BLE001
            logger.warning("Falling back to heuristic repo summary for %s", repo.repo_name, exc_info=True)
            return fallback

    def _select_key_files(self, files: list[RepoFile]) -> list[str]:
        scored: list[tuple[int, str]] = []
        for repo_file in files:
            score = 0
            lower_path = repo_file.path.lower()
            if lower_path.endswith("readme.md"):
                score += 5
            if repo_file.role in {"entrypoint", "training", "inference", "config", "data_loading"}:
                score += 4
            if lower_path.count("/") == 0:
                score += 2
            if lower_path.endswith(("main.py", "app.py", "server.py", "index.ts", "index.js")):
                score += 3
            score += max(0, 2 - lower_path.count("/"))
            scored.append((score, repo_file.path))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _score, path in scored[:10]]

    def _paths_for_role(self, files: list[RepoFile], role: str) -> list[str]:
        paths = [repo_file.path for repo_file in files if repo_file.role == role]
        return paths[:6]

    def _readme_excerpt(self, files: list[RepoFile]) -> str | None:
        readme = next((repo_file for repo_file in files if repo_file.path.lower().endswith("readme.md")), None)
        if not readme:
            return None
        lines = first_nonempty_lines(readme.content, max_lines=5)
        if not lines:
            return None
        cleaned_lines = [self._clean_readme_line(line) for line in lines]
        cleaned = compact_whitespace(" ".join(line for line in cleaned_lines if line))
        if not cleaned:
            return None
        sentence_split = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [sentence.strip() for sentence in sentence_split if sentence.strip()]
        if not sentences:
            return None

        excerpt_parts: list[str] = []
        current_length = 0
        for sentence in sentences[:2]:
            projected_length = current_length + len(sentence) + (1 if excerpt_parts else 0)
            if projected_length > 220 and excerpt_parts:
                break
            excerpt_parts.append(sentence)
            current_length = projected_length

        excerpt = " ".join(excerpt_parts) or sentences[0]
        return truncate_text(excerpt, 220)

    def _main_components(self, chunks: list[ChunkRecord], key_files: list[str]) -> list[str]:
        component_chunks = [
            chunk.symbol_name
            for chunk in chunks
            if chunk.symbol_name
            and chunk.file_path in key_files
            and chunk.chunk_type in {"python_class", "python_function", "python_method", "code_block"}
        ]
        return unique_preserve_order(component_chunks)[:10]

    def _format_paths(self, paths: list[str], limit: int = 3) -> str:
        selected = paths[:limit]
        if not selected:
            return "no obvious files"
        if len(paths) > limit:
            return ", ".join(selected) + ", and related files"
        return ", ".join(selected)

    def _clean_readme_line(self, line: str) -> str:
        cleaned = line.strip()
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"^[-*+]\s*", "", cleaned)
        cleaned = re.sub(r"`+", "", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
        cleaned = re.sub(r"[^\w\s:/().,+-]", "", cleaned)
        cleaned = cleaned.strip(":- ")
        return cleaned

    def _get_client(self) -> OpenAI | None:
        if self.settings is None:
            return None
        if not self.settings.openai_api_key:
            return None
        if looks_like_placeholder_secret(self.settings.openai_api_key):
            return None
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client
