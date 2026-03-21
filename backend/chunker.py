from __future__ import annotations

import logging

from backend.models import ChunkRecord, RepoDescriptor, RepoFile
from backend.parsers import (
    SectionSpan,
    build_file_summary,
    extract_python_sections,
    split_code_sections,
    split_markdown_sections,
    split_structured_sections,
    split_text_sections,
)
from backend.utils import build_chunk_id


logger = logging.getLogger(__name__)


class CodeAwareChunker:
    def chunk_files(self, repo: RepoDescriptor, files: list[RepoFile]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for repo_file in files:
            chunks.extend(self.chunk_file(repo, repo_file))
        logger.info("Created %s chunks for %s", len(chunks), repo.repo_name)
        return chunks

    def chunk_file(self, repo: RepoDescriptor, repo_file: RepoFile) -> list[ChunkRecord]:
        sections = self._sections_for_file(repo_file)
        chunks = [self._section_to_chunk(repo, repo_file, section) for section in sections]

        summary_text = build_file_summary(
            file_path=repo_file.path,
            language=repo_file.language,
            role=repo_file.role,
            content=repo_file.content,
        )
        chunks.append(
            ChunkRecord(
                id=build_chunk_id(
                    repo.repo_id,
                    repo_file.path,
                    "file_summary",
                    None,
                    1,
                    min(len(repo_file.content.splitlines()), 20),
                    summary_text,
                ),
                repo_id=repo.repo_id,
                repo_name=repo.repo_name,
                file_path=repo_file.path,
                language=repo_file.language,
                chunk_type="file_summary",
                start_line=1,
                end_line=min(len(repo_file.content.splitlines()), 20),
                short_summary="Heuristic file summary",
                file_role=repo_file.role,
                text=summary_text,
            )
        )
        return chunks

    def _sections_for_file(self, repo_file: RepoFile) -> list[SectionSpan]:
        if repo_file.language == "python":
            python_sections = extract_python_sections(repo_file.content)
            if python_sections:
                return python_sections
        if repo_file.language == "markdown":
            return split_markdown_sections(repo_file.content)
        if repo_file.language in {"json", "yaml", "toml"}:
            return split_structured_sections(repo_file.path, repo_file.content)
        if repo_file.language in {"javascript", "typescript", "tsx"}:
            return split_code_sections(repo_file.content)
        return split_text_sections(repo_file.content)

    def _section_to_chunk(
        self,
        repo: RepoDescriptor,
        repo_file: RepoFile,
        section: SectionSpan,
    ) -> ChunkRecord:
        return ChunkRecord(
            id=build_chunk_id(
                repo.repo_id,
                repo_file.path,
                section.chunk_type,
                section.symbol_name,
                section.start_line,
                section.end_line,
                section.text,
            ),
            repo_id=repo.repo_id,
            repo_name=repo.repo_name,
            file_path=repo_file.path,
            language=repo_file.language,
            chunk_type=section.chunk_type,
            symbol_name=section.symbol_name,
            start_line=section.start_line,
            end_line=section.end_line,
            short_summary=section.short_summary,
            file_role=repo_file.role,
            text=section.text,
        )
