from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class RepoURLRequest(BaseModel):
    repo_url: str = Field(..., min_length=10)


class AskRequest(BaseModel):
    repo_url: str = Field(..., min_length=10)
    question: str = Field(..., min_length=3)


class RepoDescriptor(BaseModel):
    owner: str
    repo: str
    branch: str
    default_branch: str
    normalized_repo_url: str
    repo_id: str

    @property
    def repo_name(self) -> str:
        return f"{self.owner}/{self.repo}"


class RepoFile(BaseModel):
    path: str
    size: int
    sha: str | None = None
    blob_url: str | None = None
    language: str
    role: str
    content: str


class ChunkRecord(BaseModel):
    id: str
    repo_id: str
    repo_name: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    short_summary: str | None = None
    file_role: str | None = None
    text: str

    def embedding_input(self) -> str:
        header = [
            f"Repository: {self.repo_name}",
            f"File: {self.file_path}",
            f"Language: {self.language}",
            f"Chunk type: {self.chunk_type}",
        ]
        if self.symbol_name:
            header.append(f"Symbol: {self.symbol_name}")
        if self.short_summary:
            header.append(f"Summary: {self.short_summary}")
        if self.file_role:
            header.append(f"Role: {self.file_role}")
        return "\n".join(header) + "\n\n" + self.text

    def chroma_metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "chunk_id": self.id,
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "symbol_name": self.symbol_name or "",
            "start_line": self.start_line or 0,
            "end_line": self.end_line or 0,
            "short_summary": self.short_summary or "",
            "file_role": self.file_role or "",
        }


class SourceSnippet(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    chunk_type: str
    symbol_name: str | None = None
    short_summary: str | None = None
    snippet: str
    score: float

    @classmethod
    def from_chunk(cls, chunk: ChunkRecord, score: float) -> "SourceSnippet":
        return cls(
            chunk_id=chunk.id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            symbol_name=chunk.symbol_name,
            short_summary=chunk.short_summary,
            snippet=chunk.text,
            score=round(score, 4),
        )


class RepoSummary(BaseModel):
    repo_name: str
    owner: str
    branch: str
    normalized_repo_url: str
    detected_languages: list[str] = Field(default_factory=list)
    language_distribution: dict[str, int] = Field(default_factory=dict)
    key_files: list[str] = Field(default_factory=list)
    high_level_summary: str
    readme_excerpt: str | None = None
    probable_entry_points: list[str] = Field(default_factory=list)
    probable_training_files: list[str] = Field(default_factory=list)
    probable_inference_files: list[str] = Field(default_factory=list)
    probable_config_files: list[str] = Field(default_factory=list)
    probable_data_files: list[str] = Field(default_factory=list)
    files_indexed: int = 0
    chunks_indexed: int = 0


class AnalyzeRepoResponse(BaseModel):
    status: str
    cached: bool
    files_seen: int
    files_indexed: int
    skipped_files: int
    chunks_created: int
    message: str
    repo_summary: RepoSummary


class DeleteRepoCacheResponse(BaseModel):
    status: str
    repo_url: str
    repo_name: str
    cache_deleted: bool
    deleted_manifest: bool
    deleted_vector_index: bool
    message: str


class ClearAllCacheResponse(BaseModel):
    status: str
    deleted_manifests: int
    deleted_vector_indexes: int
    message: str


class AskResponse(BaseModel):
    repo_url: str
    question: str
    answer: str
    sources: list[SourceSnippet] = Field(default_factory=list)
    repo_summary: RepoSummary


class RepoManifest(BaseModel):
    repo: RepoDescriptor
    summary: RepoSummary
    files_seen: int
    files_indexed: int
    skipped_files: int
    chunks_created: int
    created_at: datetime
    updated_at: datetime
