from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CommitRecord(BaseModel):
    sha: str
    short_sha: str
    author_name: str
    date: str  # ISO 8601 string
    message: str  # first line of commit message
    file_changes: list[dict[str, str]] = Field(default_factory=list)
    # Each entry: {"path": "src/foo.py", "status": "added"|"modified"|"deleted"|"renamed"}
    summary: str = ""  # 1-sentence LLM explanation of what this commit does


class ChangesSummary(BaseModel):
    old_sha: str
    new_sha: str
    files_changed: int
    insertions: int
    deletions: int
    changed_files: list[str] = Field(default_factory=list)
    # File-level breakdown (not line counts)
    files_added: int = 0
    files_removed: int = 0
    files_modified: int = 0


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
    global_context: str = ""
    critical_paths: list[list[str]] = Field(default_factory=list)
    dependency_links: list[list[str]] = Field(default_factory=list)
    graph_hubs: list[str] = Field(default_factory=list)
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
    head_commit_sha: str | None = None
    first_commit: CommitRecord | None = None
    changes_since_previous: ChangesSummary | None = None
    commit_history: list[CommitRecord] = Field(default_factory=list)
    activity_summary: str = ""  # 1-2 sentence LLM summary of recent commit activity


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
    head_commit_sha: str | None = None
    first_commit: CommitRecord | None = None
    previous_head_sha: str | None = None
    commit_history: list[CommitRecord] = Field(default_factory=list)
    changes_since_previous: ChangesSummary | None = None
    activity_summary: str = ""


class HistoryResponse(BaseModel):
    repo_name: str
    head_commit_sha: str | None
    first_commit: CommitRecord | None = None
    commit_history: list[CommitRecord]
    changes_since_previous: ChangesSummary | None


# ── Version comparison ──────────────────────────────────────────────────────

class VersionListResponse(BaseModel):
    repo_name: str
    branch: str
    head_commit_sha: str | None
    first_commit: CommitRecord | None = None
    commit_history: list[CommitRecord] = Field(default_factory=list)


class VersionFileChange(BaseModel):
    filename: str
    status: str  # added | removed | modified | renamed
    additions: int
    deletions: int
    changes: int
    patch: str | None = None  # raw unified diff text (may be absent for binary)


class CompareVersionsRequest(BaseModel):
    repo_url: str = Field(..., min_length=10)
    base_ref: str = Field(..., min_length=1)
    head_ref: str = Field(..., min_length=1)
    base_label: str | None = None
    head_label: str | None = None


class CompareVersionsResponse(BaseModel):
    repo_name: str
    base_ref: str
    head_ref: str
    base_label: str
    head_label: str
    status: str  # ahead | behind | diverged | identical
    ahead_by: int
    behind_by: int
    total_commits: int
    commits: list[CommitRecord] = Field(default_factory=list)
    files_changed: int
    total_additions: int
    total_deletions: int
    changed_files: list[VersionFileChange] = Field(default_factory=list)
    compare_id: str  # stable key used by /ask-compare


class AskVersionCompareRequest(BaseModel):
    compare_id: str
    question: str = Field(..., min_length=3)


class AskVersionCompareResponse(BaseModel):
    compare_id: str
    question: str
    answer: str


# ── Branch comparison compatibility aliases ─────────────────────────────────

class BranchListResponse(BaseModel):
    repo_name: str
    branches: list[str]


class BranchFileChange(VersionFileChange):
    pass


class CompareBranchesRequest(BaseModel):
    repo_url: str = Field(..., min_length=10)
    base_branch: str = Field(..., min_length=1)
    head_branch: str = Field(..., min_length=1)


class CompareBranchesResponse(BaseModel):
    repo_name: str
    base_branch: str
    head_branch: str
    status: str  # ahead | behind | diverged | identical
    ahead_by: int
    behind_by: int
    total_commits: int
    commits: list[CommitRecord] = Field(default_factory=list)
    files_changed: int
    total_additions: int
    total_deletions: int
    changed_files: list[BranchFileChange] = Field(default_factory=list)
    compare_id: str  # stable key used by /ask-compare


class AskCompareRequest(BaseModel):
    compare_id: str
    question: str = Field(..., min_length=3)


class AskCompareResponse(BaseModel):
    compare_id: str
    question: str
    answer: str


# ── Onboarding Guide ────────────────────────────────────────────────────────

class ReadingStep(BaseModel):
    step: int
    file_path: str
    role: str
    reason: str

class CoreConcept(BaseModel):
    name: str
    description: str
    key_files: list[str] = Field(default_factory=list)

class ContributorProfile(BaseModel):
    name: str
    commits: int
    focus_area: str
    recent_files: list[str] = Field(default_factory=list)

class OnboardingResponse(BaseModel):
    repo_name: str
    reading_order: list[ReadingStep] = Field(default_factory=list)
    core_concepts: list[CoreConcept] = Field(default_factory=list)
    contributors: list[ContributorProfile] = Field(default_factory=list)
    complexity_note: str = ""
    cached: bool = False


# ── Repository Map ───────────────────────────────────────────────────────────

class RepoMapNode(BaseModel):
    id: str            # file path, used as D3 node id
    file_path: str
    language: str
    role: str
    line_count: int    # estimated from chunk line ranges
    short_summary: str
    key_symbols: list[str] = Field(default_factory=list)


class RepoMapEdge(BaseModel):
    source: str        # file_path of importer
    target: str        # file_path of imported file


class RepoMapResponse(BaseModel):
    repo_name: str
    nodes: list[RepoMapNode] = Field(default_factory=list)
    edges: list[RepoMapEdge] = Field(default_factory=list)
