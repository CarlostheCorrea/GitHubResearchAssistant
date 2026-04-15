from __future__ import annotations

import logging
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, AuthenticationError, BadRequestError, RateLimitError

from backend.chunker import CodeAwareChunker
from backend.compare_service import BranchCompareService
from backend.config import Settings, get_settings
from backend.onboarding_service import OnboardingService
from backend.embedder import OpenAIEmbedder
from backend.file_filter import FileFilter
from backend.github_loader import GitHubLoader
from backend.judge_service import LLMJudgeService
from backend.knowledge_graph import KnowledgeGraphService
from backend.models import (
    AnalyzeRepoResponse,
    AskCompareRequest,
    AskCompareResponse,
    AskVersionCompareRequest,
    AskVersionCompareResponse,
    AskRequest,
    AskResponse,
    BranchListResponse,
    ChunkRecord,
    ClearAllCacheResponse,
    CompareBranchesRequest,
    CompareBranchesResponse,
    CompareVersionsRequest,
    CompareVersionsResponse,
    DeleteRepoCacheResponse,
    HealthResponse,
    HistoryResponse,
    OnboardingResponse,
    RepoDescriptor,
    RepoManifest,
    RepoSummary,
    RepoURLRequest,
    VersionListResponse,
)
from backend.qa_graph import RepoQAGraph
from backend.qa_service import QAService
from backend.repo_summarizer import RepoSummarizer
from backend.retriever import HybridRetriever
from backend.utils import build_chunk_id, detect_language, setup_logging, utc_now
from backend.vector_store import ChromaVectorStore


setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


class ResearchAssistantService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.file_filter = FileFilter(settings.max_file_bytes)
        self.loader = GitHubLoader(settings)
        self.chunker = CodeAwareChunker()
        self.embedder = OpenAIEmbedder(settings)
        self.vector_store = ChromaVectorStore(settings)
        self.retriever = HybridRetriever(settings, self.embedder, self.vector_store)
        self.knowledge_graph = KnowledgeGraphService()
        self.repo_summarizer = RepoSummarizer(settings)
        self.qa_service = QAService(settings)
        self.judge_service = LLMJudgeService(settings)
        self.qa_graph = RepoQAGraph(self.retriever, self.qa_service, self.judge_service)
        self.compare_service = BranchCompareService(settings, self.loader)
        self.onboarding_service = OnboardingService(settings)

    def analyze_repo(self, repo_url: str) -> AnalyzeRepoResponse:
        repo = self.loader.resolve_repo(repo_url)
        manifest = self._load_manifest(repo.repo_id)
        previous_sha: str | None = None

        # Fast-path: check if cached and unchanged via a single lightweight API call
        if manifest and self.vector_store.repo_has_data(repo.repo_id):
            try:
                current_sha = self.loader.get_head_sha(repo)
            except Exception:
                current_sha = None

            if current_sha and current_sha == manifest.head_commit_sha:
                manifest.summary = self.knowledge_graph.ensure_summary_global_context(manifest.summary)
                self._save_manifest(manifest)
                logger.info("Using cached repo analysis for %s (SHA unchanged)", repo.repo_name)
                return AnalyzeRepoResponse(
                    status="ready",
                    cached=True,
                    files_seen=manifest.files_seen,
                    files_indexed=manifest.files_indexed,
                    skipped_files=manifest.skipped_files,
                    chunks_created=manifest.chunks_created,
                    message="Loaded cached ingestion — no new commits detected.",
                    repo_summary=manifest.summary,
                    head_commit_sha=manifest.head_commit_sha,
                    first_commit=manifest.first_commit,
                    changes_since_previous=None,
                    commit_history=manifest.commit_history,
                    activity_summary=manifest.activity_summary,
                )

            # SHA changed — record previous SHA for diff, then clear stale vectors
            previous_sha = manifest.head_commit_sha
            self.vector_store.delete_repo(repo.repo_id)
            logger.info(
                "New commits detected for %s (%s → re-indexing)",
                repo.repo_name,
                (previous_sha or "none")[:7],
            )

        files, stats, head_sha, first_commit, commit_history, changes, file_diffs = (
            self.loader.load_repository_files(repo, self.file_filter, previous_sha=previous_sha)
        )
        if not files:
            raise ValueError("No supported text/code files were found in the repository.")

        chunks = self.chunker.chunk_files(repo, files)
        if not chunks:
            raise ValueError("No retrievable chunks could be created from the repository.")

        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.upsert_chunks(repo.repo_id, chunks, embeddings)

        # Embed commit history so QA can answer questions about commits/authors/timeline
        if commit_history:
            history_chunk = self._create_history_chunk(repo, commit_history, head_sha)
            history_embeddings = self.embedder.embed_chunks([history_chunk])
            self.vector_store.upsert_chunks(repo.repo_id, [history_chunk], history_embeddings)

        # Store diff chunks so the QA system can answer "what changed?"
        if file_diffs:
            diff_chunks = self._create_diff_chunks(repo, file_diffs, previous_sha or "", head_sha)
            if diff_chunks:
                diff_embeddings = self.embedder.embed_chunks(diff_chunks)
                self.vector_store.upsert_chunks(repo.repo_id, diff_chunks, diff_embeddings)

        graph_snapshot = self.knowledge_graph.build_snapshot(repo, files, chunks)
        summary = self.repo_summarizer.summarize(
            repo,
            files,
            chunks,
            global_context=graph_snapshot.global_context,
            critical_paths=graph_snapshot.critical_paths,
            dependency_links=graph_snapshot.dependency_links,
            graph_hubs=graph_snapshot.graph_hubs,
        )

        # Batch-generate per-commit summaries and the overall activity summary in parallel steps
        commit_summaries = self.qa_service.summarize_commits(repo.repo_name, commit_history)
        if commit_summaries:
            commit_history = [
                c.model_copy(update={"summary": commit_summaries.get(c.short_sha, "")})
                for c in commit_history
            ]

        activity_summary = self.qa_service.summarize_activity(repo.repo_name, commit_history)

        manifest = RepoManifest(
            repo=repo,
            summary=summary,
            files_seen=stats["files_seen"],
            files_indexed=stats["files_indexed"],
            skipped_files=stats["skipped_files"],
            chunks_created=len(chunks),
            created_at=utc_now(),
            updated_at=utc_now(),
            head_commit_sha=head_sha,
            first_commit=first_commit,
            previous_head_sha=previous_sha,
            commit_history=commit_history,
            changes_since_previous=changes,
            activity_summary=activity_summary,
        )
        self._save_manifest(manifest)

        logger.info(
            "Repository %s indexed: %s files, %s chunks, HEAD %s",
            repo.repo_name,
            len(files),
            len(chunks),
            head_sha[:7],
        )
        return AnalyzeRepoResponse(
            status="ready",
            cached=False,
            files_seen=stats["files_seen"],
            files_indexed=stats["files_indexed"],
            skipped_files=stats["skipped_files"],
            chunks_created=len(chunks),
            message="Repository analyzed and indexed successfully.",
            repo_summary=summary,
            head_commit_sha=head_sha,
            first_commit=first_commit,
            changes_since_previous=changes,
            commit_history=commit_history,
            activity_summary=activity_summary,
        )

    def get_repo_history(self, repo_url: str) -> HistoryResponse:
        repo = self.loader.resolve_repo(repo_url)
        manifest = self._load_manifest(repo.repo_id)
        if not manifest:
            raise ValueError("Repository has not been analyzed yet. Run an analysis first.")
        return HistoryResponse(
            repo_name=repo.repo_name,
            head_commit_sha=manifest.head_commit_sha,
            first_commit=manifest.first_commit,
            commit_history=manifest.commit_history,
            changes_since_previous=manifest.changes_since_previous,
        )

    def list_versions(self, repo_url: str) -> VersionListResponse:
        repo = self.loader.resolve_repo(repo_url)
        manifest = self._load_manifest(repo.repo_id)
        if not manifest:
            raise ValueError("Repository has not been analyzed yet. Run an analysis first.")
        return VersionListResponse(
            repo_name=repo.repo_name,
            branch=repo.branch,
            head_commit_sha=manifest.head_commit_sha,
            first_commit=manifest.first_commit,
            commit_history=manifest.commit_history,
        )

    def get_onboarding(self, repo_url: str) -> OnboardingResponse:
        repo = self.loader.resolve_repo(repo_url)
        manifest = self._load_manifest(repo.repo_id)
        if not manifest:
            raise ValueError("Repository has not been analyzed yet. Run an analysis first.")
        return self.onboarding_service.generate(
            repo=repo,
            summary=manifest.summary,
            commit_history=manifest.commit_history,
        )

    def list_branches(self, repo_url: str) -> BranchListResponse:
        return self.compare_service.list_branches(repo_url)

    def compare_branches(
        self,
        repo_url: str,
        base_branch: str,
        head_branch: str,
    ) -> CompareBranchesResponse:
        return self.compare_service.compare_branches(repo_url, base_branch, head_branch)

    def compare_versions(
        self,
        repo_url: str,
        base_ref: str,
        head_ref: str,
        base_label: str | None,
        head_label: str | None,
    ) -> CompareVersionsResponse:
        return self.compare_service.compare_versions(
            repo_url,
            base_ref,
            head_ref,
            base_label=base_label,
            head_label=head_label,
        )

    def ask_compare(self, compare_id: str, question: str) -> AskCompareResponse:
        return self.compare_service.ask_compare(compare_id, question)

    def ask_version_compare(self, compare_id: str, question: str) -> AskVersionCompareResponse:
        return self.compare_service.ask_version_compare(compare_id, question)

    def _create_history_chunk(
        self,
        repo: RepoDescriptor,
        commit_history: list,
        head_sha: str,
    ) -> ChunkRecord:
        lines = [
            f"Git commit history for {repo.repo_name} (branch: {repo.branch})",
            f"HEAD commit: {head_sha[:7]}",
            f"Total commits recorded: {len(commit_history)}",
            "",
        ]
        for i, commit in enumerate(commit_history, 1):
            lines.append(
                f"{i}. [{commit.short_sha}] {commit.message} — {commit.author_name} ({commit.date})"
            )
        text = "\n".join(lines)
        return ChunkRecord(
            id=build_chunk_id(repo.repo_id, ".git/history", "commit_history", None, None, None, text),
            repo_id=repo.repo_id,
            repo_name=repo.repo_name,
            file_path=".git/history",
            language="text",
            chunk_type="commit_history",
            symbol_name=None,
            start_line=None,
            end_line=None,
            short_summary=f"Full commit history for {repo.repo_name}: {len(commit_history)} commits, HEAD {head_sha[:7]}",
            file_role="general",
            text=text,
        )

    def _create_diff_chunks(
        self,
        repo: RepoDescriptor,
        file_diffs: list[tuple[str, str]],
        old_sha: str,
        new_sha: str,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for file_path, diff_text in file_diffs:
            if not diff_text.strip():
                continue
            language = detect_language(file_path)
            text = (
                f"Git diff for {file_path}\n"
                f"From commit {old_sha[:7]} → {new_sha[:7]}\n\n"
                f"{diff_text}"
            )
            chunks.append(
                ChunkRecord(
                    id=build_chunk_id(repo.repo_id, file_path, "git_diff", None, None, None, diff_text),
                    repo_id=repo.repo_id,
                    repo_name=repo.repo_name,
                    file_path=file_path,
                    language=language,
                    chunk_type="git_diff",
                    symbol_name=None,
                    start_line=None,
                    end_line=None,
                    short_summary=f"Changes to {file_path} ({old_sha[:7]} → {new_sha[:7]})",
                    file_role="general",
                    text=text,
                )
            )
        return chunks

    def ask(self, repo_url: str, question: str) -> AskResponse:
        analyze_response = self.analyze_repo(repo_url)
        repo_summary = analyze_response.repo_summary
        repo = self.loader.resolve_repo(repo_url)
        graph_result = self.qa_graph.run(repo, repo_summary, question)
        sources = graph_result.get("sources", [])
        answer = graph_result.get("answer", "")

        return AskResponse(
            repo_url=repo.normalized_repo_url,
            question=question,
            answer=answer,
            sources=sources,
            repo_summary=repo_summary,
        )

    def get_repo_summary(self, repo_url: str) -> RepoSummary:
        analyze_response = self.analyze_repo(repo_url)
        return analyze_response.repo_summary

    def clear_repo_cache(self, repo_url: str) -> DeleteRepoCacheResponse:
        repo = self.loader.resolve_repo(repo_url)
        manifest_path = self._manifest_path(repo.repo_id)
        deleted_manifest = False
        if manifest_path.exists():
            manifest_path.unlink()
            deleted_manifest = True

        deleted_vector_index = self.vector_store.delete_repo(repo.repo_id)
        cache_deleted = deleted_manifest or deleted_vector_index
        if cache_deleted:
            message = f"Cleared cached data for {repo.repo_name}."
        else:
            message = f"No cached data was found for {repo.repo_name}."
        return DeleteRepoCacheResponse(
            status="cleared" if cache_deleted else "not_found",
            repo_url=repo.normalized_repo_url,
            repo_name=repo.repo_name,
            cache_deleted=cache_deleted,
            deleted_manifest=deleted_manifest,
            deleted_vector_index=deleted_vector_index,
            message=message,
        )

    def clear_all_cache(self) -> ClearAllCacheResponse:
        deleted_manifests = 0
        for manifest_path in self.settings.manifest_dir.glob("*.json"):
            manifest_path.unlink()
            deleted_manifests += 1

        deleted_vector_indexes = self.vector_store.delete_all()
        message = (
            (
                f"Cleared all cached repository data "
                f"({deleted_manifests} manifests, {deleted_vector_indexes} vector indexes)."
            )
            if deleted_manifests or deleted_vector_indexes
            else "No cached repository data was found."
        )
        return ClearAllCacheResponse(
            status="cleared" if deleted_manifests or deleted_vector_indexes else "not_found",
            deleted_manifests=deleted_manifests,
            deleted_vector_indexes=deleted_vector_indexes,
            message=message,
        )

    def _manifest_path(self, repo_id: str) -> Path:
        return self.settings.manifest_dir / f"{repo_id}.json"

    def _load_manifest(self, repo_id: str) -> RepoManifest | None:
        path = self._manifest_path(repo_id)
        if not path.exists():
            return None
        return RepoManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: RepoManifest) -> None:
        path = self._manifest_path(manifest.repo.repo_id)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

settings = get_settings()
service = ResearchAssistantService(settings)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=utc_now())


@app.post("/analyze-repo", response_model=AnalyzeRepoResponse)
async def analyze_repo(request: RepoURLRequest) -> AnalyzeRepoResponse:
    try:
        return service.analyze_repo(request.repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    try:
        return service.ask(request.repo_url, request.question)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.get("/history", response_model=HistoryResponse)
async def repo_history(repo_url: str = Query(..., min_length=10)) -> HistoryResponse:
    try:
        return service.get_repo_history(repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.get("/branches", response_model=BranchListResponse)
async def list_branches(repo_url: str = Query(..., min_length=10)) -> BranchListResponse:
    try:
        return service.list_branches(repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.get("/versions", response_model=VersionListResponse)
async def list_versions(repo_url: str = Query(..., min_length=10)) -> VersionListResponse:
    try:
        return service.list_versions(repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.post("/compare-branches", response_model=CompareBranchesResponse)
async def compare_branches(request: CompareBranchesRequest) -> CompareBranchesResponse:
    try:
        return service.compare_branches(
            request.repo_url,
            request.base_branch,
            request.head_branch,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.post("/compare-versions", response_model=CompareVersionsResponse)
async def compare_versions(request: CompareVersionsRequest) -> CompareVersionsResponse:
    try:
        return service.compare_versions(
            request.repo_url,
            request.base_ref,
            request.head_ref,
            request.base_label,
            request.head_label,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.post("/ask-compare", response_model=AskCompareResponse)
async def ask_compare(request: AskCompareRequest) -> AskCompareResponse:
    try:
        return service.ask_compare(request.compare_id, request.question)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.post("/ask-version-compare", response_model=AskVersionCompareResponse)
async def ask_version_compare(request: AskVersionCompareRequest) -> AskVersionCompareResponse:
    try:
        return service.ask_version_compare(request.compare_id, request.question)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.get("/repo-summary", response_model=RepoSummary)
async def repo_summary(repo_url: str = Query(..., min_length=10)) -> RepoSummary:
    try:
        return service.get_repo_summary(repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.get("/onboarding", response_model=OnboardingResponse)
async def get_onboarding(repo_url: str = Query(..., min_length=10)) -> OnboardingResponse:
    try:
        return service.get_onboarding(repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.delete("/cache", response_model=ClearAllCacheResponse)
async def clear_all_cache() -> ClearAllCacheResponse:
    try:
        return service.clear_all_cache()
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


@app.delete("/cache/repo", response_model=DeleteRepoCacheResponse)
async def clear_repo_cache(repo_url: str = Query(..., min_length=10)) -> DeleteRepoCacheResponse:
    try:
        return service.clear_repo_cache(repo_url)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc


def _to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, AuthenticationError):
        return HTTPException(
            status_code=401,
            detail="OpenAI authentication failed. Check OPENAI_API_KEY in .env, replace any placeholder value, and restart uvicorn.",
        )
    if isinstance(exc, RateLimitError):
        return HTTPException(
            status_code=429,
            detail="OpenAI rate limit reached. Wait a moment or use an API key with available quota.",
        )
    if isinstance(exc, APIConnectionError):
        return HTTPException(
            status_code=502,
            detail="Could not reach the OpenAI API. Check your network connection and try again.",
        )
    if isinstance(exc, requests.RequestException):
        return HTTPException(
            status_code=502,
            detail="Could not reach the GitHub API. Check your network connection and try again.",
        )
    if isinstance(exc, BadRequestError):
        return HTTPException(status_code=400, detail="OpenAI rejected the request. Check your configured models and request payload.")
    logger.exception("Unhandled backend error", exc_info=exc)
    return HTTPException(status_code=500, detail="Unexpected server error.")
