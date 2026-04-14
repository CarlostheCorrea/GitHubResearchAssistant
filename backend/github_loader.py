from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests

from backend.config import Settings
from backend.file_filter import FileFilter
from backend.models import RepoDescriptor, RepoFile
from backend.utils import (
    build_repo_id,
    detect_language,
    normalize_repo_url,
    parse_github_repo_url,
    seems_binary,
)


logger = logging.getLogger(__name__)


class GitHubLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": "ai-github-research-assistant",
            }
        )
        if settings.github_token:
            self.session.headers["Authorization"] = f"Bearer {settings.github_token}"

    def resolve_repo(self, repo_url: str) -> RepoDescriptor:
        owner, repo, branch = parse_github_repo_url(repo_url)
        repo_data = self._request_json(f"https://api.github.com/repos/{owner}/{repo}")
        default_branch = repo_data["default_branch"]
        selected_branch = branch or default_branch
        return RepoDescriptor(
            owner=owner,
            repo=repo,
            branch=selected_branch,
            default_branch=default_branch,
            normalized_repo_url=normalize_repo_url(owner, repo, selected_branch),
            repo_id=build_repo_id(owner, repo, selected_branch),
        )

    def load_repository_files(
        self,
        repo: RepoDescriptor,
        file_filter: FileFilter,
    ) -> tuple[list[RepoFile], dict[str, int]]:
        clone_url = self._build_clone_url(repo.owner, repo.repo)
        clone_path = Path(tempfile.mkdtemp(prefix=f"ghra_{repo.repo_id}_"))

        try:
            self._clone_repo(clone_url, repo.branch, clone_path, repo.repo_name)
            return self._walk_and_load(repo, file_filter, clone_path)
        finally:
            shutil.rmtree(clone_path, ignore_errors=True)
            logger.info("Cleaned up clone directory for %s", repo.repo_name)

    def _clone_repo(self, clone_url: str, branch: str, target: Path, repo_name: str) -> None:
        logger.info("Cloning %s (branch: %s)", repo_name, branch)
        cmd = [
            "git", "clone",
            "--depth=1",
            "--branch", branch,
            "--single-branch",
            clone_url,
            str(target),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.settings.clone_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {repo_name}: {result.stderr.strip()[:300]}"
            )

    def _walk_and_load(
        self,
        repo: RepoDescriptor,
        file_filter: FileFilter,
        clone_path: Path,
    ) -> tuple[list[RepoFile], dict[str, int]]:
        files: list[RepoFile] = []
        files_seen = 0
        skipped_files = 0

        for abs_path in sorted(clone_path.rglob("*")):
            if not abs_path.is_file():
                continue

            rel_path = abs_path.relative_to(clone_path)
            path_str = rel_path.as_posix()
            size = abs_path.stat().st_size
            files_seen += 1

            should_ingest, _reason = file_filter.should_ingest(path_str, size)
            if not should_ingest:
                skipped_files += 1
                continue

            file_bytes = abs_path.read_bytes()
            if seems_binary(file_bytes):
                skipped_files += 1
                continue

            text = file_bytes.decode("utf-8", errors="ignore")
            if not text.strip():
                skipped_files += 1
                continue

            language = detect_language(path_str)
            role = file_filter.classify_role(path_str, text)
            files.append(
                RepoFile(
                    path=path_str,
                    size=size,
                    language=language,
                    role=role,
                    content=text,
                )
            )

        stats = {
            "files_seen": files_seen,
            "files_indexed": len(files),
            "skipped_files": skipped_files,
        }
        logger.info(
            "Loaded %s text files from %s (%s files seen, %s skipped)",
            len(files),
            repo.repo_name,
            files_seen,
            skipped_files,
        )
        return files, stats

    def _build_clone_url(self, owner: str, repo: str) -> str:
        if self.settings.github_token:
            return f"https://{self.settings.github_token}@github.com/{owner}/{repo}.git"
        return f"https://github.com/{owner}/{repo}.git"

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.settings.request_timeout_seconds)
        self._raise_for_status(response)
        return response.json()

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.status_code < 400:
            return

        if response.status_code == 404:
            raise ValueError("Repository, branch, or file was not found on GitHub.")

        if (
            response.status_code == 403
            and response.headers.get("X-RateLimit-Remaining") == "0"
        ):
            raise RuntimeError(
                "GitHub API rate limit exceeded. Add a GITHUB_TOKEN to increase the limit."
            )

        detail = response.text.strip()[:300]
        raise RuntimeError(f"GitHub API request failed with status {response.status_code}: {detail}")
