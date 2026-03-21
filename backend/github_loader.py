from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import quote

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
        tree_data = self._request_json(
            f"https://api.github.com/repos/{repo.owner}/{repo.repo}/git/trees/{quote(repo.branch, safe='')}",
            params={"recursive": 1},
        )

        if tree_data.get("truncated"):
            logger.warning("GitHub tree response was truncated for %s", repo.repo_name)

        files: list[RepoFile] = []
        files_seen = 0
        skipped_files = 0
        total_bytes = 0

        for item in tree_data.get("tree", []):
            if item.get("type") != "blob":
                continue

            files_seen += 1
            path = item["path"]
            size = item.get("size") or 0
            should_ingest, _reason = file_filter.should_ingest(path, size)
            if not should_ingest:
                skipped_files += 1
                continue

            if len(files) >= self.settings.max_files_per_repo:
                skipped_files += 1
                continue

            if total_bytes + size > self.settings.max_total_repo_bytes:
                skipped_files += 1
                continue

            file_bytes = self._fetch_blob_bytes(item["url"])
            if seems_binary(file_bytes):
                skipped_files += 1
                continue

            text = file_bytes.decode("utf-8", errors="ignore")
            if not text.strip():
                skipped_files += 1
                continue

            language = detect_language(path)
            role = file_filter.classify_role(path, text)
            files.append(
                RepoFile(
                    path=path,
                    size=size,
                    sha=item.get("sha"),
                    blob_url=item.get("url"),
                    language=language,
                    role=role,
                    content=text,
                )
            )
            total_bytes += size

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

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.settings.request_timeout_seconds)
        self._raise_for_status(response)
        return response.json()

    def _fetch_blob_bytes(self, blob_url: str) -> bytes:
        blob_data = self._request_json(blob_url)
        encoding = blob_data.get("encoding")
        content = blob_data.get("content", "")
        if encoding == "base64":
            return base64.b64decode(content)
        return content.encode("utf-8")

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
