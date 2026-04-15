from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from backend.config import Settings
from backend.file_filter import FileFilter
from backend.models import ChangesSummary, CommitRecord, RepoDescriptor, RepoFile
from backend.utils import (
    build_repo_id,
    detect_language,
    normalize_repo_url,
    parse_github_repo_url,
    seems_binary,
)


logger = logging.getLogger(__name__)

_MAX_DIFF_CHARS = 3_000  # truncate per-file diffs before embedding


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

    def get_head_sha(self, repo: RepoDescriptor) -> str:
        """Fetch the current HEAD commit SHA for the branch via API (no clone needed)."""
        data = self._request_json(
            f"https://api.github.com/repos/{repo.owner}/{repo.repo}/commits/{repo.branch}"
        )
        return data["sha"]

    def list_branches(self, repo: RepoDescriptor) -> list[str]:
        branches: list[str] = []
        page = 1
        while True:
            data = self._request_json(
                f"https://api.github.com/repos/{repo.owner}/{repo.repo}/branches",
                params={"per_page": 100, "page": page},
            )
            if not isinstance(data, list) or not data:
                break
            branches.extend(str(item.get("name")) for item in data if item.get("name"))
            if len(data) < 100:
                break
            page += 1
        return branches

    def compare_branches(
        self,
        repo: RepoDescriptor,
        base_branch: str,
        head_branch: str,
    ) -> dict[str, Any]:
        return self.compare_refs(repo, base_branch, head_branch)

    def compare_refs(
        self,
        repo: RepoDescriptor,
        base_ref: str,
        head_ref: str,
    ) -> dict[str, Any]:
        base = quote(base_ref, safe="")
        head = quote(head_ref, safe="")
        data = self._request_json(
            f"https://api.github.com/repos/{repo.owner}/{repo.repo}/compare/{base}...{head}"
        )
        if not isinstance(data, dict):
            raise RuntimeError("GitHub returned an unexpected version comparison response.")
        return data

    def load_repository_files(
        self,
        repo: RepoDescriptor,
        file_filter: FileFilter,
        previous_sha: str | None = None,
    ) -> tuple[list[RepoFile], dict[str, int], str, CommitRecord | None, list[CommitRecord], ChangesSummary | None, list[tuple[str, str]]]:
        """
        Clone the full repo and return:
          (files, stats, head_sha, first_commit, commit_history, changes_summary, file_diffs)
        file_diffs is a list of (file_path, diff_text) pairs populated only when
        previous_sha is supplied and differs from head_sha.
        """
        clone_url = self._build_clone_url(repo.owner, repo.repo)
        clone_path = Path(tempfile.mkdtemp(prefix=f"ghra_{repo.repo_id}_"))

        try:
            head_sha = self._clone_repo(clone_url, repo.branch, clone_path, repo.repo_name)
            files, stats = self._walk_and_load(repo, file_filter, clone_path)
            first_commit = self._get_first_commit(clone_path)
            commit_history = self._get_commit_history(clone_path)

            changes: ChangesSummary | None = None
            file_diffs: list[tuple[str, str]] = []
            if previous_sha and previous_sha != head_sha:
                changes = self._get_changes_summary(clone_path, previous_sha, head_sha)
                file_diffs = self._get_file_diffs(clone_path, previous_sha, head_sha)

            return files, stats, head_sha, first_commit, commit_history, changes, file_diffs
        finally:
            shutil.rmtree(clone_path, ignore_errors=True)
            logger.info("Cleaned up clone directory for %s", repo.repo_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clone_repo(self, clone_url: str, branch: str, target: Path, repo_name: str) -> str:
        """Full-history clone (single branch, no tags). Returns HEAD commit SHA."""
        logger.info("Cloning %s (branch: %s, full history)", repo_name, branch)
        cmd = [
            "git", "clone",
            "--single-branch",
            "--no-tags",
            "--branch", branch,
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

        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=target,
        )
        return sha_result.stdout.strip()

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

    def _get_commit_history(self, clone_path: Path) -> list[CommitRecord]:
        """Return up to max_commit_history commits with per-commit file-change lists."""
        _SENTINEL = "===COMMIT==="
        _STATUS_MAP = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed", "C": "renamed"}

        result = subprocess.run(
            [
                "git", "log",
                f"--max-count={self.settings.max_commit_history}",
                f"--pretty=format:{_SENTINEL}%H\x1f%an\x1f%ai\x1f%s",
                "--name-status",
            ],
            capture_output=True,
            text=True,
            cwd=clone_path,
        )

        commits: list[CommitRecord] = []
        current_header: str | None = None
        current_files: list[dict[str, str]] = []

        def _flush() -> None:
            if current_header is None:
                return
            parts = current_header.split("\x1f", 3)
            if len(parts) != 4:
                return
            sha, author_name, date, message = parts
            commits.append(
                CommitRecord(
                    sha=sha.strip(),
                    short_sha=sha.strip()[:7],
                    author_name=author_name.strip(),
                    date=date.strip(),
                    message=message.strip(),
                    file_changes=current_files[:],
                )
            )

        for raw_line in result.stdout.splitlines():
            if raw_line.startswith(_SENTINEL):
                _flush()
                current_header = raw_line[len(_SENTINEL):]
                current_files = []
            elif "\t" in raw_line:
                parts = raw_line.strip().split("\t")
                status_code = parts[0].strip()
                # Renames: "R100\told_name\tnew_name" — use the new name (last part)
                filepath = parts[-1].strip()
                status = _STATUS_MAP.get(status_code[0].upper(), "modified")
                if filepath:
                    current_files.append({"path": filepath, "status": status})

        _flush()
        return commits

    def _get_first_commit(self, clone_path: Path) -> CommitRecord | None:
        result = subprocess.run(
            [
                "git", "log",
                "--max-parents=0",
                "--reverse",
                "--pretty=format:%H\x1f%an\x1f%ai\x1f%s",
            ],
            capture_output=True,
            text=True,
            cwd=clone_path,
        )
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        parts = line.split("\x1f", 3)
        if len(parts) != 4:
            return None
        sha, author_name, date, message = parts
        return CommitRecord(
            sha=sha.strip(),
            short_sha=sha.strip()[:7],
            author_name=author_name.strip(),
            date=date.strip(),
            message=message.strip(),
        )

    def _get_changes_summary(
        self, clone_path: Path, old_sha: str, new_sha: str
    ) -> ChangesSummary | None:
        """Parse `git diff --stat` between two SHAs."""
        stat_result = subprocess.run(
            ["git", "diff", "--stat", f"{old_sha}..{new_sha}"],
            capture_output=True,
            text=True,
            cwd=clone_path,
        )
        if stat_result.returncode != 0:
            logger.warning("git diff --stat failed between %s and %s", old_sha[:7], new_sha[:7])
            return None

        # --name-status gives per-file status codes: A (added), M (modified), D (deleted), R (renamed)
        status_result = subprocess.run(
            ["git", "diff", "--name-status", f"{old_sha}..{new_sha}"],
            capture_output=True,
            text=True,
            cwd=clone_path,
        )
        changed_files: list[str] = []
        files_added = files_removed = files_modified = 0
        for line in status_result.stdout.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            status_code, filepath = parts[0].strip(), parts[1].strip()
            changed_files.append(filepath)
            if status_code.startswith("A"):
                files_added += 1
            elif status_code.startswith("D"):
                files_removed += 1
            else:
                files_modified += 1  # M, R, C, T all count as modified

        # Parse the summary line for total line counts: "3 files changed, 45 insertions(+), 12 deletions(-)"
        last_line = stat_result.stdout.strip().splitlines()[-1] if stat_result.stdout.strip() else ""
        fc_match = re.search(r"(\d+) file", last_line)
        ins_match = re.search(r"(\d+) insertion", last_line)
        del_match = re.search(r"(\d+) deletion", last_line)

        return ChangesSummary(
            old_sha=old_sha,
            new_sha=new_sha,
            files_changed=int(fc_match.group(1)) if fc_match else len(changed_files),
            insertions=int(ins_match.group(1)) if ins_match else 0,
            deletions=int(del_match.group(1)) if del_match else 0,
            changed_files=changed_files,
            files_added=files_added,
            files_removed=files_removed,
            files_modified=files_modified,
        )

    def _get_file_diffs(
        self, clone_path: Path, old_sha: str, new_sha: str
    ) -> list[tuple[str, str]]:
        """Return per-file diff texts (truncated) for embeddable diff chunks."""
        files_result = subprocess.run(
            ["git", "diff", "--name-only", f"{old_sha}..{new_sha}"],
            capture_output=True,
            text=True,
            cwd=clone_path,
        )
        changed_files = [f for f in files_result.stdout.strip().splitlines() if f.strip()]

        file_diffs: list[tuple[str, str]] = []
        for file_path in changed_files[:50]:  # cap at 50 files
            diff_result = subprocess.run(
                ["git", "diff", f"{old_sha}..{new_sha}", "--", file_path],
                capture_output=True,
                text=True,
                cwd=clone_path,
            )
            diff_text = diff_result.stdout.strip()
            if not diff_text:
                continue
            if len(diff_text) > _MAX_DIFF_CHARS:
                diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n... [diff truncated]"
            file_diffs.append((file_path, diff_text))

        return file_diffs

    def _build_clone_url(self, owner: str, repo: str) -> str:
        if self.settings.github_token:
            return f"https://{self.settings.github_token}@github.com/{owner}/{repo}.git"
        return f"https://github.com/{owner}/{repo}.git"

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
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
