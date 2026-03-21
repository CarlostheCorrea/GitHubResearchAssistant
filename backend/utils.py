from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_github_repo_url(repo_url: str) -> tuple[str, str, str | None]:
    parsed = urlparse(repo_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Repository URL must start with http:// or https://.")
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise ValueError("Only public GitHub repository URLs are supported.")

    parts = [segment for segment in parsed.path.split("/") if segment]
    if len(parts) < 2:
        raise ValueError("Repository URL must point to a GitHub owner and repository.")

    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    branch: str | None = None

    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        branch = parts[3]

    return owner, repo, branch


def build_repo_id(owner: str, repo: str, branch: str) -> str:
    seed = f"{owner}/{repo}@{branch}".lower()
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def normalize_repo_url(owner: str, repo: str, branch: str) -> str:
    return f"https://github.com/{owner}/{repo}/tree/{branch}"


def safe_collection_name(repo_id: str) -> str:
    return f"repo_{repo_id}".replace("-", "_")


def detect_language(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(suffix, "text")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "item"


def hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def build_chunk_id(
    repo_id: str,
    file_path: str,
    chunk_type: str,
    symbol_name: str | None,
    start_line: int | None,
    end_line: int | None,
    text: str,
) -> str:
    seed = "|".join(
        [
            repo_id,
            file_path,
            chunk_type,
            symbol_name or "",
            str(start_line or 0),
            str(end_line or 0),
            hash_text(text)[:12],
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def truncate_text(text: str, max_chars: int = 800) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_nonempty_lines(text: str, max_lines: int = 4) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:max_lines]


def slice_lines(text: str, start_line: int | None, end_line: int | None) -> str:
    if start_line is None or end_line is None:
        return text
    lines = text.splitlines()
    start_index = max(start_line - 1, 0)
    end_index = min(end_line, len(lines))
    return "\n".join(lines[start_index:end_index])


def line_range_label(start_line: int | None, end_line: int | None) -> str:
    if not start_line and not end_line:
        return "lines unavailable"
    if start_line == end_line:
        return f"line {start_line}"
    return f"lines {start_line}-{end_line}"


def tokenize_for_matching(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_/-]+", text.lower()) if len(token) > 2}


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def seems_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    if not data:
        return False
    printable = sum(1 for byte in data if 32 <= byte <= 126 or byte in {9, 10, 13})
    return printable / len(data) < 0.75


def looks_like_placeholder_secret(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    placeholder_values = {
        "your_openai_api_key",
        "your_actual_openai_api_key",
        "your_api_key",
        "replace_me",
    }
    return normalized in placeholder_values or normalized.startswith(("your_", "<your", "paste_", "replace_"))
