from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}

SKIP_DIRECTORIES = {
    ".git",
    ".github",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "__pycache__",
    "coverage",
    ".next",
    ".turbo",
    ".idea",
    ".vscode",
}

SKIP_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    ".ds_store",
}


class FileFilter:
    def __init__(self, max_file_bytes: int) -> None:
        self.max_file_bytes = max_file_bytes

    def should_ingest(self, path: str, size: int | None) -> tuple[bool, str | None]:
        normalized_parts = {part.lower() for part in Path(path).parts}
        filename = Path(path).name.lower()
        suffix = Path(path).suffix.lower()

        if normalized_parts & SKIP_DIRECTORIES:
            return False, "ignored directory"
        if filename in SKIP_FILENAMES:
            return False, "generated lockfile"
        if filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".bin")):
            return False, "binary or asset file"
        if filename.endswith((".min.js", ".min.css")):
            return False, "minified asset"
        if suffix not in SUPPORTED_EXTENSIONS:
            return False, "unsupported extension"
        if size and size > self.max_file_bytes:
            return False, "file too large"
        return True, None

    def classify_role(self, path: str, content: str) -> str:
        lower_path = path.lower()
        lower_content = content.lower()

        if any(marker in lower_path for marker in ("train", "trainer", "finetune", "fit")):
            return "training"
        if any(marker in lower_path for marker in ("infer", "predict", "serve", "endpoint", "api", "generate")):
            return "inference"
        if any(marker in lower_path for marker in ("dataset", "dataloader", "loader", "preprocess", "data")):
            return "data_loading"
        if any(marker in lower_path for marker in ("config", "settings", ".yaml", ".yml", ".toml", ".json")):
            return "config"
        if any(marker in lower_path for marker in ("main.", "app.", "server.", "cli.", "index.")):
            return "entrypoint"
        if lower_path.endswith(".md"):
            return "documentation"
        if "fastapi(" in lower_content or "__name__ == \"__main__\"" in lower_content:
            return "entrypoint"
        if "torch.utils.data" in lower_content or "dataloader" in lower_content:
            return "data_loading"
        if "optimizer" in lower_content and "loss" in lower_content:
            return "training"
        if "predict(" in lower_content or "inference" in lower_content:
            return "inference"
        return "general"

    def is_probably_important(self, path: str, role: str) -> bool:
        lower_path = path.lower()
        return role != "general" or lower_path.endswith(
            ("readme.md", "main.py", "app.py", "server.py", "index.ts", "index.js")
        )
