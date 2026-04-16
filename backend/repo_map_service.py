from __future__ import annotations

import re
from collections import defaultdict

from backend.models import ChunkRecord, RepoMapEdge, RepoMapNode, RepoMapResponse
from backend.vector_store import ChromaVectorStore


class RepoMapService:
    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store

    def get_map(
        self,
        repo_id: str,
        repo_name: str,
        known_edges: list[list[str]] | None = None,
    ) -> RepoMapResponse:
        chunks = self.vector_store.get_all_chunks(repo_id)
        if not chunks:
            return RepoMapResponse(repo_name=repo_name)
        nodes = self._build_nodes(chunks)
        valid_paths: set[str] = {n.id for n in nodes}

        # Primary: use pre-computed edges from the knowledge graph (built from
        # actual file content during analysis — much more accurate than chunk parsing)
        if known_edges:
            seen: set[tuple[str, str]] = set()
            edges: list[RepoMapEdge] = []
            for pair in known_edges:
                if len(pair) >= 2:
                    src, tgt = pair[0], pair[1]
                    if src in valid_paths and tgt in valid_paths and (src, tgt) not in seen:
                        edges.append(RepoMapEdge(source=src, target=tgt))
                        seen.add((src, tgt))
        else:
            # Fallback: try to extract from chunk texts (works for some languages)
            edges = self._build_edges(chunks, valid_paths)

        return RepoMapResponse(repo_name=repo_name, nodes=nodes, edges=edges)

    # ── Node building ─────────────────────────────────────────────────────────

    def _build_nodes(self, chunks: list[ChunkRecord]) -> list[RepoMapNode]:
        file_data: dict[str, dict] = defaultdict(lambda: {
            "language": "unknown",
            "role": "general",
            "max_line": 0,
            "symbols": [],
            "summaries": [],
        })
        for chunk in chunks:
            fd = file_data[chunk.file_path]
            fd["language"] = chunk.language
            if chunk.file_role:
                fd["role"] = chunk.file_role
            if chunk.end_line:
                fd["max_line"] = max(fd["max_line"], chunk.end_line)
            if chunk.symbol_name and chunk.symbol_name not in fd["symbols"]:
                fd["symbols"].append(chunk.symbol_name)
            if chunk.short_summary and chunk.short_summary not in fd["summaries"]:
                fd["summaries"].append(chunk.short_summary)

        nodes: list[RepoMapNode] = []
        for path, data in file_data.items():
            summary = data["summaries"][0] if data["summaries"] else ""
            symbols = data["symbols"][:8]
            line_count = data["max_line"] if data["max_line"] > 0 else 50
            nodes.append(RepoMapNode(
                id=path,
                file_path=path,
                language=data["language"],
                role=data["role"],
                line_count=line_count,
                short_summary=summary,
                key_symbols=symbols,
            ))
        return nodes

    # ── Edge building ─────────────────────────────────────────────────────────

    def _build_edges(self, chunks: list[ChunkRecord], valid_paths: set[str]) -> list[RepoMapEdge]:
        file_chunks: dict[str, list[ChunkRecord]] = defaultdict(list)
        for chunk in chunks:
            file_chunks[chunk.file_path].append(chunk)

        seen_edges: set[tuple[str, str]] = set()
        edges: list[RepoMapEdge] = []

        for file_path, fchunks in file_chunks.items():
            lang = fchunks[0].language if fchunks else "unknown"
            raw_imports: set[str] = set()
            for chunk in fchunks:
                raw_imports.update(self._extract_imports(chunk.text, lang))

            for imp in raw_imports:
                target = self._resolve_import(file_path, imp, lang, valid_paths)
                if target and target != file_path:
                    key = (file_path, target)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges.append(RepoMapEdge(source=file_path, target=target))

        return edges

    def _extract_imports(self, text: str, lang: str) -> list[str]:
        imports: list[str] = []
        if lang == "python":
            for m in re.finditer(r"^import\s+([\w.]+)", text, re.MULTILINE):
                imports.append(m.group(1))
            for m in re.finditer(r"^from\s+([\w.]+)\s+import", text, re.MULTILINE):
                imports.append(m.group(1))
        elif lang in ("javascript", "typescript"):
            for m in re.finditer(r'from\s+[\'"]([^\'"]+)[\'"]', text):
                if m.group(1).startswith("."):
                    imports.append(m.group(1))
            for m in re.finditer(r'require\([\'"]([^\'"]+)[\'"]\)', text):
                if m.group(1).startswith("."):
                    imports.append(m.group(1))
        elif lang == "go":
            for m in re.finditer(r'"([a-zA-Z0-9_/.-]+)"', text):
                val = m.group(1)
                if "/" in val and not val.startswith("http"):
                    imports.append(val)
        elif lang == "java":
            for m in re.finditer(r"^import\s+([\w.]+);", text, re.MULTILINE):
                imports.append(m.group(1))
        elif lang == "rust":
            for m in re.finditer(r"^use\s+([\w:]+)", text, re.MULTILINE):
                imports.append(m.group(1))
        elif lang in ("cpp", "c"):
            for m in re.finditer(r'#include\s+"([^"]+)"', text):
                imports.append(m.group(1))
        elif lang == "ruby":
            for m in re.finditer(r"require_relative\s+['\"]([^'\"]+)['\"]", text):
                imports.append(m.group(1))
        return imports

    def _resolve_import(
        self, importer: str, imp_str: str, lang: str, all_paths: set[str]
    ) -> str | None:
        """Attempt to map an import string to an actual file path in the repo."""
        importer_dir = importer.rsplit("/", 1)[0] if "/" in importer else ""

        if lang == "python":
            if imp_str.startswith("."):
                dots = len(imp_str) - len(imp_str.lstrip("."))
                mod = imp_str.lstrip(".")
                parts = importer_dir.split("/") if importer_dir else []
                parts = parts[:max(0, len(parts) - dots + 1)]
                base = "/".join(parts + [mod.replace(".", "/")]) if mod else "/".join(parts)
            else:
                base = imp_str.replace(".", "/")
            for candidate in [f"{base}.py", f"{base}/__init__.py"]:
                if candidate in all_paths:
                    return candidate

        elif lang in ("javascript", "typescript"):
            if imp_str.startswith("."):
                raw = (importer_dir + "/" + imp_str) if importer_dir else imp_str
                parts = raw.split("/")
                normalized: list[str] = []
                for part in parts:
                    if part == "..":
                        if normalized:
                            normalized.pop()
                    elif part and part != ".":
                        normalized.append(part)
                base = "/".join(normalized)
                for candidate in [
                    base,
                    f"{base}.js", f"{base}.ts", f"{base}.tsx", f"{base}.jsx",
                    f"{base}/index.js", f"{base}/index.ts",
                ]:
                    if candidate in all_paths:
                        return candidate

        elif lang == "go":
            pkg = imp_str.rstrip("/").split("/")[-1]
            for path in all_paths:
                if path.endswith(f"/{pkg}.go"):
                    return path

        elif lang == "java":
            base = imp_str.replace(".", "/") + ".java"
            if base in all_paths:
                return base
            tail = imp_str.rsplit(".", 1)[-1] + ".java"
            for path in all_paths:
                if path.endswith(f"/{tail}"):
                    return path

        elif lang in ("cpp", "c"):
            for path in all_paths:
                if path.endswith(f"/{imp_str}") or path == imp_str:
                    return path

        elif lang == "ruby":
            base = (importer_dir + "/" + imp_str) if importer_dir else imp_str
            for ext in [".rb", ""]:
                candidate = base + ext
                if candidate in all_paths:
                    return candidate

        return None
