from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

from backend.models import ChunkRecord, RepoDescriptor, RepoFile, RepoSummary

logger = logging.getLogger(__name__)


@dataclass
class GraphSnapshot:
    repo_id: str
    repo_name: str
    repo_url: str
    files: list[dict[str, str]]
    symbols: list[dict[str, object]]
    languages: list[str]
    dependencies: list[dict[str, str]]
    global_context: str
    critical_paths: list[list[str]]
    dependency_links: list[list[str]]
    graph_hubs: list[str]


class KnowledgeGraphService:
    def build_snapshot(
        self,
        repo: RepoDescriptor,
        files: list[RepoFile],
        chunks: list[ChunkRecord],
    ) -> GraphSnapshot:
        file_rows = self._build_file_rows(files)
        symbol_rows = self._build_symbol_rows(chunks)
        languages = sorted({file["language"] for file in file_rows if file["language"]})
        dependencies = self._build_dependency_rows(files, chunks)
        critical_paths = self._build_critical_paths(file_rows, dependencies)
        graph_hubs = self._build_graph_hubs(dependencies)
        dependency_links = self._build_dependency_links(dependencies, graph_hubs)
        global_context = self._build_global_context(
            repo=repo,
            file_rows=file_rows,
            symbol_rows=symbol_rows,
            dependencies=dependencies,
            critical_paths=critical_paths,
        )
        return GraphSnapshot(
            repo_id=repo.repo_id,
            repo_name=repo.repo_name,
            repo_url=repo.normalized_repo_url,
            files=file_rows,
            symbols=symbol_rows,
            languages=languages,
            dependencies=dependencies,
            global_context=global_context,
            critical_paths=critical_paths,
            dependency_links=dependency_links,
            graph_hubs=graph_hubs,
        )

    def ensure_summary_global_context(self, summary: RepoSummary) -> RepoSummary:
        if summary.global_context.strip() and (summary.critical_paths or summary.dependency_links or summary.graph_hubs):
            return summary

        hydrated = summary.model_copy(deep=True)
        hydrated.global_context = self._build_summary_fallback_global_context(hydrated)
        if not hydrated.critical_paths:
            hydrated.critical_paths = self._build_summary_fallback_critical_paths(hydrated)
        if not hydrated.dependency_links:
            hydrated.dependency_links = [path[:2] for path in hydrated.critical_paths if len(path) >= 2][:4]
        if not hydrated.graph_hubs:
            hydrated.graph_hubs = self._build_summary_fallback_graph_hubs(hydrated)
        return hydrated

    def _build_file_rows(self, files: list[RepoFile]) -> list[dict[str, str]]:
        rows = [
            {
                "path": repo_file.path,
                "language": repo_file.language,
                "role": repo_file.role or "general",
            }
            for repo_file in files
        ]
        rows.sort(key=lambda row: row["path"])
        return rows

    def _build_symbol_rows(self, chunks: list[ChunkRecord]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            if not chunk.symbol_name:
                continue
            key = (chunk.file_path, chunk.symbol_name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "file_path": chunk.file_path,
                    "name": chunk.symbol_name,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "start_line": chunk.start_line or 0,
                    "end_line": chunk.end_line or 0,
                }
            )
        rows.sort(key=lambda row: (row["file_path"], row["name"]))
        return rows

    def _build_dependency_rows(
        self,
        files: list[RepoFile],
        chunks: list[ChunkRecord],
    ) -> list[dict[str, str]]:
        del chunks
        python_module_to_path = self._build_python_module_index(files)
        path_lookup = {repo_file.path: repo_file for repo_file in files}
        dependencies: set[tuple[str, str]] = set()
        for repo_file in files:
            lower_language = repo_file.language.lower()
            if lower_language == "python":
                for imported_module in self._extract_python_imports(repo_file.content):
                    target_path = python_module_to_path.get(imported_module)
                    if target_path and target_path != repo_file.path:
                        dependencies.add((repo_file.path, target_path))
            elif lower_language in {"javascript", "typescript"}:
                for import_target in self._extract_js_imports(repo_file.content):
                    target_path = self._resolve_relative_import(repo_file.path, import_target, path_lookup)
                    if target_path and target_path != repo_file.path:
                        dependencies.add((repo_file.path, target_path))
        return [
            {"source_path": source_path, "target_path": target_path}
            for source_path, target_path in sorted(dependencies)
        ]

    def _build_python_module_index(self, files: list[RepoFile]) -> dict[str, str]:
        index: dict[str, str] = {}
        for repo_file in files:
            if repo_file.language.lower() != "python" or not repo_file.path.endswith(".py"):
                continue
            path = Path(repo_file.path)
            parts = list(path.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            module_name = ".".join(parts)
            index[module_name] = repo_file.path
        return index

    def _extract_python_imports(self, text: str) -> set[str]:
        imports: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            import_match = re.match(r"import\s+([a-zA-Z0-9_.,\s]+)", stripped)
            if import_match:
                for module_name in import_match.group(1).split(","):
                    cleaned = module_name.strip().split(" as ")[0].strip()
                    if cleaned:
                        imports.add(cleaned)
                continue
            from_match = re.match(r"from\s+([a-zA-Z0-9_\.]+)\s+import\s+", stripped)
            if from_match:
                imports.add(from_match.group(1).strip())
        return imports

    def _extract_js_imports(self, text: str) -> set[str]:
        imports: set[str] = set()
        patterns = [
            r"import\s+(?:.+?\s+from\s+)?['\"]([^'\"]+)['\"]",
            r"require\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                imports.add(match.strip())
        return {item for item in imports if item.startswith(".")}

    def _resolve_relative_import(
        self,
        file_path: str,
        import_target: str,
        path_lookup: dict[str, RepoFile],
    ) -> str | None:
        base_dir = Path(file_path).parent
        candidate_base = (base_dir / import_target).resolve().relative_to(Path.cwd().resolve()) if False else base_dir / import_target
        candidate_strings = []
        if candidate_base.suffix:
            candidate_strings.append(candidate_base.as_posix())
        else:
            candidate_strings.extend(
                [
                    candidate_base.as_posix(),
                    f"{candidate_base.as_posix()}.js",
                    f"{candidate_base.as_posix()}.ts",
                    f"{candidate_base.as_posix()}.jsx",
                    f"{candidate_base.as_posix()}.tsx",
                    f"{candidate_base.as_posix()}/index.js",
                    f"{candidate_base.as_posix()}/index.ts",
                ]
            )
        for candidate in candidate_strings:
            if candidate in path_lookup:
                return candidate
        return None

    def _build_critical_paths(
        self,
        file_rows: list[dict[str, str]],
        dependencies: list[dict[str, str]],
    ) -> list[list[str]]:
        if not dependencies:
            return []

        adjacency: dict[str, list[str]] = {}
        for dependency in dependencies:
            adjacency.setdefault(dependency["source_path"], []).append(dependency["target_path"])

        entrypoints = [row["path"] for row in file_rows if row["role"] == "entrypoint"]
        if not entrypoints:
            entrypoints = [row["path"] for row in file_rows if Path(row["path"]).name.startswith(("main", "app", "run"))][:3]

        target_roles = {"training", "inference", "config", "data_loading"}
        targets = [row["path"] for row in file_rows if row["role"] in target_roles]
        if not targets:
            targets = sorted({dependency["target_path"] for dependency in dependencies})

        paths: list[list[str]] = []
        seen_paths: set[tuple[str, ...]] = set()
        for entrypoint in entrypoints[:4]:
            for target in targets[:8]:
                path = self._shortest_path(adjacency, entrypoint, target)
                if path and len(path) > 1 and tuple(path) not in seen_paths:
                    seen_paths.add(tuple(path))
                    paths.append(path)
        paths.sort(key=lambda path: (len(path), path))
        return paths[:4]

    def _build_graph_hubs(self, dependencies: list[dict[str, str]]) -> list[str]:
        counts: dict[str, int] = {}
        for dependency in dependencies:
            counts[dependency["source_path"]] = counts.get(dependency["source_path"], 0) + 1
            counts[dependency["target_path"]] = counts.get(dependency["target_path"], 0) + 1
        return [path for path, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]]

    def _build_dependency_links(
        self,
        dependencies: list[dict[str, str]],
        graph_hubs: list[str],
    ) -> list[list[str]]:
        prioritized: list[list[str]] = []
        seen: set[tuple[str, str]] = set()
        hub_set = set(graph_hubs)
        for dependency in dependencies:
            edge = (dependency["source_path"], dependency["target_path"])
            if edge in seen:
                continue
            if dependency["source_path"] in hub_set or dependency["target_path"] in hub_set:
                prioritized.append([dependency["source_path"], dependency["target_path"]])
                seen.add(edge)
        for dependency in dependencies:
            edge = (dependency["source_path"], dependency["target_path"])
            if edge in seen:
                continue
            prioritized.append([dependency["source_path"], dependency["target_path"]])
            seen.add(edge)
        return prioritized[:6]

    def _shortest_path(
        self,
        adjacency: dict[str, list[str]],
        source: str,
        target: str,
    ) -> list[str] | None:
        if source == target:
            return [source]
        queue: list[list[str]] = [[source]]
        seen = {source}
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in adjacency.get(node, []):
                if neighbor in seen:
                    continue
                next_path = path + [neighbor]
                if neighbor == target:
                    return next_path
                seen.add(neighbor)
                queue.append(next_path)
        return None

    def _build_global_context(
        self,
        repo: RepoDescriptor,
        file_rows: list[dict[str, str]],
        symbol_rows: list[dict[str, object]],
        dependencies: list[dict[str, str]],
        critical_paths: list[list[str]],
    ) -> str:
        role_map = {
            "entrypoint": [],
            "training": [],
            "inference": [],
            "config": [],
            "data_loading": [],
        }
        outgoing_counts: dict[str, int] = {}
        incoming_counts: dict[str, int] = {}

        for file_row in file_rows:
            role = file_row["role"]
            if role in role_map:
                role_map[role].append(file_row["path"])

        for dependency in dependencies:
            outgoing_counts[dependency["source_path"]] = outgoing_counts.get(dependency["source_path"], 0) + 1
            incoming_counts[dependency["target_path"]] = incoming_counts.get(dependency["target_path"], 0) + 1

        most_connected = sorted(
            {path for path in outgoing_counts} | {path for path in incoming_counts},
            key=lambda path: (outgoing_counts.get(path, 0) + incoming_counts.get(path, 0), path),
            reverse=True,
        )[:3]

        sentences = [
            (
                f"Repository-wide graph context for {repo.repo_name}: "
                f"{len(file_rows)} files, {len(symbol_rows)} named symbols, and {len(dependencies)} inferred file dependency links."
            )
        ]
        if role_map["entrypoint"]:
            sentences.append(f"Likely entrypoint files: {', '.join(role_map['entrypoint'][:3])}.")
        if role_map["training"]:
            sentences.append(f"Training-related files: {', '.join(role_map['training'][:3])}.")
        if role_map["inference"]:
            sentences.append(f"Inference-related files: {', '.join(role_map['inference'][:3])}.")
        if role_map["config"]:
            sentences.append(f"Configuration files: {', '.join(role_map['config'][:3])}.")
        if role_map["data_loading"]:
            sentences.append(f"Data-loading files: {', '.join(role_map['data_loading'][:3])}.")
        if most_connected:
            sentences.append(f"Most connected files in the inferred dependency graph: {', '.join(most_connected)}.")
        if critical_paths:
            preview = " | ".join(" -> ".join(path) for path in critical_paths[:2])
            sentences.append(f"Critical dependency paths: {preview}.")
        return " ".join(sentences)

    def _build_summary_fallback_global_context(self, summary: RepoSummary) -> str:
        sentences = [
            (
                f"Repository-wide graph context for {summary.repo_name}: "
                f"{summary.files_indexed} indexed files and {summary.chunks_indexed} retrieval chunks."
            )
        ]
        if summary.probable_entry_points:
            sentences.append(f"Likely entrypoint files: {', '.join(summary.probable_entry_points[:3])}.")
        if summary.probable_training_files:
            sentences.append(f"Training-related files: {', '.join(summary.probable_training_files[:3])}.")
        if summary.probable_inference_files:
            sentences.append(f"Inference-related files: {', '.join(summary.probable_inference_files[:3])}.")
        if summary.probable_config_files:
            sentences.append(f"Configuration files: {', '.join(summary.probable_config_files[:3])}.")
        if summary.probable_data_files:
            sentences.append(f"Data-loading files: {', '.join(summary.probable_data_files[:3])}.")
        if summary.key_files:
            sentences.append(f"Key repository files include: {', '.join(summary.key_files[:4])}.")
        if summary.critical_paths:
            sentences.append(
                "Critical dependency paths: "
                + " | ".join(" -> ".join(path) for path in summary.critical_paths[:2])
                + "."
            )
        return " ".join(sentences)

    def _build_summary_fallback_critical_paths(self, summary: RepoSummary) -> list[list[str]]:
        if summary.probable_entry_points and summary.probable_config_files:
            return [[summary.probable_entry_points[0], summary.probable_config_files[0]]]
        if summary.probable_entry_points and summary.probable_data_files:
            return [[summary.probable_entry_points[0], summary.probable_data_files[0]]]
        return []

    def _build_summary_fallback_graph_hubs(self, summary: RepoSummary) -> list[str]:
        return [
            *summary.probable_entry_points[:2],
            *summary.probable_config_files[:1],
            *summary.probable_data_files[:1],
        ]
