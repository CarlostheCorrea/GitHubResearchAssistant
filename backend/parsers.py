from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from backend.utils import compact_whitespace, first_nonempty_lines, truncate_text


@dataclass(slots=True)
class SectionSpan:
    chunk_type: str
    text: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    short_summary: str | None = None


def extract_python_sections(content: str) -> list[SectionSpan]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.splitlines()
    sections: list[SectionSpan] = []
    class_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            symbol = ".".join(class_stack + [node.name]) if class_stack else node.name
            sections.append(
                SectionSpan(
                    chunk_type="python_class",
                    text=_lines_to_text(lines, node.lineno, node.end_lineno or node.lineno),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    symbol_name=symbol,
                    short_summary=_python_docstring_summary(node, default=f"Class {symbol}"),
                )
            )
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            symbol = ".".join(class_stack + [node.name]) if class_stack else node.name
            chunk_type = "python_method" if class_stack else "python_function"
            sections.append(
                SectionSpan(
                    chunk_type=chunk_type,
                    text=_lines_to_text(lines, node.lineno, node.end_lineno or node.lineno),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    symbol_name=symbol,
                    short_summary=_python_docstring_summary(node, default=f"{chunk_type.replace('_', ' ')} {symbol}"),
                )
            )
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.visit_FunctionDef(node)

    Visitor().visit(tree)

    top_level_symbols = [section.symbol_name for section in sections if section.symbol_name][:8]
    module_summary_text = _python_module_summary(content, top_level_symbols)
    sections.insert(
        0,
        SectionSpan(
            chunk_type="python_module_summary",
            text=module_summary_text,
            start_line=1,
            end_line=min(len(lines), 40),
            short_summary="Heuristic module summary",
        ),
    )
    return sections


def split_markdown_sections(content: str) -> list[SectionSpan]:
    lines = content.splitlines()
    if not lines:
        return []

    heading_indices = [
        index for index, line in enumerate(lines, start=1) if re.match(r"^#{1,6}\s+\S", line)
    ]
    if not heading_indices:
        return _split_by_blank_lines(content, chunk_type="markdown_block")

    boundaries = [1, *heading_indices]
    unique_boundaries: list[int] = []
    for boundary in boundaries:
        if boundary not in unique_boundaries:
            unique_boundaries.append(boundary)
    unique_boundaries.append(len(lines) + 1)

    sections: list[SectionSpan] = []
    for start, end in zip(unique_boundaries, unique_boundaries[1:]):
        section_lines = lines[start - 1 : end - 1]
        text = "\n".join(section_lines).strip()
        if not text:
            continue
        heading = next((line.lstrip("# ").strip() for line in section_lines if line.startswith("#")), None)
        sections.append(
            SectionSpan(
                chunk_type="markdown_section",
                text=text,
                start_line=start,
                end_line=end - 1,
                symbol_name=heading,
                short_summary=heading or truncate_text(compact_whitespace(text), 100),
            )
        )
    return sections


def split_structured_sections(file_path: str, content: str) -> list[SectionSpan]:
    suffix = file_path.lower().split(".")[-1]
    if suffix == "json":
        return _split_json_sections(content)
    return _split_root_key_sections(content)


def split_code_sections(content: str) -> list[SectionSpan]:
    lines = content.splitlines()
    if not lines:
        return []

    declaration_pattern = re.compile(
        r"^\s*(export\s+)?(async\s+)?function\s+\w+|^\s*class\s+\w+|^\s*(const|let|var)\s+\w+\s*=\s*(async\s*)?\(|^\s*(interface|type|enum)\s+\w+"
    )
    boundaries = [1]
    for index, line in enumerate(lines, start=1):
        if index == 1:
            continue
        if declaration_pattern.search(line):
            boundaries.append(index)
    boundaries.append(len(lines) + 1)

    sections: list[SectionSpan] = []
    for start, end in zip(boundaries, boundaries[1:]):
        section_lines = lines[start - 1 : end - 1]
        text = "\n".join(section_lines).strip()
        if not text:
            continue
        sections.append(
            SectionSpan(
                chunk_type="code_block",
                text=text,
                start_line=start,
                end_line=end - 1,
                short_summary=truncate_text(compact_whitespace(text), 100),
            )
        )

    if len(sections) <= 1:
        return _split_by_blank_lines(content, chunk_type="code_block")
    return _merge_small_sections(sections)


def split_text_sections(content: str) -> list[SectionSpan]:
    return _split_by_blank_lines(content, chunk_type="text_block")


def build_file_summary(file_path: str, language: str, role: str, content: str) -> str:
    first_lines = first_nonempty_lines(content, max_lines=3)
    preview = compact_whitespace(" ".join(first_lines))
    summary_bits = [f"{file_path} is a {language} file"]
    if role != "general":
        summary_bits.append(f"that likely handles {role.replace('_', ' ')}")
    if preview:
        summary_bits.append(f"Preview: {truncate_text(preview, 180)}")
    return ". ".join(summary_bits) + "."


def _split_json_sections(content: str) -> list[SectionSpan]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _split_root_key_sections(content)

    if not isinstance(parsed, dict):
        return [
            SectionSpan(
                chunk_type="config_section",
                text=truncate_text(json.dumps(parsed, indent=2), 1600),
                start_line=1,
                end_line=max(len(content.splitlines()), 1),
                short_summary="Structured JSON content",
            )
        ]

    lines = content.splitlines()
    sections: list[SectionSpan] = []
    for key in list(parsed.keys())[:12]:
        pattern = re.compile(rf'^\s*"{re.escape(str(key))}"\s*:')
        start_line = next((idx for idx, line in enumerate(lines, start=1) if pattern.search(line)), 1)
        value = json.dumps({key: parsed[key]}, indent=2)
        sections.append(
            SectionSpan(
                chunk_type="config_section",
                text=value,
                start_line=start_line,
                end_line=min(start_line + value.count("\n"), len(lines) or 1),
                symbol_name=str(key),
                short_summary=f"Top-level JSON key: {key}",
            )
        )
    return sections or _split_root_key_sections(content)


def _split_root_key_sections(content: str) -> list[SectionSpan]:
    lines = content.splitlines()
    if not lines:
        return []

    root_key_pattern = re.compile(r"^[A-Za-z0-9_.\"'-]+\s*[:=]")
    boundaries = [1]
    for index, line in enumerate(lines, start=1):
        if index == 1:
            continue
        if root_key_pattern.match(line.strip()):
            boundaries.append(index)
    boundaries.append(len(lines) + 1)

    sections: list[SectionSpan] = []
    for start, end in zip(boundaries, boundaries[1:]):
        text = "\n".join(lines[start - 1 : end - 1]).strip()
        if not text:
            continue
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        symbol = first_line.split(":")[0].split("=")[0].strip().strip("\"'")
        sections.append(
            SectionSpan(
                chunk_type="config_section",
                text=text,
                start_line=start,
                end_line=end - 1,
                symbol_name=symbol or None,
                short_summary=truncate_text(compact_whitespace(text), 100),
            )
        )
    return _merge_small_sections(sections)


def _split_by_blank_lines(content: str, chunk_type: str, max_lines: int = 80) -> list[SectionSpan]:
    lines = content.splitlines()
    sections: list[SectionSpan] = []
    buffer: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if not text:
            return
        sections.append(
            SectionSpan(
                chunk_type=chunk_type,
                text=text,
                start_line=start_line,
                end_line=end_line,
                short_summary=truncate_text(compact_whitespace(text), 100),
            )
        )

    for index, line in enumerate(lines, start=1):
        if not buffer and line.strip():
            start_line = index
        buffer.append(line)
        if (not line.strip() and len(buffer) > 8) or len(buffer) >= max_lines:
            flush(index)
            buffer = []
    if buffer:
        flush(len(lines) or 1)
    return _merge_small_sections(sections)


def _merge_small_sections(sections: list[SectionSpan], min_lines: int = 6) -> list[SectionSpan]:
    if not sections:
        return []

    merged: list[SectionSpan] = []
    current = sections[0]
    for section in sections[1:]:
        current_lines = current.end_line - current.start_line + 1
        if current_lines < min_lines:
            current = SectionSpan(
                chunk_type=current.chunk_type,
                text=current.text.rstrip() + "\n\n" + section.text.lstrip(),
                start_line=current.start_line,
                end_line=section.end_line,
                symbol_name=current.symbol_name or section.symbol_name,
                short_summary=current.short_summary or section.short_summary,
            )
            continue
        merged.append(current)
        current = section
    merged.append(current)
    return merged


def _lines_to_text(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line]).strip()


def _python_docstring_summary(node: ast.AST, default: str) -> str:
    docstring = ast.get_docstring(node)
    if docstring:
        first_line = docstring.strip().splitlines()[0]
        return truncate_text(compact_whitespace(first_line), 120)
    return default


def _python_module_summary(content: str, symbols: list[str]) -> str:
    docstring = ""
    try:
        module = ast.parse(content)
        docstring = ast.get_docstring(module) or ""
    except SyntaxError:
        docstring = ""
    summary_parts = []
    if docstring:
        summary_parts.append(truncate_text(compact_whitespace(docstring), 220))
    if symbols:
        preview = ", ".join(symbols[:8])
        summary_parts.append(f"Top-level symbols: {preview}")
    if not summary_parts:
        summary_parts.append(truncate_text(compact_whitespace(content), 220))
    return ". ".join(summary_parts)
