# Product Spec

## Product Name

AI GitHub Research Assistant

## Purpose

AI GitHub Research Assistant helps a user understand an unfamiliar public GitHub repository without manually reading the entire codebase. It ingests repository files, builds a code-aware retrieval index, and answers natural-language questions with grounded citations and supporting snippets.

## Problem

Developers, recruiters, founders, students, and technical reviewers often need to understand a repository quickly. Reading the whole repo is slow, generic LLM prompting over entire files is noisy, and answers without evidence are hard to trust.

The product solves this by combining repository ingestion, chunking, retrieval, answer generation, and source-backed review into a single local workflow.

## Product Goals

1. Help users understand repository architecture faster.
2. Keep answers grounded in real repository evidence.
3. Show inspectable citations and source snippets for every answer.
4. Surface a usable repository overview before the user asks questions.
5. Keep the user-facing workflow simple: analyze, ask, inspect evidence.

## Non-Goals

1. Full static analysis or formal code verification.
2. Editing or refactoring the analyzed repository automatically.
3. Perfect semantic understanding of every language or framework.
4. Multi-repo knowledge management.
5. Hosted team collaboration or auth workflows.

## Target Users

1. Developers onboarding to a new codebase.
2. Students learning from public repositories.
3. Hiring managers or recruiters evaluating technical projects.
4. Founders or PMs reviewing implementation details without deep repo familiarity.
5. Researchers comparing repository structure and implementation patterns.

## Core User Workflow

1. User pastes a public GitHub repository URL.
2. Backend fetches and filters supported files.
3. Backend chunks the repository into retrieval units.
4. Backend creates embeddings and stores them locally in Chroma.
5. Backend produces a high-level repository summary.
6. User asks a natural-language question.
7. Backend retrieves relevant repository chunks.
8. LLM drafts an answer using only retrieved evidence.
9. Internal judge step revises the answer if needed.
10. UI shows the final answer, citations, and source snippets.

## Functional Scope

### Repository Analysis

- Accept a public GitHub repository URL.
- Resolve owner, repo, and branch metadata.
- Fetch repository files through the GitHub API.
- Filter out unsupported or irrelevant files.
- Build structured chunks from supported repository content.
- Persist repository analysis artifacts locally for reuse.

### Code-Aware Retrieval

- Support code-aware chunking for Python and logical chunking for other text/config formats.
- Store embeddings in a local Chroma vector index.
- Retrieve relevant chunks for a question using hybrid scoring.
- Preserve metadata such as file path, chunk type, role, symbol name, and line range.

### Repository Overview

- Show a high-level summary of what the repo appears to do.
- Surface detected languages and indexed footprint.
- Highlight key files and likely entry points.
- Highlight likely training, inference, config, and data-loading files.

### Question Answering

- Accept a natural-language repository question.
- Use only retrieved repository evidence to answer.
- Return inline citations with file paths and line ranges.
- Explicitly handle insufficient evidence when context is missing.

### Answer Review

- Run an internal LLM-as-a-Judge pass after draft answer generation.
- Prefer a usable revised final answer over strict diagnostic scoring.
- Fall back to the draft answer if judge formatting or API behavior fails.

### Evidence Inspection

- Show retrieved source snippets in the UI.
- Show chunk metadata and retrieval scores.
- Make it easy for the user to inspect where the answer came from.

### Optional Knowledge Graph

- Optionally project repository structure into Neo4j.
- Create graph entities for repositories, files, languages, and symbols.
- Store structural relationships such as repository-to-file and file-to-symbol.
- Keep graph sync optional and non-blocking for the main app workflow.

## API-Level Capabilities

- `POST /analyze-repo`
  - Analyze a repository and return repo summary data.
- `POST /ask`
  - Return final answer, sources, and repo summary.
- `GET /repo-summary`
  - Return the cached or generated repository summary.
- `GET /health`
  - Return service health status.
- `DELETE /cache`
  - Clear cached manifests and vector indexes.

## Inputs

- GitHub repository URL
- User question
- Environment configuration for OpenAI and optional GitHub token
- Optional Neo4j connection settings

## Outputs

- Repository summary
- Grounded answer
- Inline citations
- Retrieved source snippets
- Local cache artifacts
- Optional Neo4j knowledge graph records

## Success Criteria

1. User can analyze a public repository without manual preprocessing.
2. User receives a grounded answer with supporting evidence.
3. Answers do not fail when the internal judge returns partial or nonstandard diagnostics.
4. Repository overview helps the user ask better follow-up questions.
5. The app remains usable on repeated analyses through caching.

## Constraints

1. Depends on external APIs for GitHub content and OpenAI inference.
2. Quality depends on repository file quality, retrieval coverage, and prompt behavior.
3. Large or unusual repositories may have incomplete coverage.
4. Native Python dependencies can make setup more fragile across environments.
5. Optional Neo4j support adds infrastructure only when explicitly configured.

## Future Directions

1. Broader parser support across more languages.
2. Stronger evaluation and retrieval benchmarks.
3. Better cache refresh and invalidation controls.
4. Larger-repo background indexing.
5. Richer graph-based exploration and graph-assisted retrieval.
