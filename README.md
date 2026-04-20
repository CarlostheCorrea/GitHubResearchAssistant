# AI GitHub Research Assistant

AI GitHub Research Assistant is a production-style MVP for understanding public GitHub repositories with code-aware retrieval-augmented generation. A user pastes a repository URL, the system ingests the repo, builds a local vector index over structured code chunks, uses hybrid search to retrieve the most relevant evidence, and answers repository questions with grounded evidence, citations, and source snippets. It also supports GraphRAG-style global context through a local repository knowledge graph.

## Program Summary

This project is designed to help a user quickly understand an unfamiliar codebase. Instead of sending entire files or the whole repository into an LLM, it first fetches the repository, filters relevant files, chunks the code intelligently, embeds those chunks, and stores them in a local Chroma vector index. When the user asks a question, the app uses hybrid search to retrieve the most relevant chunks, generates an evidence-based answer, and shows the supporting snippets in the UI.

The goal is not just to answer questions, but to answer them in a way that is inspectable. The app highlights where the answer came from, which files were used, and which parts of the repository appear to be the key architectural areas such as entry points, configuration, data loading, and inference or training logic.

Beyond question answering, the app offers four workspace tabs that each give a different lens into the repository: Q&A with source citations, Version Comparison for diffing two commits or refs, an Onboarding Path that generates a judged reading guide for new contributors, and an Interactive Repository Map that visualizes file dependencies as a live force-directed graph.

## Architecture Summary

The application is split into a FastAPI backend and a lightweight static frontend. The backend handles GitHub ingestion, filtering, chunking, embedding, hybrid retrieval, repository summarization, knowledge-graph construction, question answering, internal judge-based revision, version comparison, onboarding guide generation, and dependency map building. The frontend provides a local interface for analyzing a repo, viewing the repository overview, asking questions, inspecting cited snippets, comparing versions, navigating an onboarding guide, and exploring the visual dependency graph.

At a high level, the architecture works like this:

1.  The frontend sends a repository URL to the backend.
2.  The backend resolves the GitHub repo, fetches supported files, and filters out irrelevant content.
3.  The chunking layer converts files into structured retrieval units using language-specific parsers.
4.  The embedding layer creates vectors for those chunks and stores them in Chroma.
5.  The backend derives repository-wide graph relationships and a GraphRAG global context summary.
6.  The retriever uses hybrid search to find the most relevant chunks for a question.
7.  The QA layer drafts an answer using retrieved evidence plus the global graph context as structural guidance.
8.  An internal LLM-as-a-Judge pass reviews the draft and revises it if needed before the final answer is returned.
9.  Commit history is fetched with per-file change tracking and LLM-generated per-commit summaries.
10. On demand, the onboarding service generates a reading path, core concepts, and contributor profiles, then runs a structured LLM-as-a-Judge pass before returning the guide.
11. On demand, the repo map service builds a dependency graph from the stored knowledge graph edges.

Project layout:

``` text
frontend/
  index.html
  styles.css
  app.js
backend/
  __init__.py
  main.py
  config.py
  github_loader.py
  file_filter.py
  chunker.py
  parsers.py
  qa_graph.py
  embedder.py
  vector_store.py
  retriever.py
  knowledge_graph.py
  repo_summarizer.py
  judge_service.py
  qa_service.py
  compare_service.py
  onboarding_service.py
  repo_map_service.py
  models.py
  utils.py
data/
  cache/
scripts/
  evaluate_repo.py
requirements.txt
.env.example
README.md
```

## Project Highlights

### Repository Q&A

The core of the project is repository-specific RAG. During analysis, the backend ingests the repository and builds a retrieval index over code, configuration, and documentation chunks. During question answering, the app retrieves only the most relevant repository evidence and passes that context to the LLM. This keeps the answer grounded in real repo content and avoids the much weaker pattern of prompting the model with whole files or entire repositories.

The retrieval pipeline is hybrid rather than purely semantic. It combines embedding similarity with lightweight boosting for file paths, symbol names, and important repository terms such as `train`, `inference`, `config`, `dataset`, `main`, and `endpoint`. That helps the system find the right code even when the user's phrasing does not exactly match the source text.

The right-hand panel shows the retrieved source snippets used to produce the answer. Each source card includes the file path, line range, chunk type, snippet preview, and retrieval score so the answer is fully inspectable.

### GraphRAG And Global Context

The project builds a repository-wide knowledge graph from the same analyzed files and chunks used by the RAG pipeline. This graph captures structural entities such as repositories, files, symbols, languages, and inferred file-level dependencies. From that graph, the backend produces a global context summary that helps the answering layer reason about relationships across the entire repository instead of only the top retrieved chunks.

The GraphRAG layer is additive. The app builds the graph locally from the same repository files already ingested, then turns that structure into global context for the answer layer, the repository overview UI, the dependency edges shown in the Repo Map tab, and the key dependencies listed on the summary page.

### Chunking and Language Support

Chunking is intentionally code-aware. Python files are parsed with AST-based logic so the system creates chunks at the function, method, and class level. Go, Rust, Java, C++, and Ruby files use regex-based declaration boundary detection with the same logical sectioning approach. JavaScript and TypeScript files use function and export detection. All other supported formats fall back to logical section splitting using blank lines and natural boundaries.

Every chunk carries metadata including file path, language, chunk type, symbol name, start line, and end line so the answer layer can cite evidence precisely and the UI can display relevant source context.

Supported file types:

| Category | Extensions |
|------------------------------------|------------------------------------|
| Python | `.py` |
| JavaScript / TypeScript | `.js` `.ts` `.tsx` |
| Web templates and styles | `.html` `.htm` `.css` `.scss` `.sass` `.svelte` `.vue` |
| Config and data | `.json` `.yaml` `.yml` `.toml` |
| Documentation | `.md` |
| Go | `.go` |
| Rust | `.rs` |
| Java | `.java` |
| C and C++ | `.c` `.h` `.cpp` `.cc` `.cxx` `.hpp` |
| Ruby | `.rb` |

### LLM-as-a-Judge

The project uses LLM-as-a-Judge as an internal answer quality gate in the `/ask` flow. After the QA layer produces a draft answer from retrieved evidence, a second OpenAI call reviews that draft against the same cited sources. If the draft is weak, incomplete, poorly cited, or too confident given the evidence, the judge rewrites it before the final answer is returned to the user.

The same quality-gate idea is also used in the Version Comparison and Onboarding sections. Version Comparison drafts an answer from retrieved diff evidence, then the judge revises that answer against the comparison context. Onboarding drafts structured JSON for the reading path, core concepts, and complexity note, then a structured judge pass reviews and revises that JSON before the UI renders it.

This means the user only sees the revised final answer or guide, not the intermediate draft or the judge process. The goal is to improve groundedness and clarity without adding extra UI complexity.

### Version History and Activity Flow

After analysis, the Version History panel shows the full commit timeline for the repository. Each commit entry shows the short SHA, author name, date, commit message, and a list of file-level change badges colored by status: added, modified, deleted, or renamed. Below each commit, a one-sentence LLM-generated explanation describes what the change actually accomplished rather than restating the commit message.

The Activity Flow sub-tab shows the same commits as a vertical flowchart with a dot-and-line track visualization. Above the flow, a one-to-two sentence AI summary describes what kind of work is actively happening in the repository based on recent commit activity.

The changes banner at the top of the history panel shows only genuinely new or removed files rather than line-level insertion and deletion counts, so the signal is cleaner for understanding actual structural changes.

### Version Comparison

The Version Comparison tab lets the user compare any two commits or refs. Preset options include first commit to newest, the two most recent commits, or a custom pair selected from dropdowns. The comparison result shows the status (ahead, behind, diverged, identical), commit count, files changed, and total diff size. A file-by-file change list shows each changed file with its status badge and line counts.

After comparing, the user can ask a natural-language question about the differences directly in the comparison panel. The backend chunks the diff into per-file, per-commit, and summary chunks, embeds those chunks, and keeps them in a temporary in-memory RAG store for that comparison. Each comparison question embeds the question, retrieves the most relevant diff chunks, drafts an answer from that focused context, and then runs LLM-as-a-Judge before returning the final response.

### Onboarding Path

The Onboarding Path tab generates a personalized onboarding guide for any analyzed repository. It is produced on demand by clicking Generate Guide and consists of four sections.

The Reading Path gives a numbered list of five to eight files ordered from easiest to hardest, with a one-sentence explanation of why each file belongs at that position in the reading sequence. The Core Concepts section lists three to five key abstractions specific to the repository with a two-to-three sentence explanation and the files where each concept is most visible. The Contributors section shows each author who appears in the commit history along with their commit count, inferred focus area, and the files they touched most often. A Complexity note at the bottom gives a single plain-English warning about the hardest part of the codebase for a new contributor to understand.

The onboarding guide is generated by a structured JSON OpenAI call and then reviewed by a structured LLM-as-a-Judge pass. No cache clear is needed; the guide is generated fresh on every request from the current analyzed repository summary and commit history.

### Interactive Repository Map

The Repository Map tab is the trademark feature of this project. It renders a live, zoomable, force-directed dependency graph of the entire repository using D3.js version 7. Every indexed file appears as a circle node. Node size scales with the file's line count. Node color is determined by language using a consistent color palette. Dependency edges show import relationships extracted from the knowledge graph.

Hovering over any node highlights only its direct neighbors and their edges while dimming everything else. Clicking a node opens a right-side panel showing the file path, language, role, estimated line count, LLM summary, key symbols as monospace chips, and clickable lists of the files it imports and the files that import it. Clicking a connection in the side panel pans and zooms the graph to that target file.

The Find File search box instantly pans and zooms to any matching node by name. The Cluster by Directory toggle applies a spatial force that pulls nodes toward cluster centroids based on their top-level directory, grouping frontend, backend, and test files into visible regions. Toggling again returns to the free physics layout. The graph also displays a file and dependency count below the legend so the data density is always visible.

After a fresh analysis, the Repository Map uses all dependency edges computed by the knowledge graph service from actual file content. Clearing the cache and re-analyzing will update both the summary page dependency count and the map edge count to reflect the current state of the repository.

## Setup Instructions

### Platform compatibility

The application code is designed to run on macOS, Linux, and Windows. The main cross-platform risk is Python dependency installation, especially native packages pulled in by `chromadb` such as `grpcio`. If installation fails on Windows, use a current 64-bit Python release, upgrade `pip`, and install dependencies inside a fresh virtual environment.

### 1. Clone and enter the project

If you are starting from GitHub, clone the repository and move into the project folder first.

``` bash
git clone https://github.com/CarlostheCorrea/GitHubResearchAssistant.git
cd GitHubResearchAssistant
```

### 2. Install dependencies

Create and activate a virtual environment first, then install dependencies.

macOS / Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Windows Command Prompt:

``` bash
py -m venv .venv
.venv\Scripts\activate.bat
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 3. Create the environment file

macOS / Linux:

``` bash
cp .env.example .env
```

Windows PowerShell:

``` powershell
Copy-Item .env.example .env
```

Windows Command Prompt:

``` bat
copy .env.example .env
```

Then add your keys to `.env`:

-   `OPENAI_API_KEY` is required
-   `GITHUB_TOKEN` is recommended but not required to use the project

The app can analyze public repositories without a `GITHUB_TOKEN`, but adding one improves GitHub API reliability. The main benefits are higher GitHub API rate limits, fewer ingestion failures when testing multiple repositories, and more reliable fetching of repository metadata, trees, and file contents.

Example:

``` env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GITHUB_TOKEN=
```

### 4. Start the app

macOS / Linux:

``` bash
python3 -m uvicorn backend.main:app --reload --reload-dir backend --reload-dir frontend
```

Windows PowerShell or Command Prompt:

``` powershell
py -m uvicorn backend.main:app --reload --reload-dir backend --reload-dir frontend
```

Then open <http://127.0.0.1:8000>.

If you keep your virtual environment inside the project as `.venv`, avoid running `uvicorn --reload` without `--reload-dir`. Otherwise the file watcher can detect package changes inside `.venv/site-packages` and repeatedly restart the server even when your app code has not changed.

### 5. Optional environment variables

| Variable | Required | Purpose |
|------------------------|------------------------|------------------------|
| `OPENAI_API_KEY` | Yes | OpenAI access for embeddings, answer generation, repo summary generation, commit summarization, version comparison answers, onboarding guide generation, and judge revision |
| `OPENAI_CHAT_MODEL` | No | Chat model used for repo summaries, answers, commit explanations, version comparison answers, onboarding guides, and judge revision |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model used for repo chunks and query embeddings |
| `GITHUB_TOKEN` | No | Recommended for higher GitHub API rate limits and more reliable public repo ingestion |
| `REQUEST_TIMEOUT_SECONDS` | No | HTTP timeout for GitHub requests |
| `CLONE_TIMEOUT_SECONDS` | No | Timeout for temporary Git clone operations |
| `MAX_FILE_BYTES` | No | Per-file size limit |
| `MAX_COMMIT_HISTORY` | No | Maximum number of recent commits stored in the timeline |
| `EMBEDDING_BATCH_SIZE` | No | Number of chunks sent per embedding batch |
| `VECTOR_QUERY_K` | No | Number of vector candidates before reranking |
| `ANSWER_CONTEXT_K` | No | Final chunk count passed to the LLM |

## User Guide

### 1. Analyze a repository

Open the app in your browser, paste a public GitHub repository URL into the repository field, and click Analyze Repo.

The backend will resolve the repository, temporarily clone the selected branch, filter supported files, chunk the contents, generate embeddings, build a local Chroma index, compute the knowledge graph, fetch commit history with per-file change tracking, and generate LLM summaries for each commit. Once analysis completes, the Repository Overview section will show the repo summary, language mix, key files, likely entry points, configuration files, and other surfaced repository areas. The temporary clone is deleted after analysis; the reusable saved data is the manifest and Chroma vector index.

If the same repository is analyzed again and the HEAD commit SHA has not changed, the cached manifest is returned immediately without re-indexing. The status badge in the top-right of the analyze panel indicates whether the result came from cache or a fresh index.

<img width="1016" height="635" alt="Screenshot 2026-04-20 at 2 49 01 PM" src="https://github.com/user-attachments/assets/712d2d17-f6ac-4c59-b962-c3388bbf204f" />

<img width="968" height="702" alt="Screenshot 2026-04-20 at 2 49 19 PM" src="https://github.com/user-attachments/assets/3c8f3d21-1751-4174-b26e-b9011a64947d" />

### 2. Ask repository questions

After analysis, use the question box in the Repository Q&A tab to ask natural-language questions. Good example questions include:

-   How is data loaded?
-   Where is the inference code?
-   What are the main components of this repository?
-   Where is the configuration defined?
-   How does this repo train the model?
-   What files look like the entry points?
-   What changed since the last analysis?

When you submit a question, the system retrieves the most relevant chunks, drafts an answer, internally reviews that answer with the judge pass, and returns the final grounded answer. The right-hand evidence panel shows each source snippet with its file path, line range, chunk type, and retrieval score.

<img width="1512" height="817" alt="Screenshot 2026-04-20 at 2 50 41 PM" src="https://github.com/user-attachments/assets/03ec390a-0ed4-412c-9926-b6d9598a09a0" />



### 3. View commit history and activity flow

After analysis, scroll down to the Version History section. The Timeline sub-tab shows each commit with its SHA badge, author, date, commit message, file-change badges by status, and a one-sentence LLM explanation of what the commit accomplished.

Switch to the Activity Flow sub-tab to see the same commits rendered as a vertical flowchart. An AI-generated summary above the flow describes what kind of work is actively happening in the repository.

<img width="1022" height="329" alt="Screenshot 2026-04-20 at 2 49 33 PM" src="https://github.com/user-attachments/assets/35fca751-d612-4a68-a2c1-e10bf2a27c17" />

<img width="1197" height="788" alt="Screenshot 2026-04-20 at 2 50 04 PM" src="https://github.com/user-attachments/assets/82d8ed14-5bff-4eb5-8ed5-d57ab8e6687b" />


### 4. Compare two versions

Click the Version Comparison tab. After a repository has been analyzed, select a preset or choose custom base and head versions from the dropdowns, then click Compare Versions. The result shows status, commit count, files changed, and a diff size summary. Below that, a file list shows each changed file with its status and line counts.

Use the question box in the comparison panel to ask anything about what changed between the two selected versions. The comparison question flow uses a temporary in-memory RAG store over the diff chunks and then runs LLM-as-a-Judge on the drafted answer.

<img width="1512" height="814" alt="Screenshot 2026-04-20 at 2 51 51 PM" src="https://github.com/user-attachments/assets/536ce74c-1b6f-4c81-91a3-fa5a84cceaa1" />


### 5. Generate an onboarding guide

Click the Onboarding tab and then Generate Guide. The backend makes a structured LLM call, runs a structured LLM-as-a-Judge pass, and returns a reading path of five to eight files in recommended order, three to five core concepts specific to the repository, a contributor breakdown derived from commit history, and a complexity note about the hardest part of the codebase.

No cache clear is needed for the onboarding guide. It is generated fresh on every request.


<img width="1283" height="707" alt="Screenshot 2026-04-20 at 2 53 33 PM" src="https://github.com/user-attachments/assets/93e8b66d-8fa0-46b0-9684-fa92f8401d31" />


### 6. Explore the repository map

Click the Repo Map tab and then Generate Map. The backend loads all dependency edges from the knowledge graph and all file metadata from the vector index, then returns both to the frontend.

The D3 force simulation runs and then auto-zooms to fit all nodes. Use scroll or pinch to zoom, drag the canvas to pan, and drag individual nodes to rearrange them. Hover a node to highlight its connections. Click a node to open the side panel with its details and clickable neighbor lists. Use the Find File box to jump to any file by name. Toggle Cluster by Directory to group nodes spatially by their top-level folder.

For the most accurate dependency edges, clear the cache and re-analyze so the map reads from the freshest knowledge graph data.

<img width="1362" height="638" alt="Screenshot 2026-04-20 at 2 54 20 PM" src="https://github.com/user-attachments/assets/321e9b12-dc56-4850-9855-854ecc79132d" />

### 7. Clear cached data

Use the Clear All Cache button to remove all cached repository manifests and vector indexes before re-running analysis or before committing the project. Use the per-repo cache clear to remove only the currently loaded repository's cached data while keeping other repos intact.

## API Endpoints

| Method | Endpoint | Description |
|------------------------|------------------------|------------------------|
| `GET` | `/health` | Service health check |
| `POST` | `/analyze-repo` | Analyze and index a repository |
| `POST` | `/ask` | Ask a grounded question about an analyzed repo |
| `GET` | `/repo-summary` | Fetch the stored repository summary |
| `GET` | `/history` | Fetch commit history with file changes |
| `GET` | `/versions` | List commits available for version comparison |
| `POST` | `/compare-versions` | Compare two commits or refs |
| `POST` | `/ask-version-compare` | Ask a question about a version diff |
| `GET` | `/branches` | List branches for a repository |
| `POST` | `/compare-branches` | Compare two branches |
| `POST` | `/ask-compare` | Ask a question about a branch diff |
| `GET` | `/onboarding` | Generate an onboarding guide for an analyzed repo |
| `GET` | `/repo-map` | Build a dependency graph for the Repo Map tab |
| `DELETE` | `/cache` | Clear all cached manifests and vector indexes |
| `DELETE` | `/cache/repo` | Clear the cache for a single repository |

Example:

``` bash
curl -X POST http://127.0.0.1:8000/analyze-repo \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/pallets/flask"}'
```

## Example Questions

-   How does this repo train the model?
-   Where is the inference code?
-   How is data loaded?
-   Where is the configuration defined?
-   What are the main components of this repository?
-   What files look like the entry points?
-   Which modules are most relevant to API serving?
-   What changed between the first and most recent commit?
-   Which files does main.py depend on?

## Error Handling

The backend includes explicit handling for:

-   invalid GitHub URLs
-   missing repositories or invalid branches
-   GitHub API rate limits
-   empty or unsupported repos
-   missing or placeholder OpenAI API key
-   OpenAI authentication and rate limit errors
-   unexpected runtime failures returned as structured HTTP error responses

## Future Improvements

-   Add background indexing for larger repositories so the UI does not block during ingestion.
-   Add stronger retrieval evaluation and benchmark coverage.
-   Improve refresh and invalidation behavior for cached repo analyses.
-   Expand config parsing for YAML, TOML, and config-heavy repos.
-   Add support for private repositories using OAuth tokens.
-   Persist the full dependency graph separately from the manifest so the Repo Map can be loaded without re-analysis.
-   Add export of the onboarding guide and repository summary as a shareable HTML report.
