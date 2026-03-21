# AI GitHub Research Assistant

AI GitHub Research Assistant is a production-style MVP for understanding public GitHub repositories with code-aware retrieval-augmented generation. A user pastes a repository URL, the system ingests the repo, builds a local vector index over structured code chunks, and answers repository questions with grounded evidence, citations, and source snippets.

## Program Summary

This project is designed to help a user quickly understand an unfamiliar codebase. Instead of sending entire files or the whole repository into an LLM, it first fetches the repository, filters relevant files, chunks the code intelligently, embeds those chunks, and stores them in a local Chroma vector index. When the user asks a question, the app retrieves the most relevant chunks, generates an evidence-based answer, and shows the supporting snippets in the UI.

The goal is not just to answer questions, but to answer them in a way that is inspectable. The app highlights where the answer came from, which files were used, and which parts of the repository appear to be the key architectural areas such as entry points, configuration, data loading, and inference or training logic.

## Architecture Summary

The application is split into a FastAPI backend and a lightweight static frontend. The backend handles GitHub ingestion, filtering, chunking, embedding, vector search, retrieval, repository summarization, question answering, and internal judge-based revision. The frontend provides a local interface for analyzing a repo, viewing the repository overview, asking questions, and inspecting cited snippets.

At a high level, the architecture works like this:

1. The frontend sends a repository URL to the backend.
2. The backend resolves the GitHub repo, fetches supported files, and filters out irrelevant content.
3. The chunking layer converts files into structured retrieval units.
4. The embedding layer creates vectors for those chunks and stores them in Chroma.
5. The retriever finds the most relevant chunks for a question.
6. The QA layer drafts an answer from retrieved evidence only.
7. An internal LLM-as-a-Judge pass reviews the draft and revises it if needed before the final answer is returned.

Project layout:

```text
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
  repo_summarizer.py
  judge_service.py
  qa_service.py
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

### RAG

The core of the project is repository-specific RAG. During analysis, the backend ingests the repository and builds a retrieval index over code, configuration, and documentation chunks. During question answering, the app retrieves only the most relevant repository evidence and passes that context to the LLM. This keeps the answer grounded in real repo content and avoids the much weaker pattern of prompting the model with whole files or entire repositories.

The retrieval pipeline is hybrid rather than purely semantic. It combines embedding similarity with lightweight boosting for file paths, symbol names, and important repository terms such as `train`, `inference`, `config`, `dataset`, `main`, and `endpoint`. That helps the system find the right code even when the user’s phrasing does not exactly match the source text.

### Chunking

Chunking is intentionally code-aware. Python files are parsed with AST-based logic so the system can create chunks at the function, method, and class level, plus file-summary chunks where useful. This makes retrieval much more precise than naive character splitting because the retriever can target meaningful program units instead of arbitrary text windows.

For non-Python files, the system falls back to logical section chunking. Markdown is split by headings, configuration formats are grouped into meaningful sections when possible, and general text or code files are split around natural boundaries rather than using a fixed-size strategy alone. Every chunk carries metadata such as file path, language, chunk type, symbol name, and line range so the answer layer can cite evidence precisely.

### Vectorization

After chunking, each chunk is converted into an embedding using the configured OpenAI embedding model. Those embeddings are stored locally in Chroma, which acts as the project’s vector database. The storage is local and reusable, so repeated repository analysis during development is fast and inexpensive outside of the embedding calls themselves.

This vectorization layer is what allows the app to perform semantic retrieval over repository content. Instead of matching only exact words, the system can retrieve code and documentation that are conceptually related to the user’s question. Because vectors are stored with chunk metadata, retrieval can return both the relevant content and the structured context needed for citations and snippet display.

### LLM-as-a-Judge

The project uses LLM-as-a-Judge as an internal answer quality gate in the `/ask` flow. After the QA layer produces a draft answer from retrieved evidence, a second OpenAI call reviews that draft against the same cited sources. If the draft is weak, incomplete, poorly cited, or too confident given the evidence, the judge rewrites it before the final answer is returned to the user.

This means the user only sees the revised final answer, not the intermediate draft or the judge process. The goal is to improve groundedness and clarity without adding extra UI complexity. In practice, this gives the app a second pass that can tighten citations, reduce unsupported claims, and make insufficiency handling more explicit when the retrieval context is thin.

## Setup Instructions

### 1. Clone and enter the project

If you are starting from GitHub, clone the repository and move into the project folder first.

```bash
git clone https://github.com/CarlostheCorrea/GitHubResearchAssistant.git
cd GitHubResearchAssistant
```

Replace the URL and folder name above with the actual repository you pushed to GitHub.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the environment file

```bash
cp .env.example .env
```

Then add your keys to `.env`:

- `OPENAI_API_KEY` is required
- `GITHUB_TOKEN` is recommended but not required to use the project

The app can analyze public repositories without a `GITHUB_TOKEN`, but adding one improves GitHub API reliability. The main benefits are:

- higher GitHub API rate limits
- fewer ingestion failures when testing multiple repositories
- more reliable fetching of repository metadata, trees, and file contents

Example:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GITHUB_TOKEN=
```

### 4. Start the app

```bash
uvicorn backend.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 5. Optional environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | OpenAI access for embeddings, answer generation, repo summary generation, and judge revision |
| `OPENAI_CHAT_MODEL` | No | Chat model used for repo summaries, answers, and judge revision |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model used for repo chunks and query embeddings |
| `GITHUB_TOKEN` | No | Recommended for higher GitHub API rate limits and more reliable public repo ingestion, but not required |
| `REQUEST_TIMEOUT_SECONDS` | No | HTTP timeout for GitHub requests |
| `MAX_FILE_BYTES` | No | Per-file size limit |
| `MAX_TOTAL_REPO_BYTES` | No | Total repository ingestion budget |
| `MAX_FILES_PER_REPO` | No | Maximum files to ingest |
| `VECTOR_QUERY_K` | No | Number of vector candidates before reranking |
| `ANSWER_CONTEXT_K` | No | Final chunk count passed to the LLM |

## User Guide

### 1. Analyze a repository

Open the app in your browser, paste a public GitHub repository URL into the repository field, and click `Analyze Repo`.

The backend will fetch the repository, filter supported files, chunk the contents, generate embeddings, and build a local Chroma index. Once analysis completes, the Repository Overview section will show the repo summary, language mix, key files, likely entry points, configuration files, and other surfaced repository areas.

<img width="1505" height="705" alt="Screenshot 2026-03-21 at 2 57 20 PM" src="https://github.com/user-attachments/assets/59a0b105-f4a6-4412-961c-6d62ee67cdbe" />

<img width="524" height="598" alt="Screenshot 2026-03-21 at 2 58 03 PM" src="https://github.com/user-attachments/assets/9f084f42-ea43-4e38-887b-64fdd9156e8c" />

### 2. Ask repository questions

After analysis completes, use the question box in Step 3 to ask natural-language questions about the repository. Good example questions include:

- How is data loaded?
- Where is the inference code?
- What are the main components of this repository?
- Where is the configuration defined?
- How does this repo train the model?
- What files look like the entry points?

When you submit a question, the system retrieves the most relevant chunks, drafts an answer, internally reviews that answer with the judge pass, and then returns the final grounded answer.

<img width="502" height="144" alt="Screenshot 2026-03-21 at 3 00 41 PM" src="https://github.com/user-attachments/assets/5fc65a4f-dc35-4051-99f7-cbcac7396130" />

### Inspect sources

The right-hand evidence panel shows the retrieved source snippets used to answer the question. Each source card includes:

- file path
- line range when available
- chunk type
- snippet preview
- retrieval score

Use this panel to verify where the answer came from and to inspect the underlying code directly.

<img width="510" height="504" alt="Screenshot 2026-03-21 at 3 00 31 PM" src="https://github.com/user-attachments/assets/2f541011-457d-4714-b708-9e48cced640a" />

### Clear cached data

Use the `Clear All Cache` button if you want to remove cached repository manifests and vector indexes before re-running analysis or before committing the project. This clears stored local repo analysis artifacts from the cache directory.

## API Endpoints

- `GET /health`
- `POST /analyze-repo`
- `POST /ask`
- `GET /repo-summary`
- `DELETE /cache`

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze-repo \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/pallets/flask"}'
```

## Example Questions

- How does this repo train the model?
- Where is the inference code?
- How is data loaded?
- Where is the configuration defined?
- What are the main components of this repository?
- What files look like the entry points?
- Which modules are most relevant to API serving?

## Error Handling

The backend includes explicit handling for:

- invalid GitHub URLs
- missing repositories or invalid branches
- GitHub API rate limits
- empty or unsupported repos
- missing OpenAI API key
- unexpected runtime failures

## Future Improvements

- Add deeper parser support for more languages beyond Python and JS/TS.
- Add background indexing for larger repositories.
- Add stronger retrieval evaluation and benchmark coverage.
- Improve refresh and invalidation behavior for cached repo analyses.
- Expand config parsing for YAML, TOML, and config-heavy repos.
