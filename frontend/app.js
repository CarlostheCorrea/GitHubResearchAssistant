const state = {
  repoUrl: "",
  repoSummary: null,
};

const elements = {
  analyzeForm: document.getElementById("analyze-form"),
  askForm: document.getElementById("ask-form"),
  repoUrl: document.getElementById("repo-url"),
  questionInput: document.getElementById("question-input"),
  analyzeButton: document.getElementById("analyze-button"),
  clearCacheButton: document.getElementById("clear-cache-button"),
  cacheIndicator: document.getElementById("cache-indicator"),
  askButton: document.getElementById("ask-button"),
  statusBadge: document.getElementById("status-badge"),
  statusMessage: document.getElementById("status-message"),
  repoSummaryEmpty: document.getElementById("repo-summary-empty"),
  repoSummary: document.getElementById("repo-summary"),
  summaryText: document.getElementById("summary-text"),
  summaryName: document.getElementById("summary-name"),
  summaryBranch: document.getElementById("summary-branch"),
  summaryFootprint: document.getElementById("summary-footprint"),
  summaryLanguages: document.getElementById("summary-languages"),
  keyFiles: document.getElementById("key-files"),
  entryFiles: document.getElementById("entry-files"),
  trainingFiles: document.getElementById("training-files"),
  inferenceFiles: document.getElementById("inference-files"),
  configFiles: document.getElementById("config-files"),
  dataFiles: document.getElementById("data-files"),
  answerEmpty: document.getElementById("answer-empty"),
  answerContent: document.getElementById("answer-content"),
  answerText: document.getElementById("answer-text"),
  sourcesEmpty: document.getElementById("sources-empty"),
  sourcesList: document.getElementById("sources-list"),
  sampleQuestions: Array.from(document.querySelectorAll(".sample-chip")),
};

elements.analyzeForm.addEventListener("submit", handleAnalyze);
elements.askForm.addEventListener("submit", handleAsk);
elements.clearCacheButton.addEventListener("click", handleClearCache);
elements.sampleQuestions.forEach((button) => {
  button.addEventListener("click", () => {
    elements.questionInput.value = button.textContent.trim();
    elements.questionInput.focus();
  });
});

async function handleAnalyze(event) {
  event.preventDefault();
  const repoUrl = elements.repoUrl.value.trim();
  if (!repoUrl) {
    setStatus("error", "Enter a public GitHub repository URL first.");
    return;
  }

  setLoading(true, "Analyzing repository, fetching files, chunking code, and building the vector index...");
  clearAnswer();

  try {
    const response = await fetch("/analyze-repo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_url: repoUrl }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Repository analysis failed.");
    }

    state.repoUrl = repoUrl;
    state.repoSummary = payload.repo_summary;
    renderRepoSummary(payload.repo_summary, payload);
    setCacheIndicator("", "");
    elements.questionInput.disabled = false;
    elements.askButton.disabled = false;

    const cacheText = payload.cached ? "Loaded cached index." : "Fresh index created.";
    setStatus("success", `${payload.message} ${cacheText}`);
  } catch (error) {
    setStatus("error", error.message || "Repository analysis failed.");
  } finally {
    setLoading(false);
  }
}

async function handleAsk(event) {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  if (!state.repoUrl) {
    setStatus("error", "Analyze a repository before asking a question.");
    return;
  }
  if (!question) {
    setStatus("error", "Enter a question about the repository.");
    return;
  }

  setLoading(true, "Retrieving relevant chunks and generating a grounded answer...");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_url: state.repoUrl,
        question,
      }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Question answering failed.");
    }

    renderAnswer(payload.answer);
    renderSources(payload.sources || []);
    if (payload.repo_summary) {
      state.repoSummary = payload.repo_summary;
      renderRepoSummary(payload.repo_summary);
    }
    setStatus("success", "Answer generated from retrieved repository evidence.");
  } catch (error) {
    setStatus("error", error.message || "Question answering failed.");
  } finally {
    setLoading(false);
  }
}

async function handleClearCache() {
  setLoading(true, "Deleting all cached vector data and repo summaries...");

  try {
    const response = await fetch("/cache", {
      method: "DELETE",
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Failed to clear all cached repository data.");
    }

    clearRepoSummary();
    clearAnswer();
    state.repoUrl = "";
    state.repoSummary = null;
    elements.repoUrl.value = "";
    elements.questionInput.value = "";
    elements.questionInput.disabled = true;
    elements.askButton.disabled = true;
    setCacheIndicator(
      payload.status === "cleared" ? "success" : "info",
      payload.status === "cleared" ? "All cache cleared" : "No cache found"
    );
    setStatus("success", payload.message);
  } catch (error) {
    setCacheIndicator("error", "Clear failed");
    setStatus("error", error.message || "Failed to clear all cached repository data.");
  } finally {
    setLoading(false);
  }
}

function renderRepoSummary(summary, analyzePayload = null) {
  elements.repoSummaryEmpty.classList.add("hidden");
  elements.repoSummary.classList.remove("hidden");

  elements.summaryName.textContent = summary.repo_name;
  elements.summaryBranch.textContent = `Branch: ${summary.branch}`;

  const fileCount = analyzePayload?.files_indexed ?? summary.files_indexed ?? 0;
  const chunkCount = analyzePayload?.chunks_created ?? summary.chunks_indexed ?? 0;
  renderSummaryParagraphs(summary, fileCount, chunkCount);
  elements.summaryFootprint.textContent = `${fileCount} files • ${chunkCount} chunks`;
  elements.summaryLanguages.textContent = `Languages: ${formatList(summary.detected_languages)}`;

  renderPathGroup(elements.keyFiles, summary.key_files, 4);
  renderPathGroup(elements.entryFiles, summary.probable_entry_points, 2);
  renderPathGroup(elements.trainingFiles, summary.probable_training_files, 3);
  renderPathGroup(elements.inferenceFiles, summary.probable_inference_files, 3);
  renderPathGroup(elements.configFiles, summary.probable_config_files, 3);
  renderPathGroup(elements.dataFiles, summary.probable_data_files, 3);
}

function renderSummaryParagraphs(summary, fileCount, chunkCount) {
  elements.summaryText.replaceChildren();

  const paragraphs = splitSummaryParagraphs(summary.high_level_summary);
  if (!paragraphs.length) {
    paragraphs.push(buildFallbackSummary(summary, fileCount, chunkCount));
  }

  paragraphs.forEach((paragraph) => {
    const node = document.createElement("p");
    node.textContent = paragraph;
    elements.summaryText.appendChild(node);
  });
}

function splitSummaryParagraphs(summaryText) {
  if (!summaryText || !summaryText.trim()) {
    return [];
  }

  return summaryText
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

function buildFallbackSummary(summary, fileCount, chunkCount) {
  const topLanguage = summary.detected_languages?.[0] || "mixed-language";
  const sentences = [
    `${summary.repo_name} appears to be a ${topLanguage} repository.`,
    `This analysis indexed ${fileCount} supported files and ${chunkCount} retrieval chunks.`,
  ];

  if (summary.readme_excerpt) {
    sentences.push(`README signal: ${summary.readme_excerpt}`);
  }

  const architectureBits = [];
  if (summary.probable_entry_points?.length) {
    architectureBits.push("likely entry points were detected");
  }
  if (summary.probable_training_files?.length) {
    architectureBits.push("training-related code was identified");
  }
  if (summary.probable_inference_files?.length) {
    architectureBits.push("inference or serving logic was surfaced");
  }
  if (summary.probable_config_files?.length) {
    architectureBits.push("configuration files were found");
  }
  if (summary.probable_data_files?.length) {
    architectureBits.push("data-loading logic was detected");
  }
  if (architectureBits.length) {
    sentences.push(`${capitalizeFirst(joinNaturalLanguage(architectureBits))}.`);
  }

  return sentences.join(" ");
}

function renderAnswer(answer) {
  elements.answerEmpty.classList.add("hidden");
  elements.answerContent.classList.remove("hidden");
  elements.answerText.textContent = answer;
}

function clearAnswer() {
  elements.answerEmpty.classList.remove("hidden");
  elements.answerContent.classList.add("hidden");
  elements.answerText.textContent = "";
  elements.sourcesEmpty.classList.remove("hidden");
  elements.sourcesList.classList.add("hidden");
  elements.sourcesList.replaceChildren();
}

function clearRepoSummary() {
  elements.repoSummaryEmpty.classList.remove("hidden");
  elements.repoSummary.classList.add("hidden");
  elements.summaryText.replaceChildren();
  elements.summaryName.textContent = "";
  elements.summaryBranch.textContent = "";
  elements.summaryFootprint.textContent = "";
  elements.summaryLanguages.textContent = "";
  renderPathGroup(elements.keyFiles, []);
  renderPathGroup(elements.entryFiles, []);
  renderPathGroup(elements.trainingFiles, []);
  renderPathGroup(elements.inferenceFiles, []);
  renderPathGroup(elements.configFiles, []);
  renderPathGroup(elements.dataFiles, []);
}

function renderSources(sources) {
  elements.sourcesList.replaceChildren();

  if (!sources.length) {
    elements.sourcesEmpty.classList.remove("hidden");
    elements.sourcesList.classList.add("hidden");
    return;
  }

  elements.sourcesEmpty.classList.add("hidden");
  elements.sourcesList.classList.remove("hidden");

  sources.forEach((source) => {
    const card = document.createElement("article");
    card.className = "source-card";

    const header = document.createElement("div");
    header.className = "source-header";

    const title = document.createElement("h3");
    title.className = "source-title";
    title.textContent = source.file_path;

    const meta = document.createElement("div");
    meta.className = "source-meta";

    meta.appendChild(buildPill(source.chunk_type));
    meta.appendChild(buildPill(formatLineRange(source.start_line, source.end_line)));
    meta.appendChild(buildPill(`score ${Number(source.score).toFixed(2)}`));

    header.append(title, meta);

    card.appendChild(header);

    if (source.short_summary) {
      const summary = document.createElement("p");
      summary.className = "source-summary";
      summary.textContent = source.short_summary;
      card.appendChild(summary);
    }

    const snippet = document.createElement("pre");
    snippet.className = "source-snippet";
    snippet.textContent = source.snippet;
    card.appendChild(snippet);

    elements.sourcesList.appendChild(card);
  });
}

function renderPathGroup(container, items, visibleCount = 3) {
  container.replaceChildren();
  if (!items || !items.length) {
    const empty = document.createElement("div");
    empty.className = "path-empty";
    empty.textContent = "None detected";
    container.appendChild(empty);
    return;
  }

  const list = document.createElement("div");
  list.className = "path-list";
  const overflowRows = [];

  items.forEach((item, index) => {
    const row = buildPathRow(item);
    if (index >= visibleCount) {
      row.classList.add("hidden");
      overflowRows.push(row);
    }
    list.appendChild(row);
  });
  container.appendChild(list);

  if (overflowRows.length) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "path-toggle";
    toggle.textContent = `+${overflowRows.length} more`;
    toggle.addEventListener("click", () => {
      const expanded = toggle.dataset.expanded === "true";
      overflowRows.forEach((row) => row.classList.toggle("hidden", expanded));
      toggle.dataset.expanded = String(!expanded);
      toggle.textContent = expanded ? `+${overflowRows.length} more` : "Show less";
    });
    container.appendChild(toggle);
  }
}

function buildPill(text) {
  const pill = document.createElement("span");
  pill.className = "source-pill";
  pill.textContent = text;
  return pill;
}

function setLoading(isLoading, message = "") {
  elements.analyzeButton.disabled = isLoading;
  elements.clearCacheButton.disabled = isLoading;
  elements.askButton.disabled = isLoading || !state.repoUrl;
  elements.questionInput.disabled = isLoading || !state.repoUrl;

  if (isLoading) {
    setStatus("loading", message || "Working...");
  }
}

function setCacheIndicator(kind, text) {
  if (!kind || !text) {
    elements.cacheIndicator.className = "cache-indicator hidden";
    elements.cacheIndicator.textContent = "";
    return;
  }
  elements.cacheIndicator.className = `cache-indicator ${kind}`;
  elements.cacheIndicator.textContent = text;
}

function setStatus(kind, message) {
  elements.statusBadge.className = `status-badge ${kind}`;
  elements.statusBadge.textContent = statusLabel(kind);
  elements.statusMessage.textContent = message;
}

function statusLabel(kind) {
  if (kind === "loading") return "Working";
  if (kind === "success") return "Ready";
  if (kind === "error") return "Error";
  return "Idle";
}

function formatLineRange(start, end) {
  if (!start && !end) {
    return "lines unavailable";
  }
  if (start === end) {
    return `line ${start}`;
  }
  return `lines ${start}-${end}`;
}

function formatList(items) {
  if (!items || !items.length) {
    return "n/a";
  }
  return items.join(", ");
}

function buildPathRow(filePath) {
  const row = document.createElement("div");
  row.className = "path-row";
  row.title = filePath;

  const header = document.createElement("div");
  header.className = "path-header";

  const primary = document.createElement("div");
  primary.className = "path-primary";
  primary.textContent = getFileName(filePath);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "path-expand";
  toggle.textContent = "Full path";

  header.append(primary, toggle);

  const secondary = document.createElement("div");
  secondary.className = "path-secondary";
  secondary.textContent = formatParentPath(filePath);

  const full = document.createElement("div");
  full.className = "path-full hidden";
  full.textContent = filePath;

  toggle.addEventListener("click", () => {
    const expanded = toggle.dataset.expanded === "true";
    full.classList.toggle("hidden", expanded);
    toggle.dataset.expanded = String(!expanded);
    toggle.textContent = expanded ? "Full path" : "Hide path";
  });

  row.append(header, secondary, full);
  return row;
}

function getFileName(filePath) {
  const parts = filePath.split("/");
  return parts[parts.length - 1] || filePath;
}

function formatParentPath(filePath) {
  const parts = filePath.split("/");
  if (parts.length <= 1) {
    return "Repository root";
  }
  return shortenMiddle(parts.slice(0, -1).join("/"), 28);
}

function shortenMiddle(value, maxLength = 32) {
  if (value.length <= maxLength) {
    return value;
  }
  const side = Math.max(8, Math.floor((maxLength - 1) / 2));
  return `${value.slice(0, side)}…${value.slice(-side)}`;
}

function joinNaturalLanguage(items) {
  if (items.length === 1) {
    return items[0];
  }
  if (items.length === 2) {
    return `${items[0]} and ${items[1]}`;
  }
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function capitalizeFirst(value) {
  if (!value) {
    return value;
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}
