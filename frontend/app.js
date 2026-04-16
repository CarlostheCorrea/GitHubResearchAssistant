const state = {
  repoUrl: "",
  repoSummary: null,
  hasConversation: false,
  activeTab: "qa",
  versions: [],
  firstCommit: null,
  versionsLoadedFor: "",
  comparison: null,
  compareId: "",
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
  summaryGlobalContext: document.getElementById("summary-global-context"),
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
  chatBody: document.getElementById("chat-body"),
  chatIdle: document.getElementById("chat-idle"),
  chatMessages: document.getElementById("chat-messages"),
  chatThinking: document.getElementById("chat-thinking"),
  suggestionsArea: document.getElementById("suggestions-area"),
  newThreadButton: document.getElementById("new-thread-button"),
  sourcesEmpty: document.getElementById("sources-empty"),
  sourcesList: document.getElementById("sources-list"),
  sourcesCount: document.getElementById("sources-count"),
  sampleQuestions: Array.from(document.querySelectorAll(".sample-chip")),
  queryHint: document.getElementById("query-hint"),
  // History
  historyPanel: document.getElementById("history-panel"),
  historyHeadSha: document.getElementById("history-head-sha"),
  changesBanner: document.getElementById("changes-banner"),
  commitListEmpty: document.getElementById("commit-list-empty"),
  commitList: document.getElementById("commit-list"),
  // History sub-tabs
  historyTimelinePane: document.getElementById("history-timeline-pane"),
  historyFlowPane: document.getElementById("history-flow-pane"),
  activitySummaryBlock: document.getElementById("activity-summary-block"),
  activitySummaryText: document.getElementById("activity-summary-text"),
  activityFlowEmpty: document.getElementById("activity-flow-empty"),
  activityFlow: document.getElementById("activity-flow"),
  // Version comparison
  qaTabButton: document.getElementById("qa-tab-button"),
  compareTabButton: document.getElementById("compare-tab-button"),
  onboardingTabButton: document.getElementById("onboarding-tab-button"),
  workspacePanels: Array.from(document.querySelectorAll(".workspace-tab-content")),
  refreshBranchesButton: document.getElementById("refresh-branches-button"),
  compareEmpty: document.getElementById("compare-empty"),
  compareWorkspace: document.getElementById("compare-workspace"),
  compareForm: document.getElementById("compare-form"),
  versionPresetSelect: document.getElementById("version-preset-select"),
  baseBranchSelect: document.getElementById("base-branch-select"),
  headBranchSelect: document.getElementById("head-branch-select"),
  compareButton: document.getElementById("compare-button"),
  branchStatus: document.getElementById("branch-status"),
  compareResult: document.getElementById("compare-result"),
  compareStatus: document.getElementById("compare-status"),
  compareCommits: document.getElementById("compare-commits"),
  compareFiles: document.getElementById("compare-files"),
  compareDiffSize: document.getElementById("compare-diff-size"),
  compareFileList: document.getElementById("compare-file-list"),
  askCompareForm: document.getElementById("ask-compare-form"),
  compareQuestionInput: document.getElementById("compare-question-input"),
  askCompareButton: document.getElementById("ask-compare-button"),
  compareAnswer: document.getElementById("compare-answer"),
  // Onboarding
  generateOnboardingButton: document.getElementById("generate-onboarding-button"),
  onboardingEmpty: document.getElementById("onboarding-empty"),
  onboardingLoading: document.getElementById("onboarding-loading"),
  onboardingContent: document.getElementById("onboarding-content"),
  onboardingReadingOrder: document.getElementById("onboarding-reading-order"),
  onboardingConcepts: document.getElementById("onboarding-concepts"),
  onboardingContributors: document.getElementById("onboarding-contributors"),
  onboardingComplexity: document.getElementById("onboarding-complexity"),
  onboardingComplexityText: document.getElementById("onboarding-complexity-text"),
  mapTabButton: document.getElementById("map-tab-button"),
  generateMapButton: document.getElementById("generate-map-button"),
  mapSearch: document.getElementById("map-search"),
  mapClusterToggle: document.getElementById("map-cluster-toggle"),
  mapEmpty: document.getElementById("map-empty"),
  mapLoading: document.getElementById("map-loading"),
  mapWorkspace: document.getElementById("map-workspace"),
  mapSvg: document.getElementById("map-svg"),
  mapSidePanel: document.getElementById("map-side-panel"),
  mapSideClose: document.getElementById("map-side-close"),
  mapSidePath: document.getElementById("map-side-path"),
  mapSideLang: document.getElementById("map-side-lang"),
  mapSideRole: document.getElementById("map-side-role"),
  mapSideLines: document.getElementById("map-side-lines"),
  mapSideSummary: document.getElementById("map-side-summary"),
  mapSideSymbols: document.getElementById("map-side-symbols"),
  mapSideImportsLabel: document.getElementById("map-side-imports-label"),
  mapSideImports: document.getElementById("map-side-imports"),
  mapSideImportedByLabel: document.getElementById("map-side-importedby-label"),
  mapSideImportedBy: document.getElementById("map-side-importedby"),
  mapLegend: document.getElementById("map-legend"),
};

elements.analyzeForm.addEventListener("submit", handleAnalyze);
elements.askForm.addEventListener("submit", handleAsk);
elements.clearCacheButton.addEventListener("click", handleClearCache);
elements.newThreadButton.addEventListener("click", handleNewThread);

// History sub-tab switching
document.querySelectorAll(".history-subtab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".history-subtab").forEach((b) => b.classList.remove("history-subtab--active"));
    btn.classList.add("history-subtab--active");
    const subtab = btn.dataset.subtab;
    elements.historyTimelinePane.classList.toggle("hidden", subtab !== "timeline");
    elements.historyFlowPane.classList.toggle("hidden", subtab !== "flow");
  });
});
elements.qaTabButton.addEventListener("click", () => switchWorkspaceTab("qa"));
elements.compareTabButton.addEventListener("click", () => switchWorkspaceTab("compare"));
elements.onboardingTabButton.addEventListener("click", () => switchWorkspaceTab("onboarding"));
elements.mapTabButton.addEventListener("click", () => switchWorkspaceTab("map"));
elements.generateMapButton.addEventListener("click", handleGenerateMap);
elements.mapSideClose.addEventListener("click", () => elements.mapSidePanel.classList.add("hidden"));
elements.generateOnboardingButton.addEventListener("click", handleGenerateOnboarding);
elements.refreshBranchesButton.addEventListener("click", () => loadVersions(true));
elements.versionPresetSelect.addEventListener("change", applyVersionPreset);
elements.baseBranchSelect.addEventListener("change", () => {
  elements.versionPresetSelect.value = "custom";
});
elements.headBranchSelect.addEventListener("change", () => {
  elements.versionPresetSelect.value = "custom";
});
elements.compareForm.addEventListener("submit", handleCompareVersions);
elements.askCompareForm.addEventListener("submit", handleAskCompare);
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
    state.versions = [];
    state.firstCommit = payload.first_commit || null;
    state.versionsLoadedFor = "";
    state.comparison = null;
    state.compareId = "";
    renderRepoSummary(payload.repo_summary, payload);
    renderHistory(
      payload.head_commit_sha || null,
      payload.commit_history || [],
      payload.changes_since_previous || null,
      payload.activity_summary || "",
    );
    resetCompareResult();
    enableCompareWorkspace(true);
    setCacheIndicator("", "");
    elements.questionInput.disabled = false;
    elements.askButton.disabled = false;
    elements.generateOnboardingButton.disabled = false;
    elements.generateMapButton.disabled = false;
    elements.queryHint.textContent = `Grounded answers with cited sources from ${payload.repo_summary.repo_name}.`;
    if (!state.hasConversation) {
      elements.suggestionsArea.classList.remove("hidden");
    }
    document.querySelectorAll(".suggestion-card").forEach((card) => {
      card.disabled = false;
    });

    const cacheText = payload.cached ? "Loaded cached index." : "Fresh index created.";
    setStatus("success", `${payload.message} ${cacheText}`);
    if (state.activeTab === "compare") {
      await loadVersions(false);
    }
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

  elements.questionInput.value = "";
  setLoading(true, "Retrieving relevant chunks and generating a grounded answer...");
  showThinking(true);

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

    showThinking(false);
    appendChatExchange(question, payload.answer);
    renderSources(payload.sources || []);
    if (payload.repo_summary) {
      state.repoSummary = payload.repo_summary;
      renderRepoSummary(payload.repo_summary);
    }
    setStatus("success", "Answer generated from retrieved repository evidence.");
  } catch (error) {
    showThinking(false);
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
    state.versions = [];
    state.firstCommit = null;
    state.versionsLoadedFor = "";
    state.comparison = null;
    state.compareId = "";
    elements.repoUrl.value = "";
    elements.questionInput.value = "";
    elements.questionInput.disabled = true;
    elements.askButton.disabled = true;
    elements.queryHint.textContent = "Analyze a repository to start asking questions.";
    document.querySelectorAll(".suggestion-card").forEach((card) => {
      card.disabled = true;
    });
    setCacheIndicator(
      payload.status === "cleared" ? "success" : "info",
      payload.status === "cleared" ? "All cache cleared" : "No cache found"
    );
    enableCompareWorkspace(false);
    resetCompareResult();
    setStatus("success", payload.message);
  } catch (error) {
    setCacheIndicator("error", "Clear failed");
    setStatus("error", error.message || "Failed to clear all cached repository data.");
  } finally {
    setLoading(false);
  }
}

function switchWorkspaceTab(tabName) {
  state.activeTab = tabName;
  elements.qaTabButton.classList.toggle("active", tabName === "qa");
  elements.compareTabButton.classList.toggle("active", tabName === "compare");
  elements.onboardingTabButton.classList.toggle("active", tabName === "onboarding");
  elements.mapTabButton.classList.toggle("active", tabName === "map");
  elements.qaTabButton.setAttribute("aria-selected", String(tabName === "qa"));
  elements.compareTabButton.setAttribute("aria-selected", String(tabName === "compare"));
  elements.onboardingTabButton.setAttribute("aria-selected", String(tabName === "onboarding"));
  elements.mapTabButton.setAttribute("aria-selected", String(tabName === "map"));

  elements.workspacePanels.forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.workspaceTab !== tabName);
  });

  if (tabName === "compare" && state.repoUrl && state.versionsLoadedFor !== state.repoUrl) {
    loadVersions(false);
  }
}

function enableCompareWorkspace(enabled) {
  elements.compareEmpty.classList.toggle("hidden", enabled);
  elements.compareWorkspace.classList.toggle("hidden", !enabled);
  elements.refreshBranchesButton.disabled = !enabled;
  elements.versionPresetSelect.disabled = !enabled || !state.versions.length;
  elements.baseBranchSelect.disabled = !enabled || !state.versions.length;
  elements.headBranchSelect.disabled = !enabled || !state.versions.length;
  elements.compareButton.disabled = !enabled || state.versions.length < 1;
}

async function loadVersions(force) {
  if (!state.repoUrl) {
    setBranchStatus("info", "Analyze a repository first.");
    enableCompareWorkspace(false);
    return;
  }
  if (!force && state.versionsLoadedFor === state.repoUrl && state.versions.length) {
    enableCompareWorkspace(true);
    return;
  }

  setBranchLoading(true, "Loading versions...");
  try {
    const response = await fetch(`/versions?repo_url=${encodeURIComponent(state.repoUrl)}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load versions.");
    }
    state.versions = payload.commit_history || [];
    state.firstCommit = payload.first_commit || state.firstCommit;
    state.versionsLoadedFor = state.repoUrl;
    renderVersionOptions(state.versions, state.firstCommit);
    resetCompareResult();
    enableCompareWorkspace(true);
    setBranchStatus(
      state.versions.length ? "success" : "info",
      state.versions.length ? `${state.versions.length} versions loaded` : "No versions found"
    );
  } catch (error) {
    setBranchStatus("error", error.message || "Failed to load versions.");
  } finally {
    setBranchLoading(false);
  }
}

function renderVersionOptions(versions, firstCommit) {
  elements.baseBranchSelect.replaceChildren();
  elements.headBranchSelect.replaceChildren();

  const optionRecords = buildVersionOptionRecords(versions, firstCommit);
  optionRecords.forEach((record) => {
    elements.baseBranchSelect.appendChild(new Option(record.label, record.sha));
    elements.headBranchSelect.appendChild(new Option(record.label, record.sha));
  });

  const firstPreset = elements.versionPresetSelect.querySelector('option[value="first-to-newest"]');
  if (firstPreset) {
    firstPreset.disabled = !firstCommit;
  }
  elements.versionPresetSelect.value = firstCommit ? "first-to-newest" : "two-most-recent";
  applyVersionPreset();
}

function buildVersionOptionRecords(versions, firstCommit) {
  const seen = new Set();
  const records = [];
  if (firstCommit) {
    records.push(firstCommit);
    seen.add(firstCommit.sha);
  }
  versions.forEach((version) => {
    if (!seen.has(version.sha)) {
      records.push(version);
      seen.add(version.sha);
    }
  });
  return records.map((version) => ({
    sha: version.sha,
    label: formatVersionOption(version),
  }));
}

function applyVersionPreset() {
  const newest = state.versions[0];
  const previous = state.versions[1];
  if (elements.versionPresetSelect.value === "first-to-newest" && state.firstCommit && newest) {
    elements.baseBranchSelect.value = state.firstCommit.sha;
    elements.headBranchSelect.value = newest.sha;
  }
  if (elements.versionPresetSelect.value === "two-most-recent" && newest && previous) {
    elements.baseBranchSelect.value = previous.sha;
    elements.headBranchSelect.value = newest.sha;
  }
}

async function handleCompareVersions(event) {
  event.preventDefault();
  const baseRef = elements.baseBranchSelect.value;
  const headRef = elements.headBranchSelect.value;
  if (!state.repoUrl) {
    setBranchStatus("error", "Analyze a repository first.");
    return;
  }
  if (!baseRef || !headRef) {
    setBranchStatus("error", "Choose both versions.");
    return;
  }

  setBranchLoading(true, "Comparing versions...");
  resetCompareResult();
  try {
    const response = await fetch("/compare-versions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_url: state.repoUrl,
        base_ref: baseRef,
        head_ref: headRef,
        base_label: selectedOptionText(elements.baseBranchSelect),
        head_label: selectedOptionText(elements.headBranchSelect),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Version comparison failed.");
    }
    state.comparison = payload;
    state.compareId = payload.compare_id;
    renderComparison(payload);
    setBranchStatus("success", "Comparison ready");
  } catch (error) {
    setBranchStatus("error", error.message || "Version comparison failed.");
  } finally {
    setBranchLoading(false);
  }
}

function renderComparison(comparison) {
  elements.compareResult.classList.remove("hidden");
  elements.compareStatus.textContent = formatCompareStatus(comparison.status);
  elements.compareCommits.textContent = `${comparison.total_commits} commit${comparison.total_commits !== 1 ? "s" : ""}`;
  elements.compareFiles.textContent = `${comparison.files_changed} file${comparison.files_changed !== 1 ? "s" : ""}`;
  elements.compareDiffSize.textContent = `+${comparison.total_additions} / -${comparison.total_deletions}`;
  renderCompareFileList(comparison.changed_files || []);
  elements.compareQuestionInput.disabled = false;
  elements.askCompareButton.disabled = false;
  elements.compareQuestionInput.value = "What are the important differences between these versions?";
  elements.compareAnswer.classList.add("hidden");
  elements.compareAnswer.replaceChildren();
}

function renderCompareFileList(files) {
  elements.compareFileList.replaceChildren();
  if (!files.length) {
    const empty = document.createElement("div");
    empty.className = "path-empty";
    empty.textContent = "No file changes returned.";
    elements.compareFileList.appendChild(empty);
    return;
  }

  files.slice(0, 80).forEach((file) => {
    const row = document.createElement("div");
    row.className = "compare-file-row";

    const name = document.createElement("p");
    name.className = "compare-file-name";
    name.textContent = file.filename;
    name.title = file.filename;

    const meta = document.createElement("div");
    meta.className = "compare-file-meta";
    meta.appendChild(buildCompareBadge(file.status, "status"));
    meta.appendChild(buildCompareBadge(`+${file.additions}`, "added"));
    meta.appendChild(buildCompareBadge(`-${file.deletions}`, "removed"));

    row.append(name, meta);
    elements.compareFileList.appendChild(row);
  });

  if (files.length > 80) {
    const overflow = document.createElement("div");
    overflow.className = "path-empty";
    overflow.textContent = `${files.length - 80} more files omitted from the visible list.`;
    elements.compareFileList.appendChild(overflow);
  }
}

function buildCompareBadge(text, kind) {
  const badge = document.createElement("span");
  badge.className = `compare-badge ${kind}`;
  badge.textContent = text;
  return badge;
}

async function handleAskCompare(event) {
  event.preventDefault();
  const question = elements.compareQuestionInput.value.trim();
  if (!state.compareId) {
    setBranchStatus("error", "Run a version comparison first.");
    return;
  }
  if (!question) {
    setBranchStatus("error", "Enter a question about the version differences.");
    return;
  }

  setBranchLoading(true, "Answering from version diff context...");
  renderCompareAnswer(question, "", true);
  try {
    const response = await fetch("/ask-version-compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        compare_id: state.compareId,
        question,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Version comparison question failed.");
    }
    renderCompareAnswer(question, payload.answer, false);
    setBranchStatus("success", "Answered from version comparison data");
  } catch (error) {
    elements.compareAnswer.classList.add("hidden");
    setBranchStatus("error", error.message || "Version comparison question failed.");
  } finally {
    setBranchLoading(false);
  }
}

function renderCompareAnswer(question, answer, loading) {
  elements.compareAnswer.classList.remove("hidden");
  elements.compareAnswer.replaceChildren();

  const questionNode = document.createElement("p");
  questionNode.className = "compare-answer-question";
  questionNode.textContent = question;
  elements.compareAnswer.appendChild(questionNode);

  const body = document.createElement("div");
  body.className = "answer-text";
  if (loading) {
    const pending = document.createElement("p");
    pending.textContent = "Reading the version comparison...";
    body.appendChild(pending);
  } else {
    splitAnswerParagraphs(answer).forEach((paragraph) => {
      const p = document.createElement("p");
      p.textContent = paragraph;
      body.appendChild(p);
    });
  }
  elements.compareAnswer.appendChild(body);
}

function resetCompareResult() {
  state.comparison = null;
  state.compareId = "";
  elements.compareResult.classList.add("hidden");
  elements.compareFileList.replaceChildren();
  elements.compareQuestionInput.value = "";
  elements.compareQuestionInput.disabled = true;
  elements.askCompareButton.disabled = true;
  elements.compareAnswer.classList.add("hidden");
  elements.compareAnswer.replaceChildren();
}

function setBranchLoading(isLoading, message = "") {
  elements.refreshBranchesButton.disabled = isLoading || !state.repoUrl;
  elements.versionPresetSelect.disabled = isLoading || !state.repoUrl || !state.versions.length;
  elements.baseBranchSelect.disabled = isLoading || !state.repoUrl || !state.versions.length;
  elements.headBranchSelect.disabled = isLoading || !state.repoUrl || !state.versions.length;
  elements.compareButton.disabled = isLoading || !state.repoUrl || !state.versions.length;
  elements.askCompareButton.disabled = isLoading || !state.compareId;
  elements.compareQuestionInput.disabled = isLoading || !state.compareId;
  if (isLoading) {
    setBranchStatus("info", message || "Working...");
  }
}

function setBranchStatus(kind, text) {
  if (!kind || !text) {
    elements.branchStatus.className = "cache-indicator hidden";
    elements.branchStatus.textContent = "";
    return;
  }
  elements.branchStatus.className = `cache-indicator ${kind}`;
  elements.branchStatus.textContent = text;
}

function formatCompareStatus(status) {
  if (!status) return "Unknown";
  return status
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatVersionOption(version) {
  const date = version.date ? new Date(version.date) : null;
  const dateLabel = date && !Number.isNaN(date.getTime()) ? date.toLocaleDateString() : version.date;
  return `${version.short_sha || version.sha.slice(0, 7)} - ${version.message || "Commit"}${dateLabel ? ` (${dateLabel})` : ""}`;
}

function selectedOptionText(select) {
  return select.options[select.selectedIndex]?.textContent || select.value;
}

function renderRepoSummary(summary, analyzePayload = null) {
  elements.repoSummaryEmpty.classList.add("hidden");
  elements.repoSummary.classList.remove("hidden");

  elements.summaryName.textContent = summary.repo_name;
  elements.summaryBranch.textContent = `Branch: ${summary.branch}`;

  const fileCount = analyzePayload?.files_indexed ?? summary.files_indexed ?? 0;
  const chunkCount = analyzePayload?.chunks_created ?? summary.chunks_indexed ?? 0;
  renderSummaryParagraphs(summary, fileCount, chunkCount);
  renderGlobalContext(summary);
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

function renderGlobalContext(summary) {
  const graphContext = buildGraphContextModel(summary);
  elements.summaryGlobalContext.replaceChildren();
  elements.summaryGlobalContext.dataset.state = summary.global_context?.trim() ? "generated" : "fallback";

  // Use stored dependency_links.length as the canonical dependency count —
  // dependencyCount is regex-extracted from narrative text and can disagree.
  const storedEdgeCount = graphContext.dependencyLinks.length;
  const metrics = document.createElement("div");
  metrics.className = "graph-context-grid";
  metrics.appendChild(buildGraphMetric("Symbols", String(graphContext.symbolCount)));
  metrics.appendChild(buildGraphMetric("Dependencies", String(storedEdgeCount)));
  elements.summaryGlobalContext.appendChild(metrics);

  const pathSection = document.createElement("div");
  pathSection.className = "graph-paths-card";

  const pathTitle = document.createElement("p");
  pathTitle.className = "graph-section-label";
  pathTitle.textContent = "Key Dependencies";
  pathSection.appendChild(pathTitle);

  if (graphContext.criticalPaths.length) {
    const pathList = document.createElement("div");
    pathList.className = "graph-path-list";
    graphContext.criticalPaths.forEach((path) => {
      pathList.appendChild(buildCriticalPath(path));
    });
    pathSection.appendChild(pathList);
  } else if (graphContext.dependencyLinks.length) {
    const pathList = document.createElement("div");
    pathList.className = "graph-path-list";
    graphContext.dependencyLinks.forEach((path) => {
      pathList.appendChild(buildCriticalPath(path));
    });
    pathSection.appendChild(pathList);
  } else {
    const empty = document.createElement("p");
    empty.className = "graph-section-empty";
    empty.textContent = "No graph links surfaced yet.";
    pathSection.appendChild(empty);
  }

  elements.summaryGlobalContext.appendChild(pathSection);

  const sections = document.createElement("div");
  sections.className = "graph-context-sections";
  sections.appendChild(buildGraphSection("Graph Hubs", graphContext.hubs));
  sections.appendChild(buildGraphSection("Most Connected", graphContext.connectedFiles));
  elements.summaryGlobalContext.appendChild(sections);

  if (!graphContext.criticalPaths.length && !graphContext.dependencyLinks.length) {
    const summaryLine = document.createElement("p");
    summaryLine.className = "graph-context-summary";
    summaryLine.textContent = graphContext.summaryText;
    elements.summaryGlobalContext.appendChild(summaryLine);
  }
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

function splitAnswerParagraphs(answerText) {
  if (!answerText || !answerText.trim()) {
    return [];
  }

  const normalized = answerText.replace(/\r\n/g, "\n").trim();
  const explicitParagraphs = normalized
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  if (explicitParagraphs.length > 1) {
    return explicitParagraphs;
  }

  const lineParagraphs = normalized
    .split(/\n+/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  if (lineParagraphs.length > 1) {
    return lineParagraphs;
  }

  const sentences = normalized.match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g) || [normalized];
  const cleanedSentences = sentences.map((sentence) => sentence.trim()).filter(Boolean);

  if (cleanedSentences.length <= 2) {
    return [normalized];
  }

  const paragraphs = [];
  for (let index = 0; index < cleanedSentences.length; index += 2) {
    paragraphs.push(cleanedSentences.slice(index, index + 2).join(" "));
  }
  return paragraphs;
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

function buildFallbackGraphContext(summary) {
  const sentences = [
    `${summary.repo_name} has ${summary.files_indexed || 0} indexed files and ${summary.chunks_indexed || 0} retrieval chunks available for graph-style global context.`,
  ];

  if (summary.probable_entry_points?.length) {
    sentences.push(`Likely entrypoints include ${summary.probable_entry_points.slice(0, 3).join(", ")}.`);
  }
  if (summary.probable_training_files?.length) {
    sentences.push(`Training-related files include ${summary.probable_training_files.slice(0, 3).join(", ")}.`);
  }
  if (summary.probable_inference_files?.length) {
    sentences.push(`Inference-related files include ${summary.probable_inference_files.slice(0, 3).join(", ")}.`);
  }
  if (summary.probable_config_files?.length) {
    sentences.push(`Configuration is likely concentrated in ${summary.probable_config_files.slice(0, 3).join(", ")}.`);
  }
  if (summary.probable_data_files?.length) {
    sentences.push(`Data-loading appears in ${summary.probable_data_files.slice(0, 3).join(", ")}.`);
  }
  if (summary.key_files?.length) {
    sentences.push(`Key files include ${summary.key_files.slice(0, 4).join(", ")}.`);
  }

  return sentences.join(" ");
}

function buildGraphContextModel(summary) {
  const rawText = (summary.global_context || "").trim();
  return {
    symbolCount: extractCount(rawText, /(\d+)\s+named symbols/i),
    dependencyCount: extractCount(rawText, /(\d+)\s+inferred file dependency links/i),
    criticalPaths: summary.critical_paths || [],
    dependencyLinks: summary.dependency_links || [],
    connectedFiles: extractList(rawText, "Most connected files in the inferred dependency graph:"),
    hubs: summary.graph_hubs || buildGraphHubs(summary),
    summaryText: rawText || buildFallbackGraphContext(summary),
  };
}

function buildGraphHubs(summary) {
  return [
    ...(summary.probable_entry_points || []).slice(0, 2),
    ...(summary.probable_config_files || []).slice(0, 1),
    ...(summary.probable_data_files || []).slice(0, 1),
  ].filter((item, index, array) => array.indexOf(item) === index);
}

function extractCount(text, pattern) {
  const match = text.match(pattern);
  return match ? Number(match[1]) : 0;
}

function extractList(text, prefix) {
  const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = text.match(new RegExp(`${escapedPrefix}\\s*([^.]*)\\.`, "i"));
  if (!match) {
    return [];
  }
  return match[1]
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildGraphMetric(label, value) {
  const card = document.createElement("div");
  card.className = "graph-metric";

  const metricLabel = document.createElement("p");
  metricLabel.className = "graph-metric-label";
  metricLabel.textContent = label;

  const metricValue = document.createElement("p");
  metricValue.className = "graph-metric-value";
  metricValue.textContent = value;

  card.append(metricLabel, metricValue);
  return card;
}

function buildGraphSection(label, items) {
  const section = document.createElement("div");
  section.className = "graph-section";

  const title = document.createElement("p");
  title.className = "graph-section-label";
  title.textContent = label;
  section.appendChild(title);

  if (!items || !items.length) {
    const empty = document.createElement("p");
    empty.className = "graph-section-empty";
    empty.textContent = "None surfaced";
    section.appendChild(empty);
    return section;
  }

  const list = document.createElement("div");
  list.className = "graph-section-list";
  items.slice(0, 4).forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "graph-chip";
    chip.textContent = item;
    list.appendChild(chip);
  });
  section.appendChild(list);
  return section;
}

function buildCriticalPath(path) {
  const row = document.createElement("div");
  row.className = "graph-path";

  path.forEach((segment, index) => {
    const chip = document.createElement("span");
    chip.className = "graph-path-node";
    chip.textContent = segment;
    row.appendChild(chip);

    if (index < path.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "graph-path-arrow";
      arrow.textContent = "->";
      row.appendChild(arrow);
    }
  });

  return row;
}

function handleNewThread() {
  state.hasConversation = false;
  elements.chatMessages.querySelectorAll(".chat-exchange").forEach((el) => el.remove());
  elements.chatMessages.classList.add("hidden");
  elements.chatIdle.classList.remove("hidden");
  elements.suggestionsArea.classList.remove("hidden");
  elements.newThreadButton.classList.add("hidden");
  elements.sourcesEmpty.classList.remove("hidden");
  elements.sourcesList.classList.add("hidden");
  elements.sourcesList.replaceChildren();
  elements.sourcesCount.classList.add("hidden");
  elements.sourcesCount.textContent = "";
  elements.questionInput.value = "";
  elements.questionInput.focus();
}

function showThinking(visible) {
  if (!state.hasConversation) {
    state.hasConversation = true;
    elements.chatIdle.classList.add("hidden");
    elements.chatMessages.classList.remove("hidden");
    elements.newThreadButton.classList.remove("hidden");
  }
  elements.chatThinking.classList.toggle("hidden", !visible);
  if (visible) {
    elements.chatBody.scrollTop = elements.chatBody.scrollHeight;
  }
}

function appendChatExchange(question, answer) {
  // First exchange — hide idle, show messages (may already be done by showThinking)
  if (!state.hasConversation) {
    state.hasConversation = true;
    elements.chatIdle.classList.add("hidden");
    elements.chatMessages.classList.remove("hidden");
    elements.newThreadButton.classList.remove("hidden");
  }

  const exchange = document.createElement("div");
  exchange.className = "chat-exchange";

  // User bubble
  const userBubble = document.createElement("div");
  userBubble.className = "chat-bubble--user";

  const userLabel = document.createElement("span");
  userLabel.className = "thread-role thread-role--user";
  userLabel.textContent = "You";

  const questionText = document.createElement("p");
  questionText.className = "thread-question-text";
  questionText.textContent = question;

  userBubble.append(userLabel, questionText);

  // Assistant bubble
  const assistantBubble = document.createElement("div");
  assistantBubble.className = "chat-bubble--assistant";

  const assistantLabel = document.createElement("span");
  assistantLabel.className = "thread-role thread-role--assistant";
  assistantLabel.textContent = "Assistant";

  const answerDiv = document.createElement("div");
  answerDiv.className = "answer-text";

  const paragraphs = splitAnswerParagraphs(answer);
  if (!paragraphs.length && answer.trim()) paragraphs.push(answer.trim());
  paragraphs.forEach((paragraph) => {
    const p = document.createElement("p");
    p.textContent = paragraph;
    answerDiv.appendChild(p);
  });

  assistantBubble.append(assistantLabel, answerDiv);
  exchange.append(userBubble, assistantBubble);
  // Insert before the thinking indicator so it stays at the bottom
  elements.chatMessages.insertBefore(exchange, elements.chatThinking);

  // Scroll to bottom
  elements.chatBody.scrollTop = elements.chatBody.scrollHeight;
}

function clearAnswer() {
  state.hasConversation = false;
  elements.chatIdle.classList.remove("hidden");
  elements.chatMessages.querySelectorAll(".chat-exchange").forEach((el) => el.remove());
  elements.chatMessages.classList.add("hidden");
  elements.chatThinking.classList.add("hidden");
  elements.newThreadButton.classList.add("hidden");
  elements.sourcesEmpty.classList.remove("hidden");
  elements.sourcesList.classList.add("hidden");
  elements.sourcesList.replaceChildren();
  elements.sourcesCount.classList.add("hidden");
  elements.sourcesCount.textContent = "";
}

function clearRepoSummary() {
  elements.repoSummaryEmpty.classList.remove("hidden");
  elements.repoSummary.classList.add("hidden");
  elements.summaryText.replaceChildren();
  elements.summaryGlobalContext.replaceChildren();
  elements.summaryGlobalContext.textContent = "Repository-wide graph context will appear here after analysis.";
  elements.summaryGlobalContext.dataset.state = "empty";
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
  // Reset suggestions
  elements.suggestionsArea.classList.add("hidden");
  elements.queryHint.textContent = "Analyze a repository to start asking questions.";
  // Reset history
  elements.historyPanel.classList.add("hidden");
  elements.historyHeadSha.textContent = "";
  elements.changesBanner.classList.add("hidden");
  elements.changesBanner.replaceChildren();
  elements.commitList.classList.add("hidden");
  elements.commitList.replaceChildren();
  enableCompareWorkspace(false);
  resetCompareResult();
  state.versions = [];
  state.firstCommit = null;
  state.versionsLoadedFor = "";
}

function renderSources(sources) {
  elements.sourcesList.replaceChildren();

  if (!sources.length) {
    elements.sourcesEmpty.classList.remove("hidden");
    elements.sourcesList.classList.add("hidden");
    elements.sourcesCount.classList.add("hidden");
    return;
  }

  elements.sourcesEmpty.classList.add("hidden");
  elements.sourcesList.classList.remove("hidden");
  elements.sourcesCount.classList.remove("hidden");
  elements.sourcesCount.textContent = `${sources.length} source${sources.length !== 1 ? "s" : ""}`;

  sources.forEach((source, index) => {
    const card = document.createElement("article");
    card.className = "source-card";

    // ── Top section ──
    const top = document.createElement("div");
    top.className = "source-card-top";

    // File row: citation number + path
    const fileRow = document.createElement("div");
    fileRow.className = "source-file-row";

    const num = document.createElement("span");
    num.className = "source-citation-num";
    num.textContent = index + 1;

    const filePath = document.createElement("p");
    filePath.className = "source-file-path";
    filePath.textContent = source.file_path;
    filePath.title = source.file_path;

    fileRow.append(num, filePath);
    top.appendChild(fileRow);

    // Badges: language, chunk type, line range
    const badges = document.createElement("div");
    badges.className = "source-badges";

    if (source.language) {
      const langBadge = document.createElement("span");
      langBadge.className = "source-badge source-badge--lang";
      langBadge.dataset.lang = source.language;
      langBadge.textContent = source.language;
      badges.appendChild(langBadge);
    }

    const typeBadge = document.createElement("span");
    typeBadge.className = "source-badge source-badge--type";
    typeBadge.textContent = source.chunk_type.replace(/_/g, " ");
    badges.appendChild(typeBadge);

    if (source.start_line || source.end_line) {
      const linesBadge = document.createElement("span");
      linesBadge.className = "source-badge source-badge--lines";
      linesBadge.textContent = formatLineRange(source.start_line, source.end_line);
      badges.appendChild(linesBadge);
    }

    top.appendChild(badges);

    if (source.short_summary) {
      const summary = document.createElement("p");
      summary.className = "source-summary-text";
      summary.textContent = source.short_summary;
      top.appendChild(summary);
    }

    card.appendChild(top);

    // ── Score bar ──
    const scoreRow = document.createElement("div");
    scoreRow.className = "source-score-row";

    const scoreLabel = document.createElement("span");
    scoreLabel.className = "source-score-label";
    scoreLabel.textContent = "Relevance";

    const barWrap = document.createElement("div");
    barWrap.className = "source-score-bar-wrap";

    const bar = document.createElement("div");
    bar.className = "source-score-bar";
    bar.style.width = `${Math.min(Number(source.score) * 100, 100).toFixed(1)}%`;
    barWrap.appendChild(bar);

    const scoreVal = document.createElement("span");
    scoreVal.className = "source-score-value";
    scoreVal.textContent = Number(source.score).toFixed(2);

    scoreRow.append(scoreLabel, barWrap, scoreVal);
    card.appendChild(scoreRow);

    // ── Code snippet ──
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
  document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.disabled = isLoading || !state.repoUrl;
  });

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

// ---- Version History ----

function renderHistory(headSha, commits, changes, activitySummary = "") {
  elements.historyPanel.classList.remove("hidden");

  // HEAD SHA chip
  elements.historyHeadSha.textContent = headSha ? `HEAD ${headSha.slice(0, 7)}` : "";

  // Changes banner
  if (changes) {
    renderChangesBanner(changes);
    elements.changesBanner.classList.remove("hidden");
  } else {
    elements.changesBanner.classList.add("hidden");
    elements.changesBanner.replaceChildren();
  }

  // Commit list (Timeline pane)
  elements.commitList.replaceChildren();
  if (!commits || commits.length === 0) {
    elements.commitListEmpty.classList.remove("hidden");
    elements.commitList.classList.add("hidden");
  } else {
    elements.commitListEmpty.classList.add("hidden");
    elements.commitList.classList.remove("hidden");
    commits.forEach((commit) => {
      elements.commitList.appendChild(buildCommitItem(commit));
    });
  }

  // Activity Flow pane
  renderActivityFlow(commits, activitySummary);
}

function renderChangesBanner(changes) {
  elements.changesBanner.replaceChildren();

  const title = document.createElement("p");
  title.className = "changes-banner-title";
  title.textContent = "Changes detected since last analysis";
  elements.changesBanner.appendChild(title);

  const stats = document.createElement("div");
  stats.className = "changes-stats";

  stats.appendChild(buildChangesStat("files", `${changes.files_changed} file${changes.files_changed !== 1 ? "s" : ""} changed`));
  if ((changes.files_added ?? 0) > 0) {
    stats.appendChild(buildChangesStat("added", `+${changes.files_added} new file${changes.files_added !== 1 ? "s" : ""}`));
  }
  if ((changes.files_removed ?? 0) > 0) {
    stats.appendChild(buildChangesStat("removed", `−${changes.files_removed} deleted file${changes.files_removed !== 1 ? "s" : ""}`));
  }
  if ((changes.files_modified ?? 0) > 0) {
    stats.appendChild(buildChangesStat("files", `${changes.files_modified} modified`));
  }
  elements.changesBanner.appendChild(stats);

  const shaRange = document.createElement("p");
  shaRange.className = "changes-sha-range";
  shaRange.textContent = `${changes.old_sha.slice(0, 7)} → ${changes.new_sha.slice(0, 7)}`;
  elements.changesBanner.appendChild(shaRange);

  if (changes.changed_files && changes.changed_files.length > 0) {
    const label = document.createElement("p");
    label.className = "changes-files-label";
    label.textContent = "Modified files";
    elements.changesBanner.appendChild(label);

    const fileList = document.createElement("div");
    fileList.className = "changes-files-list";
    const visible = changes.changed_files.slice(0, 12);
    const overflow = changes.changed_files.slice(12);

    visible.forEach((filePath) => {
      const chip = document.createElement("span");
      chip.className = "changes-file-chip";
      chip.textContent = filePath;
      fileList.appendChild(chip);
    });

    if (overflow.length > 0) {
      const more = document.createElement("span");
      more.className = "changes-file-chip";
      more.style.cursor = "pointer";
      more.style.color = "var(--teal)";
      more.textContent = `+${overflow.length} more`;
      more.addEventListener("click", () => {
        overflow.forEach((filePath) => {
          const chip = document.createElement("span");
          chip.className = "changes-file-chip";
          chip.textContent = filePath;
          fileList.insertBefore(chip, more);
        });
        more.remove();
      });
      fileList.appendChild(more);
    }

    elements.changesBanner.appendChild(fileList);
  }
}

function buildChangesStat(kind, text) {
  const el = document.createElement("span");
  el.className = `changes-stat ${kind}`;
  el.textContent = text;
  return el;
}

// ---- Activity Flow ----

function renderActivityFlow(commits, activitySummary) {
  // Activity summary
  if (activitySummary && activitySummary.trim()) {
    elements.activitySummaryText.textContent = activitySummary.trim();
    elements.activitySummaryBlock.classList.remove("hidden");
  } else {
    elements.activitySummaryBlock.classList.add("hidden");
  }

  elements.activityFlow.replaceChildren();

  if (!commits || commits.length === 0) {
    elements.activityFlowEmpty.classList.remove("hidden");
    elements.activityFlow.classList.add("hidden");
    return;
  }

  elements.activityFlowEmpty.classList.add("hidden");
  elements.activityFlow.classList.remove("hidden");

  commits.forEach((commit, index) => {
    elements.activityFlow.appendChild(buildFlowCommit(commit, index === commits.length - 1));
  });
}

function buildFlowCommit(commit, isLast) {
  const block = document.createElement("div");
  block.className = "flow-commit";

  // Left: vertical track (dot + line)
  const track = document.createElement("div");
  track.className = "flow-track";

  const dot = document.createElement("div");
  dot.className = "flow-dot";
  track.appendChild(dot);

  if (!isLast) {
    const line = document.createElement("div");
    line.className = "flow-line";
    track.appendChild(line);
  }

  // Right: content
  const content = document.createElement("div");
  content.className = "flow-content";

  // Header row: SHA badge + message + date
  const headerRow = document.createElement("div");
  headerRow.className = "flow-header-row";

  const sha = document.createElement("span");
  sha.className = "commit-sha-badge";
  sha.textContent = commit.short_sha || commit.sha.slice(0, 7);

  const msg = document.createElement("p");
  msg.className = "flow-message";
  msg.textContent = commit.message;

  const date = document.createElement("span");
  date.className = "commit-date";
  date.textContent = formatRelativeDate(commit.date);
  date.title = `${commit.author_name} · ${commit.date}`;

  headerRow.append(sha, msg, date);
  content.appendChild(headerRow);

  // File change badges
  const fileChanges = commit.file_changes || [];
  if (fileChanges.length > 0) {
    const filesRow = document.createElement("div");
    filesRow.className = "flow-files";

    const MAX_VISIBLE = 6;
    const visible = fileChanges.slice(0, MAX_VISIBLE);
    const overflow = fileChanges.slice(MAX_VISIBLE);

    visible.forEach((fc) => {
      const badge = document.createElement("span");
      badge.className = `flow-file-badge flow-file-badge--${fc.status}`;
      badge.textContent = getFileName(fc.path);
      badge.title = `${fc.status}: ${fc.path}`;
      filesRow.appendChild(badge);
    });

    if (overflow.length > 0) {
      const more = document.createElement("span");
      more.className = "flow-file-badge flow-file-badge--more";
      more.textContent = `+${overflow.length} more`;
      more.title = overflow.map((f) => f.path).join("\n");
      filesRow.appendChild(more);
    }

    content.appendChild(filesRow);
  }

  // Per-commit LLM explanation
  if (commit.summary && commit.summary.trim()) {
    const explanation = document.createElement("p");
    explanation.className = "flow-commit-summary";
    explanation.textContent = commit.summary.trim();
    content.appendChild(explanation);
  }

  block.append(track, content);
  return block;
}

function buildCommitItem(commit) {
  const item = document.createElement("div");
  item.className = "commit-item";

  const sha = document.createElement("span");
  sha.className = "commit-sha-badge";
  sha.textContent = commit.short_sha || commit.sha.slice(0, 7);

  const body = document.createElement("div");
  body.className = "commit-body";

  const msg = document.createElement("p");
  msg.className = "commit-msg";
  msg.textContent = commit.message;

  const author = document.createElement("p");
  author.className = "commit-author";
  author.textContent = commit.author_name;

  body.append(msg, author);

  const date = document.createElement("span");
  date.className = "commit-date";
  date.textContent = formatRelativeDate(commit.date);
  date.title = commit.date;

  item.append(sha, body, date);
  return item;
}

function formatRelativeDate(dateStr) {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return dateStr;

  const now = new Date();
  const diffMs = now - date;
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffMonths = Math.floor(diffDays / 30);
  const diffYears = Math.floor(diffDays / 365);

  if (diffSecs < 60) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 30) return `${diffDays}d ago`;
  if (diffMonths < 12) return `${diffMonths}mo ago`;
  return `${diffYears}y ago`;
}

// ---- Onboarding ----

const AVATAR_COLORS = [
  { bg: "rgba(40,122,120,0.16)", color: "#287a78" },
  { bg: "rgba(198,91,56,0.14)", color: "#c65b38" },
  { bg: "rgba(80,60,140,0.14)", color: "#3d2e6e" },
  { bg: "rgba(23,50,74,0.12)", color: "#17324a" },
  { bg: "rgba(40,100,180,0.14)", color: "#1a5499" },
  { bg: "rgba(140,80,40,0.14)", color: "#7a4a20" },
];

async function handleGenerateOnboarding() {
  if (!state.repoUrl) return;

  elements.onboardingEmpty.classList.add("hidden");
  elements.onboardingContent.classList.add("hidden");
  elements.onboardingLoading.classList.remove("hidden");
  elements.generateOnboardingButton.disabled = true;

  try {
    const response = await fetch(`/onboarding?repo_url=${encodeURIComponent(state.repoUrl)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Failed to generate onboarding guide.");
    renderOnboarding(payload);
  } catch (err) {
    elements.onboardingLoading.classList.add("hidden");
    elements.onboardingEmpty.classList.remove("hidden");
    elements.onboardingEmpty.textContent = err.message || "Failed to generate guide.";
  } finally {
    elements.generateOnboardingButton.disabled = false;
  }
}

function renderOnboarding(data) {
  elements.onboardingLoading.classList.add("hidden");
  elements.onboardingEmpty.classList.add("hidden");

  // Reading order
  elements.onboardingReadingOrder.replaceChildren();
  (data.reading_order || []).forEach((step) => {
    const row = document.createElement("div");
    row.className = "reading-step";

    const num = document.createElement("div");
    num.className = "reading-step-number";
    num.textContent = step.step;

    const body = document.createElement("div");
    body.className = "reading-step-body";

    const file = document.createElement("p");
    file.className = "reading-step-file";
    file.textContent = step.file_path;

    const reason = document.createElement("p");
    reason.className = "reading-step-reason";
    reason.textContent = step.reason;

    body.append(file, reason);

    const role = document.createElement("span");
    role.className = "reading-step-role";
    role.textContent = step.role;

    row.append(num, body, role);
    elements.onboardingReadingOrder.appendChild(row);
  });

  // Core concepts
  elements.onboardingConcepts.replaceChildren();
  (data.core_concepts || []).forEach((concept) => {
    const card = document.createElement("div");
    card.className = "concept-card";

    const name = document.createElement("p");
    name.className = "concept-name";
    name.textContent = concept.name;

    const desc = document.createElement("p");
    desc.className = "concept-description";
    desc.textContent = concept.description;

    card.append(name, desc);

    if (concept.key_files && concept.key_files.length) {
      const chips = document.createElement("div");
      chips.className = "concept-files";
      concept.key_files.slice(0, 4).forEach((f) => {
        const chip = document.createElement("span");
        chip.className = "concept-file-chip";
        chip.textContent = f;
        chips.appendChild(chip);
      });
      card.appendChild(chips);
    }

    elements.onboardingConcepts.appendChild(card);
  });

  // Contributors
  elements.onboardingContributors.replaceChildren();
  (data.contributors || []).forEach((contributor, index) => {
    const colors = AVATAR_COLORS[index % AVATAR_COLORS.length];
    const card = document.createElement("div");
    card.className = "contributor-card";

    const avatar = document.createElement("div");
    avatar.className = "contributor-avatar";
    avatar.style.background = colors.bg;
    avatar.style.color = colors.color;
    avatar.style.border = `2px solid ${colors.bg}`;
    avatar.textContent = contributor.name.slice(0, 2).toUpperCase();

    const info = document.createElement("div");

    const name = document.createElement("p");
    name.className = "contributor-name";
    name.textContent = contributor.name;

    const meta = document.createElement("div");
    meta.className = "contributor-meta";
    meta.innerHTML = `<span>${contributor.commits} commit${contributor.commits !== 1 ? "s" : ""}</span><span class="contributor-divider">·</span><span>${contributor.focus_area}</span>`;

    info.append(name, meta);
    card.append(avatar, info);
    elements.onboardingContributors.appendChild(card);
  });

  // Complexity note
  if (data.complexity_note && data.complexity_note.trim()) {
    elements.onboardingComplexityText.textContent = data.complexity_note.trim();
    elements.onboardingComplexity.classList.remove("hidden");
  } else {
    elements.onboardingComplexity.classList.add("hidden");
  }

  elements.onboardingContent.classList.remove("hidden");
}

// ═══════════════════════════════════════════════════════════════════════════
// Repository Map — D3 Force-Directed Dependency Graph
// ═══════════════════════════════════════════════════════════════════════════

const LANG_COLORS = {
  python:     "#3776ab",
  javascript: "#f0db4f",
  typescript: "#3178c6",
  go:         "#00add8",
  rust:       "#ce422b",
  java:       "#ed8b00",
  cpp:        "#6295cb",
  c:          "#555599",
  ruby:       "#cc342d",
  html:       "#e34c26",
  css:        "#264de4",
  markdown:   "#8b949e",
  json:       "#40a040",
  yaml:       "#cb171e",
  unknown:    "#8b949e",
};

function nodeRadius(d) {
  // Minimum 22px so circles are always clearly readable; scale with line count
  return Math.min(46, Math.max(22, Math.sqrt(d.line_count / 1.8)));
}

let mapState = {
  nodes: [],
  edges: [],
  simulation: null,
  clusterOn: false,
  selectedId: null,
  zoomBehavior: null,
  svgG: null,        // the inner <g> that zoom transforms
};

async function handleGenerateMap() {
  if (!state.repoUrl) return;
  elements.mapEmpty.classList.add("hidden");
  elements.mapWorkspace.classList.add("hidden");
  elements.mapLoading.classList.remove("hidden");
  elements.mapSearch.disabled = true;
  elements.mapClusterToggle.disabled = true;

  try {
    const resp = await fetch(`/repo-map?repo_url=${encodeURIComponent(state.repoUrl)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Failed to generate map.");
    elements.mapLoading.classList.add("hidden");
    elements.mapWorkspace.classList.remove("hidden");
    elements.mapSearch.disabled = false;
    elements.mapClusterToggle.disabled = false;
    // Wait one frame so the SVG container has real clientWidth/clientHeight
    await new Promise(r => requestAnimationFrame(r));
    renderRepoMap(data);
  } catch (err) {
    elements.mapLoading.classList.add("hidden");
    elements.mapEmpty.classList.remove("hidden");
    elements.mapEmpty.textContent = err.message || "Failed to generate map.";
  }
}

function renderRepoMap(data) {
  mapState.nodes = data.nodes.map(n => ({ ...n }));
  mapState.edges = data.edges.map(e => ({ ...e }));
  mapState.clusterOn = false;
  mapState.selectedId = null;
  elements.mapClusterToggle.textContent = "Cluster by directory";

  // Build legend from languages present
  const langs = [...new Set(mapState.nodes.map(n => n.language))].sort();
  elements.mapLegend.replaceChildren();
  langs.forEach(lang => {
    const item = document.createElement("div");
    item.className = "map-legend-item";
    const dot = document.createElement("span");
    dot.className = "map-legend-dot";
    dot.style.background = LANG_COLORS[lang] || LANG_COLORS.unknown;
    const label = document.createElement("span");
    label.textContent = lang;
    item.append(dot, label);
    elements.mapLegend.appendChild(item);
  });

  // Show node/edge count beneath legend
  const statsEl = document.getElementById("map-graph-stats");
  if (statsEl) {
    statsEl.textContent = `${mapState.nodes.length} files · ${mapState.edges.length} dependencies`;
  }
  console.log(`[RepoMap] nodes=${mapState.nodes.length} edges=${mapState.edges.length}`, mapState.edges.slice(0,5));

  buildD3Graph();

  // Search handler
  elements.mapSearch.oninput = () => {
    const q = elements.mapSearch.value.trim().toLowerCase();
    if (!q) return;
    const match = mapState.nodes.find(n =>
      n.file_path.toLowerCase().includes(q)
    );
    if (match && match.x != null) {
      const svg = d3.select(elements.mapSvg);
      const w = elements.mapSvg.clientWidth;
      const h = elements.mapSvg.clientHeight;
      svg.transition().duration(600).call(
        mapState.zoomBehavior.transform,
        d3.zoomIdentity.translate(w / 2, h / 2).scale(2.2).translate(-match.x, -match.y)
      );
      showSidePanel(match);
    }
  };

  // Cluster toggle
  elements.mapClusterToggle.onclick = () => {
    mapState.clusterOn = !mapState.clusterOn;
    elements.mapClusterToggle.textContent = mapState.clusterOn
      ? "Free layout"
      : "Cluster by directory";
    applyClusterForce();
  };
}

function buildD3Graph() {
  const svgEl = elements.mapSvg;
  const wrap = svgEl.parentElement;

  // Measure the actual rendered container — never use 0
  const w = Math.max(wrap.clientWidth  || wrap.offsetWidth  || 800, 600);
  const h = Math.max(wrap.clientHeight || wrap.offsetHeight || 620, 500);

  // Set explicit SVG dimensions so the element fills the space
  svgEl.setAttribute("width",  w);
  svgEl.setAttribute("height", h);

  // Clear previous render
  d3.select(svgEl).selectAll("*").remove();
  elements.mapSidePanel.classList.add("hidden");

  const svg = d3.select(svgEl);

  // Arrowhead sits at the line endpoint (already clipped to node edge)
  svg.append("defs").append("marker")
    .attr("id", "arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 8)
    .attr("refY", 0)
    .attr("markerWidth", 7)
    .attr("markerHeight", 7)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#8da7c4");

  const g = svg.append("g");
  mapState.svgG = g;

  // Zoom
  const zoom = d3.zoom()
    .scaleExtent([0.1, 8])
    .on("zoom", (event) => g.attr("transform", event.transform));
  mapState.zoomBehavior = zoom;
  svg.call(zoom);

  // ── Pre-spread nodes in a circle so simulation starts with room to breathe ──
  const n = mapState.nodes.length;
  const spreadR = Math.min(w, h) * 0.38;
  mapState.nodes.forEach((d, i) => {
    const angle = (i / n) * 2 * Math.PI;
    d.x = w / 2 + spreadR * Math.cos(angle);
    d.y = h / 2 + spreadR * Math.sin(angle);
  });

  // Edges (drawn first so nodes render on top)
  const link = g.append("g").attr("class", "links")
    .selectAll("line")
    .data(mapState.edges)
    .join("line")
    .attr("class", "map-link")
    .attr("marker-end", "url(#arrowhead)");

  // Nodes
  const node = g.append("g").attr("class", "nodes")
    .selectAll("g")
    .data(mapState.nodes)
    .join("g")
    .attr("class", "map-node")
    .call(d3.drag()
      .on("start", (event, d) => {
        if (!event.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => {
        if (!event.active) sim.alphaTarget(0);
        d.fx = null; d.fy = null;
      })
    );

  // Circle — large enough to always be visible
  node.append("circle")
    .attr("r", d => nodeRadius(d))
    .attr("fill", d => LANG_COLORS[d.language] || LANG_COLORS.unknown);

  // Filename label below the node
  node.append("text")
    .attr("dy", d => nodeRadius(d) + 15)
    .attr("text-anchor", "middle")
    .text(d => {
      const name = d.file_path.split("/").pop();
      return name.length > 22 ? name.slice(0, 20) + "…" : name;
    });

  // ── Hover: highlight only direct neighbors ──
  const linkedSet = (d, edgesArr) => {
    const neighbors = new Set([d.id]);
    edgesArr.forEach(e => {
      const sid = typeof e.source === "object" ? e.source.id : e.source;
      const tid = typeof e.target === "object" ? e.target.id : e.target;
      if (sid === d.id) neighbors.add(tid);
      if (tid === d.id) neighbors.add(sid);
    });
    return neighbors;
  };

  node
    .on("mouseenter", (event, d) => {
      const neighbors = linkedSet(d, mapState.edges);
      node.classed("dimmed", n => !neighbors.has(n.id));
      link.classed("dimmed", e => {
        const sid = typeof e.source === "object" ? e.source.id : e.source;
        const tid = typeof e.target === "object" ? e.target.id : e.target;
        return sid !== d.id && tid !== d.id;
      });
      link.classed("highlighted", e => {
        const sid = typeof e.source === "object" ? e.source.id : e.source;
        const tid = typeof e.target === "object" ? e.target.id : e.target;
        return sid === d.id || tid === d.id;
      });
    })
    .on("mouseleave", () => {
      node.classed("dimmed", false);
      link.classed("dimmed", false).classed("highlighted", false);
    })
    .on("click", (event, d) => {
      event.stopPropagation();
      mapState.selectedId = d.id;
      node.classed("selected", n => n.id === d.id);
      showSidePanel(d);
    });

  svg.on("click", () => {
    node.classed("selected", false);
    elements.mapSidePanel.classList.add("hidden");
    mapState.selectedId = null;
  });

  // ── Force simulation ──
  // High charge pushes nodes apart; strong link pulls connected ones close
  const sim = d3.forceSimulation(mapState.nodes)
    .force("link", d3.forceLink(mapState.edges)
      .id(d => d.id)
      .distance(110)
      .strength(0.7))
    .force("charge", d3.forceManyBody().strength(-500))
    .force("center", d3.forceCenter(w / 2, h / 2).strength(0.05))
    .force("collide", d3.forceCollide(d => nodeRadius(d) + 14))
    .alphaDecay(0.025)     // slower decay → more time to settle
    .on("tick", () => {
      // Draw lines from source-node-edge to target-node-edge (not center)
      // so the arrowhead sits cleanly at the circle boundary
      link
        .attr("x1", d => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          return d.source.x + (dx / dist) * (nodeRadius(d.source) + 2);
        })
        .attr("y1", d => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          return d.source.y + (dy / dist) * (nodeRadius(d.source) + 2);
        })
        .attr("x2", d => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          return d.target.x - (dx / dist) * (nodeRadius(d.target) + 2);
        })
        .attr("y2", d => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          return d.target.y - (dy / dist) * (nodeRadius(d.target) + 2);
        });
      node.attr("transform", d => `translate(${d.x},${d.y})`);
    })
    .on("end", () => {
      // Auto-zoom to fit all nodes with 5% padding
      const bounds = g.node().getBBox();
      if (bounds.width === 0 || bounds.height === 0) return;
      const pad = 60;
      const scale = Math.min(
        (w - pad * 2) / bounds.width,
        (h - pad * 2) / bounds.height,
        1.4
      );
      const tx = w / 2 - scale * (bounds.x + bounds.width  / 2);
      const ty = h / 2 - scale * (bounds.y + bounds.height / 2);
      svg.transition().duration(900).ease(d3.easeCubicOut).call(
        zoom.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
      );
    });

  mapState.simulation = sim;
  mapState._link = link;
  mapState._node = node;
}

function applyClusterForce() {
  if (!mapState.simulation) return;
  if (mapState.clusterOn) {
    // Compute directory centroids
    const dirs = [...new Set(mapState.nodes.map(n => n.file_path.split("/")[0]))];
    const w = elements.mapSvg.clientWidth || 900;
    const h = elements.mapSvg.clientHeight || 550;
    const cols = Math.ceil(Math.sqrt(dirs.length));
    const centroids = {};
    dirs.forEach((dir, i) => {
      centroids[dir] = {
        x: (i % cols + 0.5) * (w / cols),
        y: (Math.floor(i / cols) + 0.5) * (h / Math.ceil(dirs.length / cols)),
      };
    });
    mapState.simulation
      .force("x", d3.forceX(d => {
        const dir = d.file_path.split("/")[0];
        return centroids[dir]?.x ?? w / 2;
      }).strength(0.12))
      .force("y", d3.forceY(d => {
        const dir = d.file_path.split("/")[0];
        return centroids[dir]?.y ?? h / 2;
      }).strength(0.12));
  } else {
    mapState.simulation.force("x", null).force("y", null);
  }
  mapState.simulation.alpha(0.6).restart();
}

function showSidePanel(d) {
  elements.mapSidePath.textContent = d.file_path;
  elements.mapSideLang.textContent = d.language;
  elements.mapSideRole.textContent = d.role || "general";
  elements.mapSideLines.textContent = `~${d.line_count} lines`;
  elements.mapSideSummary.textContent = d.short_summary || "";

  elements.mapSideSymbols.replaceChildren();
  (d.key_symbols || []).forEach(sym => {
    const chip = document.createElement("span");
    chip.className = "map-side-symbol";
    chip.textContent = sym;
    elements.mapSideSymbols.appendChild(chip);
  });

  // Connections
  const imports = mapState.edges
    .filter(e => (typeof e.source === "object" ? e.source.id : e.source) === d.id)
    .map(e => typeof e.target === "object" ? e.target.id : e.target);
  const importedBy = mapState.edges
    .filter(e => (typeof e.target === "object" ? e.target.id : e.target) === d.id)
    .map(e => typeof e.source === "object" ? e.source.id : e.source);

  const renderConnList = (container, labelEl, label, paths) => {
    container.replaceChildren();
    if (paths.length === 0) {
      labelEl.textContent = "";
      return;
    }
    labelEl.textContent = `${label} (${paths.length})`;
    paths.slice(0, 6).forEach(p => {
      const item = document.createElement("div");
      item.className = "map-side-conn-item";
      item.textContent = p;
      item.title = p;
      item.onclick = (ev) => {
        ev.stopPropagation();
        const target = mapState.nodes.find(n => n.id === p);
        if (target && target.x != null) {
          const svgEl = elements.mapSvg;
          const w = svgEl.clientWidth;
          const h = svgEl.clientHeight;
          d3.select(svgEl).transition().duration(500).call(
            mapState.zoomBehavior.transform,
            d3.zoomIdentity.translate(w / 2, h / 2).scale(2.0).translate(-target.x, -target.y)
          );
          if (mapState._node) {
            mapState._node.classed("selected", n => n.id === p);
          }
          showSidePanel(target);
        }
      };
      container.appendChild(item);
    });
  };

  renderConnList(elements.mapSideImports, elements.mapSideImportsLabel, "Imports", imports);
  renderConnList(elements.mapSideImportedBy, elements.mapSideImportedByLabel, "Imported by", importedBy);

  elements.mapSidePanel.classList.remove("hidden");
}
