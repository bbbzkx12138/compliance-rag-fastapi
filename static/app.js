(function () {
  "use strict";

  const state = {
    documents: [],
    meta: null,
    scope: "current",
    department: "all",
    activeView: "assistant",
    docQuery: "",
    knowledgeQuery: "",
    knowledgeStatus: "all",
    evidenceTab: "sources",
    evidenceItems: [],
    lastAnswer: null,
    busy: false,
    isComposing: false,
    evaluation: null,
    online: false
  };

  const elements = {};
  const modeLabels = {
    keyword: "关键词 BM25",
    vector: "字词向量",
    hybrid: "混合检索"
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function icon(name) {
    return `<svg aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  }

  function formatDate(value) {
    return value ? value.replaceAll("-", ".") : "长期有效";
  }

  function percent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
  }

  function normalizeText(value) {
    return String(value || "").toLowerCase().replace(/\s+/g, "");
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {})
      }
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        payload?.error?.message || payload?.detail || `HTTP ${response.status}`
      );
    }
    return payload;
  }

  async function initialize() {
    [
      "assistantView",
      "knowledgeView",
      "evaluationView",
      "conversation",
      "welcome",
      "questionForm",
      "questionInput",
      "sendButton",
      "libraryPanel",
      "libraryToggle",
      "libraryBackdrop",
      "closeLibrary",
      "docSearch",
      "departmentFilter",
      "documentCount",
      "documentList",
      "backendMeta",
      "scopeLabel",
      "contextLabel",
      "evidencePanel",
      "evidenceToggle",
      "evidenceBackdrop",
      "closeEvidence",
      "evidenceEmpty",
      "sourcesPanel",
      "tracePanel",
      "openTrace",
      "effectiveDocCount",
      "expiredDocCount",
      "chunkCount",
      "knowledgeSearch",
      "statusFilter",
      "documentTableBody",
      "runEvaluation",
      "evaluationTableBody",
      "uploadDialog",
      "uploadForm",
      "uploadTitle",
      "uploadVersion",
      "uploadDepartment",
      "uploadDate",
      "uploadFile",
      "uploadFileLabel",
      "serviceStatus",
      "serviceStatusText",
      "welcomeDocs",
      "welcomeChunks",
      "welcomeCases",
      "toastRegion"
    ].forEach((id) => {
      elements[id] = document.getElementById(id);
    });

    elements.uploadDate.value = localDateString();
    bindEvents();

    const parameters = new URLSearchParams(location.search);
    const initialView = parameters.get("view");
    if (["assistant", "knowledge", "evaluation"].includes(initialView)) {
      switchView(initialView);
    }

    try {
      await bootstrap();
      setServiceStatus("online", "FastAPI 已连接");
      renderAll();
      await runEvaluation(false);
      const initialQuestion = parameters.get("q");
      if (initialQuestion) {
        window.setTimeout(() => {
          elements.questionInput.value = initialQuestion;
          resizeTextarea();
          elements.questionForm.requestSubmit();
        }, 100);
      }
    } catch (error) {
      setServiceStatus("offline", "FastAPI 未连接");
      showToast(`无法连接 Python 后端：${error.message}`, true);
      renderDocuments();
      renderDocumentTable();
    }
  }

  function bindEvents() {
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => switchView(button.dataset.view));
    });
    document.getElementById("brandButton").addEventListener("click", () => {
      switchView("assistant");
    });

    document.querySelectorAll("[data-scope]").forEach((button) => {
      button.addEventListener("click", () => setScope(button.dataset.scope));
    });

    document.querySelectorAll("[data-question]").forEach((button) => {
      button.addEventListener("click", () => {
        elements.questionInput.value = button.dataset.question;
        resizeTextarea();
        elements.questionForm.requestSubmit();
      });
    });

    elements.questionForm.addEventListener("submit", handleQuestion);
    elements.questionInput.addEventListener("input", resizeTextarea);
    elements.questionInput.addEventListener("compositionstart", () => {
      state.isComposing = true;
    });
    elements.questionInput.addEventListener("compositionend", () => {
      state.isComposing = false;
    });
    elements.questionInput.addEventListener("keydown", (event) => {
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !state.isComposing &&
        !event.isComposing
      ) {
        event.preventDefault();
        elements.questionForm.requestSubmit();
      }
    });

    elements.docSearch.addEventListener("input", () => {
      state.docQuery = elements.docSearch.value.trim();
      renderDocuments();
    });
    elements.departmentFilter.addEventListener("change", () => {
      state.department = elements.departmentFilter.value;
      renderDocuments();
      updateContextLabel();
    });
    elements.knowledgeSearch.addEventListener("input", () => {
      state.knowledgeQuery = elements.knowledgeSearch.value.trim();
      renderDocumentTable();
    });
    elements.statusFilter.querySelectorAll("[data-status]").forEach((button) => {
      button.addEventListener("click", () => {
        state.knowledgeStatus = button.dataset.status;
        elements.statusFilter
          .querySelectorAll("[data-status]")
          .forEach((item) => item.classList.toggle("is-active", item === button));
        renderDocumentTable();
      });
    });

    elements.documentList.addEventListener("click", (event) => {
      const item = event.target.closest("[data-doc-id]");
      if (item) showDocument(item.dataset.docId);
    });
    elements.documentTableBody.addEventListener("click", (event) => {
      const button = event.target.closest("[data-preview-doc]");
      if (button) showDocument(button.dataset.previewDoc);
    });

    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => setEvidenceTab(button.dataset.tab));
    });
    elements.evidenceToggle.addEventListener("click", openEvidence);
    elements.closeEvidence.addEventListener("click", closeEvidence);
    elements.evidenceBackdrop.addEventListener("click", closeEvidence);
    elements.openTrace.addEventListener("click", () => {
      setEvidenceTab("trace");
      openEvidence();
    });

    elements.libraryToggle.addEventListener("click", () =>
      elements.libraryPanel.classList.add("is-open")
    );
    elements.closeLibrary.addEventListener("click", closeLibrary);
    elements.libraryBackdrop.addEventListener("click", closeLibrary);

    elements.sourcesPanel.addEventListener("click", (event) => {
      const card = event.target.closest("[data-evidence-index]");
      if (card) selectEvidence(Number(card.dataset.evidenceIndex));
    });
    elements.conversation.addEventListener("click", (event) => {
      const citation = event.target.closest("[data-citation-index]");
      if (!citation) return;
      setEvidenceTab("sources");
      selectEvidence(Number(citation.dataset.citationIndex));
      openEvidence();
    });

    document.querySelectorAll(".upload-trigger").forEach((button) => {
      button.addEventListener("click", () => {
        elements.uploadDialog.showModal();
        elements.uploadTitle.focus();
      });
    });
    document.querySelectorAll("[data-close]").forEach((button) => {
      button.addEventListener("click", () => elements.uploadDialog.close());
    });
    elements.uploadFile.addEventListener("change", updateFileName);
    elements.uploadForm.addEventListener("submit", handleUpload);

    elements.runEvaluation.addEventListener("click", () => runEvaluation(true));
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      closeLibrary();
      if (!elements.uploadDialog.open) closeEvidence();
    });
  }

  async function bootstrap() {
    const [health, documentResponse] = await Promise.all([
      api("/api/health"),
      api("/api/documents")
    ]);
    state.online = true;
    state.meta = health.meta || documentResponse.meta;
    state.documents = documentResponse.documents || [];
  }

  function renderAll() {
    populateDepartments();
    renderDocuments();
    renderDocumentTable();
    updateSummary();
    updateContextLabel();
  }

  function setServiceStatus(status, text) {
    elements.serviceStatus.classList.remove("is-checking", "is-offline");
    if (status === "offline") elements.serviceStatus.classList.add("is-offline");
    if (status === "checking") elements.serviceStatus.classList.add("is-checking");
    elements.serviceStatusText.textContent = text;
  }

  function switchView(view) {
    state.activeView = view;
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.view === view);
    });
    document.querySelectorAll("[data-view-panel]").forEach((panel) => {
      const active = panel.dataset.viewPanel === view;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
    closeLibrary();
    if (view === "knowledge") renderDocumentTable();
  }

  function setScope(scope) {
    state.scope = scope;
    document.querySelectorAll("[data-scope]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.scope === scope);
    });
    elements.scopeLabel.textContent =
      scope === "current" ? "仅检索现行有效制度" : "检索现行与历史版本";
    updateContextLabel();
  }

  function updateContextLabel() {
    const department =
      state.department === "all" ? "全部部门" : state.department;
    const scope = state.scope === "current" ? "现行制度" : "全部版本";
    elements.contextLabel.textContent = `${department} · ${scope}`;
  }

  function populateDepartments() {
    const departments = Array.from(
      new Set(state.documents.map((documentItem) => documentItem.department))
    ).sort((left, right) => left.localeCompare(right, "zh-CN"));
    const current = elements.departmentFilter.value;
    elements.departmentFilter.innerHTML =
      `<option value="all">全部部门</option>` +
      departments
        .map(
          (department) =>
            `<option value="${escapeHtml(department)}">${escapeHtml(department)}</option>`
        )
        .join("");
    elements.departmentFilter.value = departments.includes(current)
      ? current
      : "all";
  }

  function renderDocuments() {
    const query = normalizeText(state.docQuery);
    const filtered = state.documents.filter((documentItem) => {
      if (
        state.department !== "all" &&
        documentItem.department !== state.department
      ) {
        return false;
      }
      if (!query) return true;
      return normalizeText(
        [
          documentItem.title,
          documentItem.short_title,
          documentItem.category,
          documentItem.tags.join(" ")
        ].join(" ")
      ).includes(query);
    });
    elements.documentCount.textContent = `${filtered.length} 份文档`;
    if (!filtered.length) {
      elements.documentList.innerHTML = `<div class="empty">没有匹配的制度</div>`;
      return;
    }
    elements.documentList.innerHTML = filtered
      .map(
        (documentItem) => `
          <button class="doc-item ${documentItem.status === "expired" ? "is-expired" : ""}"
            type="button" data-doc-id="${escapeHtml(documentItem.id)}">
            <span class="doc-icon">${icon("file")}</span>
            <span class="doc-copy">
              <strong>${escapeHtml(documentItem.short_title)}</strong>
              <span>${escapeHtml(documentItem.category)} · ${escapeHtml(documentItem.department)}</span>
            </span>
            <span class="doc-version">${escapeHtml(documentItem.version)}</span>
          </button>
        `
      )
      .join("");
  }

  function renderDocumentTable() {
    const query = normalizeText(state.knowledgeQuery);
    const filtered = state.documents.filter((documentItem) => {
      if (
        state.knowledgeStatus !== "all" &&
        documentItem.status !== state.knowledgeStatus
      ) {
        return false;
      }
      if (!query) return true;
      return normalizeText(
        [
          documentItem.title,
          documentItem.version,
          documentItem.department,
          documentItem.category,
          documentItem.tags.join(" ")
        ].join(" ")
      ).includes(query);
    });
    if (!filtered.length) {
      elements.documentTableBody.innerHTML =
        `<tr><td colspan="7" class="empty">没有匹配的制度文档</td></tr>`;
      return;
    }
    elements.documentTableBody.innerHTML = filtered
      .map(
        (documentItem) => `
          <tr>
            <td>
              <div class="table-doc">
                <span class="doc-icon">${icon("file")}</span>
                <span>
                  <strong>${escapeHtml(documentItem.title)}</strong>
                  <span>${escapeHtml(documentItem.category)} · ${escapeHtml(documentItem.source_type)}</span>
                </span>
              </div>
            </td>
            <td><span class="version-tag">${escapeHtml(documentItem.version)}</span></td>
            <td><span class="status-pill ${documentItem.status === "expired" ? "expired" : ""}">${escapeHtml(documentItem.status_label)}</span></td>
            <td>${escapeHtml(documentItem.department)}</td>
            <td>${formatDate(documentItem.effective_date)}</td>
            <td>${documentItem.section_count}</td>
            <td><button class="table-action" type="button" data-preview-doc="${escapeHtml(documentItem.id)}">${icon("external")}</button></td>
          </tr>
        `
      )
      .join("");
  }

  function updateSummary() {
    const effective = state.documents.filter(
      (item) => item.status === "effective"
    ).length;
    const expired = state.documents.filter(
      (item) => item.status === "expired"
    ).length;
    elements.effectiveDocCount.textContent = effective;
    elements.expiredDocCount.textContent = expired;
    elements.chunkCount.textContent = state.meta?.chunk_count ?? "--";
    elements.welcomeDocs.textContent = state.meta?.document_count ?? "--";
    elements.welcomeChunks.textContent = state.meta?.chunk_count ?? "--";
    elements.welcomeCases.textContent =
      state.meta?.evaluation_case_count ?? "--";
    elements.backendMeta.textContent =
      `${state.meta?.document_count ?? 0} 文档 · ${state.meta?.chunk_count ?? 0} 切片`;
  }

  async function showDocument(documentId) {
    try {
      const payload = await api(`/api/documents/${encodeURIComponent(documentId)}`);
      const documentItem = payload.document;
      document
        .querySelectorAll(".doc-item")
        .forEach((item) =>
          item.classList.toggle("is-active", item.dataset.docId === documentId)
        );
      state.evidenceItems = documentItem.sections.map((section) => ({
        document: documentItem,
        section,
        score: null
      }));
      state.lastAnswer = null;
      setEvidenceTab("sources");
      renderEvidence();
      openEvidence();
    } catch (error) {
      showToast(`读取制度失败：${error.message}`, true);
    }
  }

  async function handleQuestion(event) {
    event.preventDefault();
    const question = elements.questionInput.value.trim();
    if (!question || state.busy) return;
    if (!state.online) {
      showToast("FastAPI 后端未连接，无法执行服务端检索", true);
      return;
    }

    state.busy = true;
    elements.welcome.hidden = true;
    elements.sendButton.disabled = true;
    appendUser(question);
    elements.questionInput.value = "";
    resizeTextarea();
    const thinking = appendThinking();

    try {
      const payload = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          question,
          scope: state.scope,
          department: state.department,
          mode: "hybrid"
        })
      });
      thinking.remove();
      const answer = payload.answer;
      state.lastAnswer = answer;
      state.evidenceItems = answer.evidence.map((result) => ({
        document: result.document,
        section: result.section,
        score: result.score,
        coverage: result.coverage
      }));
      renderAnswer(answer);
      renderEvidence();
    } catch (error) {
      thinking.remove();
      appendApiError(error.message);
      showToast(`问答请求失败：${error.message}`, true);
    } finally {
      state.busy = false;
      elements.sendButton.disabled = false;
      elements.questionInput.focus();
      scrollBottom();
    }
  }

  function appendUser(question) {
    const block = document.createElement("div");
    block.className = "message";
    block.innerHTML = `<div class="user-row"><div class="user-bubble">${escapeHtml(question)}</div></div>`;
    elements.conversation.appendChild(block);
    scrollBottom();
  }

  function appendThinking() {
    const block = document.createElement("div");
    block.className = "message";
    block.innerHTML = `
      <div class="answer-row">
        <span class="assistant-icon">${icon("shield")}</span>
        <div class="thinking"><span class="dots"><i></i><i></i><i></i></span>FastAPI 正在检索条款并校验证据</div>
      </div>
    `;
    elements.conversation.appendChild(block);
    scrollBottom();
    return block;
  }

  function appendApiError(message) {
    const block = document.createElement("div");
    block.className = "message";
    block.innerHTML = `
      <div class="answer-row">
        <span class="assistant-icon">${icon("alert")}</span>
        <div class="refusal">
          <h3>${icon("alert")}后端请求失败</h3>
          <p>${escapeHtml(message)}</p>
        </div>
      </div>
    `;
    elements.conversation.appendChild(block);
    scrollBottom();
  }

  function renderAnswer(answer) {
    const block = document.createElement("div");
    block.className = "message";
    const now = new Date().toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit"
    });
    const badgeClass = answer.refused
      ? "low"
      : answer.confidence === "medium"
        ? "medium"
        : "";
    const badgeText = answer.refused
      ? "证据不足"
      : answer.confidence === "high"
        ? "高可信"
        : "需核对";

    const content = answer.refused
      ? `
        <div class="refusal">
          <h3>${icon("alert")}${escapeHtml(answer.title)}</h3>
          <p>${escapeHtml(answer.lead)} ${escapeHtml(answer.note)}</p>
        </div>
      `
      : `
        <p class="answer-lead">${escapeHtml(answer.lead)}</p>
        <ol class="answer-points">
          ${answer.points
            .map(
              (point, index) => `
                <li class="answer-point ${point.status === "expired" ? "expired" : ""}">
                  <p>${escapeHtml(point.text)}<button class="citation" type="button"
                    data-citation-index="${index}">${point.citation}</button></p>
                </li>
              `
            )
            .join("")}
        </ol>
        <p class="answer-note">${icon("alert")}<span>${escapeHtml(answer.note)}</span></p>
      `;

    const uniqueDocs = [];
    answer.evidence.forEach((result) => {
      if (!uniqueDocs.some((item) => item.id === result.document.id)) {
        uniqueDocs.push(result.document);
      }
    });

    block.innerHTML = `
      <div class="answer-row">
        <span class="assistant-icon">${icon("shield")}</span>
        <div>
          <div class="answer-meta">
            <strong>规镜助手</strong>
            <span class="badge ${badgeClass}">${badgeText}</span>
            <small>${now} · ${modeLabels[answer.trace.mode]} · ${answer.trace.elapsed_ms.toFixed(1)} ms</small>
          </div>
          ${content}
          ${
            uniqueDocs.length
              ? `<div class="source-row">${uniqueDocs
                  .map(
                    (documentItem) =>
                      `<span class="source-pill">${icon("file")}${escapeHtml(documentItem.short_title)} ${escapeHtml(documentItem.version)}</span>`
                  )
                  .join("")}</div>`
              : ""
          }
        </div>
      </div>
    `;
    elements.conversation.appendChild(block);
    scrollBottom();
  }

  function scrollBottom() {
    requestAnimationFrame(() => {
      elements.conversation.scrollTop = elements.conversation.scrollHeight;
    });
  }

  function resizeTextarea() {
    elements.questionInput.style.height = "auto";
    elements.questionInput.style.height = `${Math.min(
      elements.questionInput.scrollHeight,
      126
    )}px`;
  }

  function setEvidenceTab(tab) {
    state.evidenceTab = tab;
    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.tab === tab);
    });
    document.querySelectorAll(".pane").forEach((pane) => {
      pane.classList.toggle("is-active", pane.id === `${tab}Panel`);
    });
  }

  function renderEvidence() {
    elements.evidenceEmpty.hidden = state.evidenceItems.length > 0;
    renderSources();
    renderTrace();
  }

  function renderSources() {
    if (!state.evidenceItems.length) {
      elements.sourcesPanel.innerHTML = "";
      return;
    }
    elements.sourcesPanel.innerHTML = state.evidenceItems
      .map((result, index) => {
        const documentItem = result.document;
        const score =
          typeof result.score === "number" ? percent(result.score) : "原文";
        return `
          <article class="evidence-card ${index === 0 ? "is-selected" : ""}"
            id="evidence-${index}" data-evidence-index="${index}">
            <div class="evidence-head">
              <div><span class="num">${index + 1}</span><h3>${escapeHtml(documentItem.short_title)} ${escapeHtml(documentItem.version)}</h3></div>
              <span class="status ${documentItem.status === "expired" ? "expired" : ""}">${documentItem.status === "expired" ? "已失效" : "现行有效"}</span>
            </div>
            <div class="evidence-meta">
              <span>${icon("file")}${escapeHtml(result.section.heading)}</span>
              <span>${icon("library")}第 ${result.section.page} 页</span>
              <span>${icon("clock")}${formatDate(documentItem.effective_date)}</span>
            </div>
            <p class="excerpt">${escapeHtml(result.section.text)}</p>
            <div class="score">
              <div class="score-track"><i style="width:${
                typeof result.score === "number"
                  ? Math.max(8, Math.min(100, Math.round(result.score * 100)))
                  : 100
              }%"></i></div>
              <span>${score}</span>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function renderTrace() {
    if (!state.lastAnswer) {
      elements.tracePanel.innerHTML = `
        <div class="evidence-empty" style="padding:48px 18px">
          <span>${icon("filter")}</span>
          <h3>${state.evidenceItems.length ? "制度原文预览" : "尚无检索轨迹"}</h3>
          <p>${state.evidenceItems.length ? "完成一次问答后可查看查询扩展、候选得分与阈值。" : "检索后展示 FastAPI 服务端的查询扩展和候选排序。"}</p>
        </div>
      `;
      return;
    }
    const trace = state.lastAnswer.trace;
    elements.tracePanel.innerHTML = `
      <section class="trace-section">
        <h3>服务端查询</h3>
        <div class="trace-grid">
          <div class="trace-cell"><span>检索模式</span><strong>${modeLabels[trace.mode]}</strong></div>
          <div class="trace-cell"><span>耗时</span><strong>${trace.elapsed_ms.toFixed(1)} ms</strong></div>
          <div class="trace-cell"><span>版本范围</span><strong>${trace.scope === "current" ? "仅现行有效" : "含历史版本"}</strong></div>
          <div class="trace-cell"><span>部门过滤</span><strong>${escapeHtml(trace.department === "all" ? "全部部门" : trace.department)}</strong></div>
        </div>
      </section>
      <section class="trace-section">
        <h3>查询扩展</h3>
        <div class="terms">
          ${trace.expanded_terms.map((term) => `<span class="term">+ ${escapeHtml(term)}</span>`).join("") || `<span class="trace-note">未触发同义词扩展</span>`}
          ${trace.unseen_terms.slice(0, 5).map((term) => `<span class="term unseen">? ${escapeHtml(term)}</span>`).join("")}
        </div>
      </section>
      <section class="trace-section">
        <h3>候选条款</h3>
        ${trace.candidates
          .map(
            (candidate) => `
              <div class="candidate">
                <div><strong>${escapeHtml(candidate.title)} ${escapeHtml(candidate.version)}</strong><small>覆盖 ${percent(candidate.coverage)} · ${candidate.status === "expired" ? "历史版本" : "现行制度"}</small></div>
                <b>${Math.round(candidate.score * 100)}%</b>
              </div>
            `
          )
          .join("")}
      </section>
      <section class="trace-section">
        <h3>证据判定</h3>
        <div class="trace-grid">
          <div class="trace-cell"><span>最低相关性</span><strong>${percent(trace.thresholds.minimum_score)}</strong></div>
          <div class="trace-cell"><span>最低覆盖率</span><strong>${percent(trace.thresholds.minimum_coverage)}</strong></div>
        </div>
        <p class="trace-note" style="margin-top:8px">${state.lastAnswer.refused ? "本次未通过证据充分性检查，FastAPI 返回了拒答。" : "本次通过证据充分性检查，答案仅引用上方条款。"}</p>
      </section>
    `;
  }

  function selectEvidence(index) {
    setEvidenceTab("sources");
    elements.sourcesPanel
      .querySelectorAll(".evidence-card")
      .forEach((card, cardIndex) => {
        card.classList.toggle("is-selected", cardIndex === index);
      });
    document.getElementById(`evidence-${index}`)?.scrollIntoView({
      behavior: "smooth",
      block: "nearest"
    });
  }

  function openEvidence() {
    elements.evidencePanel.classList.add("is-open");
  }

  function closeEvidence() {
    elements.evidencePanel.classList.remove("is-open");
  }

  function closeLibrary() {
    elements.libraryPanel.classList.remove("is-open");
  }

  function updateFileName() {
    const file = elements.uploadFile.files?.[0];
    elements.uploadFileLabel.textContent = file
      ? file.name
      : "选择 Markdown 或文本文件";
  }

  async function handleUpload(event) {
    event.preventDefault();
    const file = elements.uploadFile.files?.[0];
    if (!file) {
      showToast("请选择要导入的文档", true);
      return;
    }
    if (file.size > 1024 * 1024) {
      showToast("文件超过 1 MB，请拆分后导入", true);
      return;
    }
    try {
      const content = await file.text();
      const payload = await api("/api/documents", {
        method: "POST",
        body: JSON.stringify({
          title: elements.uploadTitle.value.trim(),
          version: elements.uploadVersion.value.trim(),
          department: elements.uploadDepartment.value.trim(),
          effective_date: elements.uploadDate.value,
          content
        })
      });
      state.meta = payload.meta;
      const documentResponse = await api("/api/documents");
      state.documents = documentResponse.documents;
      renderAll();
      await runEvaluation(false);
      elements.uploadDialog.close();
      elements.uploadForm.reset();
      elements.uploadDate.value = localDateString();
      elements.uploadFileLabel.textContent = "选择 Markdown 或文本文件";
      showToast(`《${payload.document.title}》已写入 SQLite 并重建索引`);
      switchView("knowledge");
    } catch (error) {
      showToast(`导入失败：${error.message}`, true);
    }
  }

  async function runEvaluation(notify) {
    elements.runEvaluation.disabled = true;
    try {
      if (!state.online) throw new Error("FastAPI 后端未连接");
      const payload = await api("/api/evaluate", {
        method: "POST",
        body: JSON.stringify({
          modes: ["keyword", "vector", "hybrid"]
        })
      });
      state.evaluation = payload.results;
      renderEvaluation(payload.results);
      if (notify) {
        const hybrid = payload.results.hybrid;
        showToast(`评测完成 · 混合检索通过 ${hybrid.passed}/${hybrid.total}`);
      }
    } catch (error) {
      elements.evaluationTableBody.innerHTML =
        `<tr><td colspan="6" class="empty">${escapeHtml(error.message)}</td></tr>`;
      if (notify) showToast(`评测失败：${error.message}`, true);
    } finally {
      elements.runEvaluation.disabled = false;
    }
  }

  function renderEvaluation(results) {
    Object.entries(results).forEach(([mode, result]) => {
      document.querySelector(`[data-pass="${mode}"]`).textContent =
        percent(result.passed / result.total);
      document.querySelector(`[data-recall="${mode}"]`).textContent =
        percent(result.recall_at_3);
      document.querySelector(`[data-mrr="${mode}"]`).textContent =
        Number(result.mrr).toFixed(2);
    });
    elements.evaluationTableBody.innerHTML = results.hybrid.rows
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(row.id)}</td>
            <td class="query-cell">${escapeHtml(row.query)}</td>
            <td><div class="doc-tags">${
              row.expected_docs.length
                ? row.expected_docs
                    .map(
                      (docId) =>
                        `<span class="doc-tag">${escapeHtml(docId)}</span>`
                    )
                    .join("")
                : `<span class="doc-tag">应拒答</span>`
            }</div></td>
            <td><div class="term-tags">${
              row.citation_terms.length
                ? row.citation_terms
                    .map(
                      (term) =>
                        `<span class="term-tag">${escapeHtml(term)}</span>`
                    )
                    .join("")
                : `<span class="term-tag">${row.should_refuse ? "无证据" : "缺失"}</span>`
            }</div></td>
            <td>${row.latency_ms.toFixed(1)} ms</td>
            <td><span class="pass-fail ${row.passed ? "" : "fail"}">${row.passed ? "通过" : "需复核"}</span></td>
          </tr>
        `
      )
      .join("");
  }

  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `toast ${isError ? "error" : ""}`;
    toast.innerHTML = `${icon(isError ? "alert" : "check")}<span>${escapeHtml(message)}</span>`;
    elements.toastRegion.appendChild(toast);
    window.setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(4px)";
      toast.style.transition = "0.2s ease";
      window.setTimeout(() => toast.remove(), 220);
    }, 2800);
  }

  function localDateString() {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  initialize();
})();
