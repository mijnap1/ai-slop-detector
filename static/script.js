let doughnutChart = null;
let barChart = null;

const STORAGE_THEME_KEY = "ai-slop-detector-theme";
const STORAGE_HISTORY_KEY = "ai-slop-detector-history";
const MAX_HISTORY_ITEMS = 8;

const body = document.body;
const themeBtn = document.getElementById("theme-btn");
const infoBtn = document.getElementById("info-btn");
const modalOverlay = document.getElementById("modal-overlay");
const modalClose = document.getElementById("modal-close");
const modalCloseBtn = document.getElementById("modal-close-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const primaryBtnRow = document.getElementById("primary-btn-row");
const fileInput = document.getElementById("input-file");
const uploadFileLabel = document.getElementById("upload-file-label");
const historyCard = document.getElementById("history-card");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");

initializeTheme();
bindIconControls();
bindTabControls();
bindInputs();
renderHistory();

document.getElementById("analyze-btn").addEventListener("click", handleAnalyze);
document.getElementById("reset-btn").addEventListener("click", resetAll);
clearHistoryBtn.addEventListener("click", clearHistory);

function bindTabControls() {
  document.querySelectorAll(".tab-btn").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
      const isApi = tab.dataset.tab === "api";
      primaryBtnRow.classList.toggle("hidden", isApi);
      btnText.textContent = tab.dataset.tab === "compare"
        ? "Compare"
        : tab.dataset.tab === "url"
          ? "Scrape & Analyze"
          : tab.dataset.tab === "upload"
            ? "Analyze File"
            : "Analyze";
      hideError();
    });
  });
}

function bindInputs() {
  document.getElementById("input-text").addEventListener("input", function () {
    const words = getWordCount(this.value);
    document.getElementById("word-count").textContent = `${words} word${words !== 1 ? "s" : ""}`;
  });

  fileInput.addEventListener("change", () => {
    uploadFileLabel.textContent = fileInput.files[0]?.name || "No file selected";
  });
}

async function handleAnalyze() {
  const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
  hideError();
  setLoading(true);

  try {
    if (activeTab === "compare") {
      await runCompareMode();
    } else if (activeTab === "upload") {
      await runUploadMode();
    } else if (activeTab === "url") {
      await runJsonAnalysis({ url: document.getElementById("input-url").value.trim() });
    } else {
      await runJsonAnalysis({ text: document.getElementById("input-text").value.trim() });
    }
  } catch (error) {
    showError(error.message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
}

async function runJsonAnalysis(payload) {
  const res = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Analysis failed.");
  saveHistoryEntry(payload, data);
  showSingleResult(data);
}

async function runUploadMode() {
  if (!fileInput.files[0]) {
    throw new Error("Select a file first.");
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("/analyze", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "File analysis failed.");
  saveHistoryEntry({ uploaded_filename: fileInput.files[0].name }, data);
  showSingleResult(data);
}

async function runCompareMode() {
  const leftText = document.getElementById("compare-left").value.trim();
  const rightText = document.getElementById("compare-right").value.trim();

  const res = await fetch("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ left_text: leftText, right_text: rightText }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Comparison failed.");
  showCompareResults(data);
}

function initializeTheme() {
  const storedTheme = localStorage.getItem(STORAGE_THEME_KEY);
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(storedTheme || (prefersDark ? "dark" : "light"));
}

function bindIconControls() {
  themeBtn.addEventListener("click", () => {
    const nextTheme = body.classList.contains("dark") ? "light" : "dark";
    applyTheme(nextTheme);
    localStorage.setItem(STORAGE_THEME_KEY, nextTheme);
  });

  infoBtn.addEventListener("click", openModal);
  modalClose.addEventListener("click", closeModal);
  modalCloseBtn.addEventListener("click", closeModal);

  modalOverlay.addEventListener("click", event => {
    if (event.target === modalOverlay) closeModal();
  });

  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !modalOverlay.classList.contains("hidden")) {
      closeModal();
    }
  });
}

function applyTheme(theme) {
  const isDark = theme === "dark";
  body.classList.toggle("dark", isDark);
  themeBtn.setAttribute("aria-pressed", String(isDark));
  themeBtn.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
}

function openModal() {
  modalOverlay.classList.remove("hidden");
  infoBtn.setAttribute("aria-expanded", "true");
  body.style.overflow = "hidden";
  modalClose.focus();
}

function closeModal() {
  modalOverlay.classList.add("hidden");
  infoBtn.setAttribute("aria-expanded", "false");
  body.style.overflow = "";
  infoBtn.focus();
}

function showSingleResult(data) {
  document.getElementById("compare-results").classList.add("hidden");
  showReliability(data.reliability);
  renderVerdict(data);
  renderScoreBreakdown(data);
  renderCharts(data);
  renderStats(data);
  renderSentenceDetails(data.sentence_details || []);
  renderPhraseHits(data.phrase_hits || []);
  renderPreview(data);
  document.getElementById("results").classList.remove("hidden");
  setTimeout(() => {
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }, 50);
}

function showCompareResults(data) {
  document.getElementById("results").classList.add("hidden");
  const grid = document.getElementById("compare-result-grid");
  grid.innerHTML = "";

  [
    { label: "Sample A", result: data.left },
    { label: "Sample B", result: data.right },
  ].forEach(item => {
    const card = document.createElement("div");
    card.className = "card compare-card";
    card.innerHTML = `
      <div class="section-label">${item.label}</div>
      <div class="compare-score-row">
        <div class="compare-score ${item.result.verdict === "Human-Written" ? "is-human" : item.result.verdict === "AI-Generated" ? "is-ai" : ""}">${item.result.ai_score}% AI</div>
        <div class="compare-pill">${item.result.verdict}</div>
      </div>
      <p class="compare-copy">${item.result.confidence_label}</p>
      <p class="compare-copy muted">${item.result.reliability.message || `${item.result.word_count} words analyzed.`}</p>
    `;
    grid.appendChild(card);
  });

  document.getElementById("compare-results").classList.remove("hidden");
  setTimeout(() => {
    document.getElementById("compare-results").scrollIntoView({ behavior: "smooth", block: "start" });
  }, 50);
}

function showReliability(reliability) {
  const card = document.getElementById("reliability-card");
  const text = document.getElementById("reliability-text");
  if (!reliability || reliability.state === "normal") {
    card.classList.add("hidden");
    return;
  }
  text.textContent = reliability.message;
  card.classList.remove("hidden");
}

function renderVerdict(data) {
  const isAI = data.verdict === "AI-Generated";
  const isHuman = data.verdict === "Human-Written";
  const dominantScore = isAI ? data.ai_score : isHuman ? data.human_score : 50;

  const card = document.getElementById("verdict-card");
  card.classList.remove("is-ai", "is-human");
  if (isAI) card.classList.add("is-ai");
  if (isHuman) card.classList.add("is-human");

  const scoreEl = document.getElementById("verdict-score");
  scoreEl.className = `verdict-score ${isAI ? "is-ai" : isHuman ? "is-human" : ""}`;
  scoreEl.textContent = "0%";
  animateCount("verdict-score", 0, dominantScore, 700, v => `${Math.round(v)}%`);

  document.getElementById("verdict-label").textContent = `${data.verdict} · ${data.confidence_label}`;

  const badge = document.getElementById("verdict-badge");
  badge.className = `verdict-badge ${isAI ? "is-ai" : isHuman ? "is-human" : ""}`;
  badge.textContent = `${data.confidence}% confidence`;

  const desc = data.reliability?.message
    ? `${data.word_count.toLocaleString()} words analyzed. ${data.reliability.message}`
    : `${data.word_count.toLocaleString()} words analyzed across ${data.chunks_analyzed} chunk${data.chunks_analyzed !== 1 ? "s" : ""}.`;
  document.getElementById("verdict-desc").textContent = desc;
}

function renderScoreBreakdown(data) {
  setTimeout(() => {
    animateBar("bar-ai", "bar-ai-val", data.ai_score);
    animateBar("bar-human", "bar-human-val", data.human_score);
  }, 100);
}

function renderCharts(data) {
  const aiColor = "rgba(239,68,68,0.8)";
  const humanColor = "rgba(34,197,94,0.8)";
  const isAI = data.ai_score >= data.human_score;

  setTimeout(() => {
    const dCtx = document.getElementById("doughnut-chart").getContext("2d");
    if (doughnutChart) doughnutChart.destroy();
    doughnutChart = new Chart(dCtx, {
      type: "doughnut",
      data: {
        labels: ["AI", "Human"],
        datasets: [{
          data: [data.ai_score, data.human_score],
          backgroundColor: [aiColor, humanColor],
          borderWidth: 0,
        }],
      },
      options: {
        cutout: "72%",
        plugins: { legend: { display: false } },
        animation: { animateRotate: true, duration: 800, easing: "easeOutQuart" },
      },
    });

    const dCenter = document.getElementById("doughnut-center");
    dCenter.textContent = `${Math.max(data.ai_score, data.human_score)}%`;
    dCenter.style.color = isAI ? "#ef4444" : "#22c55e";
    dCenter.classList.remove("show");
    setTimeout(() => dCenter.classList.add("show"), 600);

    const bCtx = document.getElementById("bar-chart").getContext("2d");
    if (barChart) barChart.destroy();
    barChart = new Chart(bCtx, {
      type: "bar",
      data: {
        labels: ["AI", "Human"],
        datasets: [{
          data: [data.ai_score, data.human_score],
          backgroundColor: ["rgba(239,68,68,0.15)", "rgba(34,197,94,0.15)"],
          borderColor: ["#ef4444", "#22c55e"],
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        }],
      },
      options: {
        indexAxis: "y",
        scales: {
          x: {
            min: 0,
            max: 100,
            grid: { color: body.classList.contains("dark") ? "#1f2937" : "#f3f4f6" },
            border: { display: false },
            ticks: { color: "#9ca3af", font: { size: 11, family: "Lexend", weight: "500" }, callback: v => `${v}%` },
          },
          y: {
            grid: { display: false },
            border: { display: false },
            ticks: { color: "#6b7280", font: { size: 12, family: "Lexend", weight: "500" } },
          },
        },
        plugins: { legend: { display: false } },
        animation: { duration: 800, easing: "easeOutQuart" },
      },
    });
  }, 150);
}

function renderStats(data) {
  setTimeout(() => {
    animateCount("meta-words", 0, data.word_count, 700, v => Math.round(v).toLocaleString());
    document.getElementById("meta-chunks").textContent = data.chunks_analyzed;
    document.getElementById("meta-ai").textContent = `${data.ai_score}%`;
    document.getElementById("meta-heuristic").textContent = `${data.heuristic_score}%`;
  }, 100);
}

function renderSentenceDetails(sentences) {
  const list = document.getElementById("sentence-list");
  list.innerHTML = "";

  if (!sentences.length) {
    list.innerHTML = `<p class="empty-state">No sentence-level details were available for this sample.</p>`;
    return;
  }

  sentences.forEach(sentence => {
    const level = sentence.ai_score >= 70 ? "high" : sentence.ai_score >= 55 ? "medium" : "low";
    const item = document.createElement("div");
    item.className = `sentence-item ${level}`;
    item.innerHTML = `
      <div class="sentence-score">${sentence.ai_score}% AI</div>
      <p class="sentence-text">${escapeHtml(sentence.text)}</p>
    `;
    list.appendChild(item);
  });
}

function renderPhraseHits(hits) {
  const list = document.getElementById("phrase-list");
  list.innerHTML = "";

  if (!hits.length) {
    list.innerHTML = `<p class="empty-state">No tracked phrase hits were detected in this sample.</p>`;
    return;
  }

  hits.forEach(hit => {
    const item = document.createElement("div");
    item.className = "phrase-item";
    item.innerHTML = `
      <div>
        <div class="phrase-label">${escapeHtml(hit.label)}</div>
        <div class="phrase-meta">${hit.count} hit${hit.count !== 1 ? "s" : ""} · weight ${hit.weight}</div>
      </div>
      <div class="phrase-value">+${hit.contribution}</div>
    `;
    list.appendChild(item);
  });
}

function renderPreview(data) {
  const previewCard = document.getElementById("preview-card");
  if (data.scraped_preview) {
    document.getElementById("preview-text").textContent = data.scraped_preview;
    document.getElementById("preview-link").href = data.scraped_url;
    previewCard.classList.remove("hidden");
    return;
  }
  previewCard.classList.add("hidden");
}

function resetAll() {
  document.getElementById("results").classList.add("hidden");
  document.getElementById("compare-results").classList.add("hidden");
  document.getElementById("input-text").value = "";
  document.getElementById("input-url").value = "";
  document.getElementById("compare-left").value = "";
  document.getElementById("compare-right").value = "";
  document.getElementById("word-count").textContent = "0 words";
  fileInput.value = "";
  uploadFileLabel.textContent = "No file selected";
  hideError();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function saveHistoryEntry(payload, data) {
  const history = getHistory();
  const sourceLabel = payload.url
    ? payload.url
    : payload.uploaded_filename || payload.text?.slice(0, 72) || data.uploaded_filename || "Analysis";

  history.unshift({
    label: sourceLabel,
    verdict: data.verdict,
    ai_score: data.ai_score,
    timestamp: new Date().toISOString(),
  });

  localStorage.setItem(STORAGE_HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY_ITEMS)));
  renderHistory();
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function clearHistory() {
  localStorage.removeItem(STORAGE_HISTORY_KEY);
  renderHistory();
}

function renderHistory() {
  const history = getHistory();
  historyList.innerHTML = "";
  historyCard.classList.toggle("hidden", history.length === 0);

  history.forEach(item => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `
      <div class="history-copy">
        <div class="history-title">${escapeHtml(item.label)}</div>
        <div class="history-meta">${item.verdict}</div>
      </div>
      <div class="history-score">${item.ai_score}% AI</div>
    `;
    historyList.appendChild(row);
  });
}

function animateBar(fillId, valId, targetPct) {
  const fill = document.getElementById(fillId);
  const val = document.getElementById(valId);
  fill.style.width = "0%";
  val.textContent = "0%";
  requestAnimationFrame(() => {
    fill.style.width = `${targetPct}%`;
  });
  animateCount(valId, 0, targetPct, 900, v => `${v.toFixed(1)}%`);
}

function animateCount(elId, from, to, duration, format = v => v) {
  const el = document.getElementById(elId);
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = format(from + (to - from) * eased);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = format(to);
  }
  requestAnimationFrame(step);
}

function setLoading(loading) {
  btnSpinner.classList.toggle("hidden", !loading);
  analyzeBtn.disabled = loading;
  if (loading) {
    btnText.textContent = "Analyzing…";
    return;
  }
  const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
  btnText.textContent = activeTab === "compare"
    ? "Compare"
    : activeTab === "url"
      ? "Scrape & Analyze"
      : activeTab === "upload"
        ? "Analyze File"
        : "Analyze";
}

function showError(message) {
  const box = document.getElementById("error-box");
  document.getElementById("error-msg-text").textContent = message;
  box.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-box").classList.add("hidden");
}

function getWordCount(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
