let doughnutChart = null;
let barChart = null;

const STORAGE_THEME_KEY = "ai-slop-detector-theme";
const body = document.body;
const themeBtn = document.getElementById("theme-btn");
const infoBtn = document.getElementById("info-btn");
const modalOverlay = document.getElementById("modal-overlay");
const modal = document.getElementById("modal");
const modalClose = document.getElementById("modal-close");
const modalCloseBtn = document.getElementById("modal-close-btn");

initializeTheme();
bindIconControls();

// ── Tab switching ──────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
    document.getElementById("btn-text").textContent =
      tab.dataset.tab === "url" ? "Scrape & Analyze" : "Analyze";
  });
});

// ── Word count ─────────────────────────────────────────────────
document.getElementById("input-text").addEventListener("input", function () {
  const words = this.value.trim() ? this.value.trim().split(/\s+/).length : 0;
  document.getElementById("word-count").textContent = `${words} word${words !== 1 ? "s" : ""}`;
});

// ── Analyze ────────────────────────────────────────────────────
document.getElementById("analyze-btn").addEventListener("click", async () => {
  const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
  const text = document.getElementById("input-text").value.trim();
  const url = document.getElementById("input-url").value.trim();

  hideError();
  setLoading(true);

  const body = activeTab === "text" ? { text } : { url };

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }
    showResults(data);
  } catch {
    showError("Could not reach the server. Make sure Flask is running.");
  } finally {
    setLoading(false);
  }
});

// ── Reset ──────────────────────────────────────────────────────
document.getElementById("reset-btn").addEventListener("click", () => {
  const resultsEl = document.getElementById("results");
  resultsEl.style.opacity = "0";
  resultsEl.style.transform = "translateY(6px)";
  resultsEl.style.transition = "opacity 0.2s ease, transform 0.2s ease";
  setTimeout(() => {
    resultsEl.classList.add("hidden");
    resultsEl.style.cssText = "";
    document.getElementById("input-text").value = "";
    document.getElementById("input-url").value = "";
    document.getElementById("word-count").textContent = "0 words";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, 210);
  hideError();
});

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

// ── Show results ───────────────────────────────────────────────
function showResults(data) {
  const isAI = data.verdict === "AI-Generated";
  const dominantScore = isAI ? data.ai_score : data.human_score;

  // Verdict card
  const card = document.getElementById("verdict-card");
  card.classList.remove("is-ai", "is-human");
  card.classList.add(isAI ? "is-ai" : "is-human");

  const scoreEl = document.getElementById("verdict-score");
  scoreEl.className = `verdict-score ${isAI ? "is-ai" : "is-human"}`;
  scoreEl.textContent = "0%";
  animateCount("verdict-score", 0, dominantScore, 700, v => `${Math.round(v)}%`);

  document.getElementById("verdict-label").textContent =
    `${data.verdict} · ${data.confidence_label}`;

  const badge = document.getElementById("verdict-badge");
  badge.className = `verdict-badge ${isAI ? "is-ai" : "is-human"}`;
  badge.textContent = `${data.confidence}% confidence`;

  document.getElementById("verdict-desc").textContent =
    `${data.word_count.toLocaleString()} words analyzed across ${data.chunks_analyzed} chunk${data.chunks_analyzed !== 1 ? "s" : ""}.`;

  // Score bars
  setTimeout(() => {
    animateBar("bar-ai",    "bar-ai-val",    data.ai_score,    "score-val-red");
    animateBar("bar-human", "bar-human-val", data.human_score, "score-val-green");
  }, 100);

  // Doughnut chart
  setTimeout(() => {
    const dCtx = document.getElementById("doughnut-chart").getContext("2d");
    if (doughnutChart) doughnutChart.destroy();
    doughnutChart = new Chart(dCtx, {
      type: "doughnut",
      data: {
        labels: ["AI", "Human"],
        datasets: [{
          data: [data.ai_score, data.human_score],
          backgroundColor: ["rgba(239,68,68,0.8)", "rgba(34,197,94,0.8)"],
          borderColor: ["#ef4444", "#22c55e"],
          borderWidth: 0,
          hoverOffset: 4,
        }],
      },
      options: {
        cutout: "72%",
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}%` } },
        },
        animation: { animateRotate: true, duration: 800, easing: "easeOutQuart" },
      },
    });

    const dCenter = document.getElementById("doughnut-center");
    dCenter.textContent = `${dominantScore}%`;
    dCenter.style.color = isAI ? "#ef4444" : "#22c55e";
    dCenter.classList.remove("show");
    setTimeout(() => dCenter.classList.add("show"), 600);

    // Bar chart
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
            min: 0, max: 100,
            grid: { color: "#f3f4f6" },
            border: { display: false },
            ticks: { color: "#9ca3af", font: { size: 11, family: "Lexend", weight: "500" }, callback: v => `${v}%` },
          },
          y: {
            grid: { display: false },
            border: { display: false },
            ticks: { color: "#6b7280", font: { size: 12, family: "Lexend", weight: "500" } },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.x}%` } },
        },
        animation: { duration: 800, easing: "easeOutQuart" },
      },
    });
  }, 150);

  // Stats
  setTimeout(() => {
    animateCount("meta-words", 0, data.word_count, 700, v => Math.round(v).toLocaleString());
    document.getElementById("meta-chunks").textContent = data.chunks_analyzed;
    document.getElementById("meta-ai").textContent = `${data.ai_score}%`;
    document.getElementById("meta-heuristic").textContent = `${data.heuristic_score}%`;
  }, 100);

  // Preview
  if (data.scraped_preview) {
    document.getElementById("preview-text").textContent = data.scraped_preview;
    document.getElementById("preview-link").href = data.scraped_url;
    document.getElementById("preview-card").classList.remove("hidden");
  } else {
    document.getElementById("preview-card").classList.add("hidden");
  }

  // Show results + scroll
  const resultsEl = document.getElementById("results");
  resultsEl.classList.remove("hidden");
  setTimeout(() => {
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 50);
}

// ── Helpers ────────────────────────────────────────────────────
function animateBar(fillId, valId, targetPct, valClass) {
  const fill = document.getElementById(fillId);
  const val  = document.getElementById(valId);
  fill.style.width = "0%";
  val.textContent = "0%";
  requestAnimationFrame(() => { fill.style.width = `${targetPct}%`; });
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
  const btn = document.getElementById("analyze-btn");
  const activeTab = document.querySelector(".tab-btn.active").dataset.tab;
  document.getElementById("btn-text").textContent = loading
    ? "Analyzing…"
    : (activeTab === "url" ? "Scrape & Analyze" : "Analyze");
  document.getElementById("btn-spinner").classList.toggle("hidden", !loading);
  btn.disabled = loading;
}

function showError(msg) {
  const box = document.getElementById("error-box");
  document.getElementById("error-msg-text").textContent = msg;
  box.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-box").classList.add("hidden");
}
