const I18N = {
  en: {
    tag: "RAG · citation-level",
    headline: "Chilean crypto regulation, answered with sources.",
    lede:
      "Ask about fintech law, AML reporting or crypto taxes. Every answer cites the official document and article it comes from — or tells you it found nothing.",
    label: "Your question",
    placeholder: "¿Pago impuestos si vendo bitcoin?",
    ask: "Ask",
    answer: "Answer",
    sources: "Sources",
    sourcesHint: "Only sources cited in the answer",
    refusal: "No passage in the corpus supports an answer. Try rephrasing the question.",
    disclaimer:
      "Informational only — not legal or tax advice. Sources: BCN, CMF, UAF and SII.",
    fine: "Answers are written in Spanish because the corpus is in Spanish.",
    loading: "Searching the official sources…",
    searchFailed: "Something went wrong while searching.",
    noKey:
      "The server has no LLM key configured. Start it with LLM_PROVIDER=extractive for a no-LLM demo.",
    openSource: "Open official source ↗",
    ms: "ms",
  },
  es: {
    tag: "RAG · nivel de cita",
    headline: "Normativa cripto chilena, respondida con fuentes.",
    lede:
      "Pregunta por la ley fintech, reportes de lavado de activos o impuestos sobre cripto. Cada respuesta cita el documento y artículo oficial del que sale — o te dice que no encontró respaldo.",
    label: "Tu pregunta",
    placeholder: "¿Pago impuestos si vendo bitcoin?",
    ask: "Preguntar",
    answer: "Respuesta",
    sources: "Fuentes",
    sourcesHint: "Solo fuentes citadas en la respuesta",
    refusal:
      "Ningún pasaje del corpus respalda una respuesta. Prueba reformulando la pregunta.",
    disclaimer:
      "Solo informativo — no es asesoría legal ni tributaria. Fuentes: BCN, CMF, UAF y SII.",
    fine: "Las respuestas se escriben en español porque el corpus está en español.",
    loading: "Buscando en las fuentes oficiales…",
    searchFailed: "Algo falló durante la búsqueda.",
    noKey:
      "El servidor no tiene una clave de LLM configurada. Inícialo con LLM_PROVIDER=extractive para una demo sin LLM.",
    openSource: "Abrir fuente oficial ↗",
    ms: "ms",
  },
};

const state = { lang: localStorage.getItem("criptolex-lang") || "en" };

const el = {
  form: document.getElementById("ask-form"),
  input: document.getElementById("question"),
  submit: document.getElementById("submit"),
  result: document.getElementById("result"),
  answer: document.getElementById("answer"),
  meta: document.getElementById("meta"),
  sources: document.getElementById("sources"),
  sourcesWrap: document.getElementById("sources-wrap"),
  refusal: document.getElementById("refusal-note"),
  error: document.getElementById("error"),
};

function t(key) {
  return I18N[state.lang][key] ?? key;
}

function applyLang() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-lang]").forEach((node) => {
    node.classList.toggle("active", node.dataset.lang === state.lang);
  });
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderAnswer(text) {
  const escaped = escapeHtml(text);
  const withCites = escaped.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, group) =>
    group
      .split(",")
      .map((value) => {
        const n = value.trim();
        return `<button type="button" class="cite" data-cite="${n}" aria-label="source ${n}">${n}</button>`;
      })
      .join(""),
  );
  el.answer.innerHTML = withCites;
  el.answer.querySelectorAll("[data-cite]").forEach((node) => {
    node.addEventListener("click", () => focusSource(node.dataset.cite));
  });
}

function focusSource(n) {
  const card = document.getElementById(`source-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("highlight");
  window.setTimeout(() => card.classList.remove("highlight"), 1600);
}

function renderSources(sources) {
  el.sources.innerHTML = "";
  if (!sources.length) {
    el.sourcesWrap.hidden = true;
    return;
  }
  el.sourcesWrap.hidden = false;
  for (const source of sources) {
    const item = document.createElement("li");
    item.className = "source";
    item.id = `source-${source.n}`;
    const heading = source.article
      ? `<span class="source-title">${escapeHtml(source.title)}</span>
         <span class="source-article">${escapeHtml(source.article)}</span>`
      : `<span class="source-title">${escapeHtml(source.title)}</span>`;
    item.innerHTML = `
      <div class="source-top">
        <span class="source-n">[${source.n}]</span>
        ${heading}
      </div>
      <p class="source-snippet">${escapeHtml(source.snippet)}</p>
      <a class="source-link" href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">
        ${escapeHtml(source.issuer)} · ${source.year} — ${t("openSource")}
      </a>
    `;
    el.sources.appendChild(item);
  }
}

function showError(message) {
  el.error.textContent = message;
  el.error.hidden = false;
}

function clearError() {
  el.error.hidden = true;
  el.error.textContent = "";
}

async function ask(question) {
  clearError();
  el.submit.disabled = true;
  el.result.hidden = false;
  el.refusal.hidden = true;
  el.meta.textContent = "";
  el.answer.textContent = t("loading");
  el.sourcesWrap.hidden = true;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 5 }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail || t("searchFailed");
      showError(response.status === 503 ? t("noKey") : detail);
      el.result.hidden = true;
      return;
    }

    renderAnswer(data.answer);
    renderSources(data.sources || []);
    el.refusal.hidden = !data.refused;
    el.meta.textContent = `${data.latency_ms} ${t("ms")} · ${data.model}`;
  } catch (error) {
    console.error(error);
    showError(t("searchFailed"));
    el.result.hidden = true;
  } finally {
    el.submit.disabled = false;
  }
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el.input.value.trim();
  if (question.length >= 3) ask(question);
});

document.querySelectorAll("[data-question]").forEach((node) => {
  node.addEventListener("click", () => {
    el.input.value = node.dataset.question;
    ask(node.dataset.question);
  });
});

document.querySelectorAll("[data-lang]").forEach((node) => {
  node.addEventListener("click", () => {
    state.lang = node.dataset.lang;
    localStorage.setItem("criptolex-lang", state.lang);
    applyLang();
    if (!el.sourcesWrap.hidden) {
      // Re-render source links so their labels follow the selected language.
      const cards = [...el.sources.querySelectorAll(".source-link")];
      cards.forEach((card) => {
        const text = card.textContent.trim();
        const dash = text.lastIndexOf("—");
        if (dash !== -1) card.textContent = `${text.slice(0, dash).trim()} — ${t("openSource")}`;
      });
    }
  });
});

applyLang();
