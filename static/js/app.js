/* ═══════════════════════════════════════════════════════════════════════════
   Verse Intelligence — Frontend Application
   Supports: MarianMT (European) · mBART-50 (Hindi) · NLLB-200 (Marathi)
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── Sample Poems ─────────────────────────────────────────────────────────────

const SAMPLES = {
  // English poems
  frost: `Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;`,

  keats: `A thing of beauty is a joy for ever:
Its loveliness increases; it will never
Pass into nothingness; but still will keep
A bower quiet for us, and a sleep
Full of sweet dreams, and health, and quiet breathing.`,

  tagore: `Where the mind is without fear and the head is held high
Where knowledge is free
Where the world has not been broken up into fragments
By narrow domestic walls
Where words come out from the depth of truth`,

  // Hindi poem — Kabir Doha
  kabir_hi: `माटी कहे कुम्हार से, तू क्या रौंदे मोय।
एक दिन ऐसा आएगा, मैं रौंदूंगी तोय॥
बड़ा भया तो क्या भया, जैसे पेड़ खजूर।
पंथी को छाया नहीं, फल लागे अति दूर॥
साधु ऐसा चाहिए, जैसा सूप सुभाय।
सार-सार को गहि रहे, थोथा देई उड़ाय॥`,

  // Marathi poem — Tukaram Abhang
  tukaram_mr: `आम्ही जातो आपुल्या गावा।
आमचा राम राम घ्यावा॥
सुख दुःख भोगिले येथे।
आता जड जाहलो तेथे॥
तुका म्हणे आता पुढे।
देव देईल त्याचे जोडे॥`,
};

// ── Engine mapping per language pair ─────────────────────────────────────────

const ENGINE_MAP = {
  "en-hi": "mBART-50",  "hi-en": "mBART-50",
  "en-mr": "NLLB-200",  "mr-en": "NLLB-200",
  "hi-mr": "NLLB-200",  "mr-hi": "NLLB-200",
  "en-fr": "MarianMT",  "fr-en": "MarianMT",
  "en-de": "MarianMT",  "de-en": "MarianMT",
  "en-es": "MarianMT",  "es-en": "MarianMT",
  "en-it": "MarianMT",  "it-en": "MarianMT",
  "en-ro": "MarianMT",
  "en-zh": "MarianMT",
};

const INDIC_PAIRS = new Set(["en-hi","hi-en","en-mr","mr-en","hi-mr","mr-hi"]);

// ── DOM ───────────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const els = {
  poemInput:           $("poemInput"),
  langPair:            $("langPair"),
  engineBadge:         $("engineBadge"),
  modelWarning:        $("modelWarning"),
  apiKey:              $("apiKey"),
  apiKeyGroup:         $("apiKeyGroup"),
  useRag:              $("useRag"),
  translateBtn:        $("translateBtn"),
  outputPlaceholder:   $("outputPlaceholder"),
  results:             $("results"),
  rawTranslation:      $("rawTranslation"),
  enhancedTranslation: $("enhancedTranslation"),
  metricsSection:      $("metrics"),
  pipelineSection:     $("pipeline"),
  loadingOverlay:      $("loadingOverlay"),
};

// ── State ─────────────────────────────────────────────────────────────────────

let currentMode  = "literal";
let chartOrig    = null;
let chartTrans   = null;

// ── Language Pair → engine badge + warning ────────────────────────────────────

function onLangPairChange() {
  const pair   = els.langPair.value;
  const engine = ENGINE_MAP[pair] || "MarianMT";

  els.engineBadge.textContent = engine;

  // Show download warning for large Indic models
  if (INDIC_PAIRS.has(pair)) {
    els.modelWarning.classList.add("show");
  } else {
    els.modelWarning.classList.remove("show");
  }

  // Auto-load a matching sample poem hint
  updatePlaceholder(pair);
}

function updatePlaceholder(pair) {
  const hints = {
    "en-hi": "Enter an English poem → will be translated to Hindi…",
    "en-mr": "Enter an English poem → will be translated to Marathi…",
    "hi-en": "हिंदी कविता यहाँ लिखें → English में अनुवाद होगा…",
    "mr-en": "मराठी कविता इथे लिहा → English मध्ये भाषांतर होईल…",
    "hi-mr": "हिंदी कविता यहाँ लिखें → Marathi मध्ये भाषांतर होईल…",
    "mr-hi": "मराठी कविता इथे लिहा → Hindi में अनुवाद होगा…",
  };
  if (hints[pair]) {
    els.poemInput.placeholder = hints[pair];
  } else {
    els.poemInput.placeholder = "Enter your poem here…";
  }
}

els.langPair.addEventListener("change", onLangPairChange);
onLangPairChange(); // run on load

// ── Mode Toggle ───────────────────────────────────────────────────────────────

document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    els.apiKeyGroup.style.display = currentMode === "context_aware" ? "block" : "none";
  });
});

// ── Tab Toggle ────────────────────────────────────────────────────────────────

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    $("tabRaw").style.display      = btn.dataset.tab === "raw"      ? "block" : "none";
    $("tabEnhanced").style.display = btn.dataset.tab === "enhanced" ? "block" : "none";
  });
});

// ── Sample Poems ──────────────────────────────────────────────────────────────

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const poem = SAMPLES[chip.dataset.poem];
    if (!poem) return;
    els.poemInput.value = poem;

    // Auto-select sensible language pair for Indic samples
    const autoMap = {
      kabir_hi:   "hi-en",
      tukaram_mr: "mr-en",
    };
    const auto = autoMap[chip.dataset.poem];
    if (auto) {
      els.langPair.value = auto;
      onLangPairChange();
    }
  });
});

// ── Loading ───────────────────────────────────────────────────────────────────

const STAGES = ["ls1","ls2","ls3","ls4","ls5"];

function showLoading(isIndic) {
  STAGES.forEach(id => $(id).classList.remove("active","done"));

  // Update stage labels for Indic
  if (isIndic) {
    $("ls2").textContent = "② Neural Translation (mBART-50 / NLLB-200)";
  } else {
    $("ls2").textContent = "② Neural Translation (MarianMT)";
  }

  els.loadingOverlay.style.display = "flex";
  els.translateBtn.disabled = true;

  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) $(STAGES[i-1]).classList.replace("active","done");
    if (i < STAGES.length) { $(STAGES[i]).classList.add("active"); i++; }
    else clearInterval(interval);
  }, isIndic ? 1400 : 900);   // Indic models are slower
  return interval;
}

function hideLoading(interval) {
  clearInterval(interval);
  STAGES.forEach(id => $(id).classList.add("done"));
  setTimeout(() => {
    els.loadingOverlay.style.display = "none";
    els.translateBtn.disabled = false;
  }, 300);
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function showToast(msg) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 5000);
}

// ── Metric Animation ──────────────────────────────────────────────────────────

function animateMetric(valueEl, fillEl, value) {
  valueEl.textContent = (value * 100).toFixed(1) + "%";
  requestAnimationFrame(() => {
    fillEl.style.width = Math.min(value * 100, 100) + "%";
  });
}

// ── Sentiment Donut ───────────────────────────────────────────────────────────

function drawDonut(canvasId, data, existing) {
  const canvas = $(canvasId);
  if (!canvas || typeof Chart === "undefined") return null;
  if (existing) existing.destroy();
  const COLORS = ["#c9a84c","#8b2e2e","#4c7a8b","#6b8b4c","#8b4c7a","#4c6b8b","#8b6b4c","#4c8b6b"];
  return new Chart(canvas.getContext("2d"), {
    type: "doughnut",
    data: {
      labels: Object.keys(data),
      datasets: [{
        data: Object.values(data),
        backgroundColor: COLORS.slice(0, Object.keys(data).length),
        borderColor: "#1a1a1a",
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#6b6457", font: { family: "JetBrains Mono", size: 10 }, padding: 8, boxWidth: 10 },
        },
      },
      cutout: "65%",
    },
  });
}

// ── Render Results ────────────────────────────────────────────────────────────

function renderResults(data) {
  // Translations
  els.rawTranslation.textContent      = data.raw_translation       || "(no translation)";
  els.enhancedTranslation.textContent = data.enhanced_translation  || data.raw_translation;

  els.outputPlaceholder.style.display = "none";
  els.results.style.display           = "block";

  // Metrics
  if (data.metrics) {
    const m = data.metrics;
    animateMetric($("cosineVal"), $("cosineFill"), m.cosine_similarity   || 0);
    animateMetric($("bleuVal"),   $("bleuFill"),   m.bleu_score          || 0);
    animateMetric($("bertVal"),   $("bertFill"),   m.bert_score?.f1      || 0);
    $("driftVal").textContent    = (m.semantic_drift || 0).toFixed(3);
    $("driftFill").style.width   = Math.min((m.semantic_drift / 5) * 100, 100) + "%";
    els.metricsSection.style.display = "block";
  }

  // Sentiment
  if (data.sentiment) {
    const { original: orig, translated: trans, drift } = data.sentiment;
    $("sentOrigEmotion").textContent  = orig.label  || "—";
    $("sentTransEmotion").textContent = trans.label || "—";
    $("sentOrigScore").textContent    = "compound: " + (orig.compound  ?? "—");
    $("sentTransScore").textContent   = "compound: " + (trans.compound ?? "—");

    const dNum = drift?.compound_drift;
    $("driftBadge").textContent = dNum !== undefined ? dNum.toFixed(3) : "—";
    $("driftBadge").style.color = (dNum < 0.15) ? "#7ec87e" : "#e09090";

    const origEmotions  = orig.emotions  && Object.keys(orig.emotions).length  ? orig.emotions  : { [orig.label  || "neutral"]: 1 };
    const transEmotions = trans.emotions && Object.keys(trans.emotions).length ? trans.emotions : { [trans.label || "neutral"]: 1 };

    chartOrig  = drawDonut("chartOrig",  origEmotions,  chartOrig);
    chartTrans = drawDonut("chartTrans", transEmotions, chartTrans);
  }

  // Preprocessing
  if (data.preprocessing) {
    const p = data.preprocessing;
    if (p.tokens?.length)
      $("stageTokens").innerHTML = p.tokens.map(t => `<span class="tag">${esc(t)}</span>`).join(" ");
    if (p.pos_tags?.length)
      $("stagePOS").innerHTML    = p.pos_tags.map(pt => `<span class="tag pos">${esc(pt.word)} <em>${pt.tag}</em></span>`).join(" ");
    if (p.lemmas?.length)
      $("stageLemmas").innerHTML = p.lemmas.map(l => `<span class="tag">${esc(l)}</span>`).join(" ");
    $("stageNER").innerHTML = p.named_entities?.length
      ? p.named_entities.map(e => `<span class="tag ner">${esc(e.text)} [${e.label}]</span>`).join(" ")
      : "No named entities detected.";
    els.pipelineSection.style.display = "block";
  }

  // RAG context display
  if (data.rag_context) {
    const ragDiv = $("ragContext");
    if (ragDiv) { ragDiv.textContent = data.rag_context; ragDiv.parentElement.style.display = "block"; }
  }

  setTimeout(() => els.results.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
}

function esc(str) {
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Translate ─────────────────────────────────────────────────────────────────

async function translate() {
  const poem = els.poemInput.value.trim();
  if (!poem) { showToast("Please enter a poem first."); return; }

  const pair    = els.langPair.value;             // e.g. "en-hi"
  const [src, tgt] = pair.split("-");
  const isIndic = INDIC_PAIRS.has(pair);

  const payload = {
    poem,
    src_lang:    src,
    tgt_lang:    tgt,
    mode:        currentMode,
    llm_api_key: els.apiKey?.value || "",
    use_rag:     els.useRag.checked,
  };

  const interval = showLoading(isIndic);

  try {
    const response = await fetch("/api/translate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Translation failed");
    hideLoading(interval);
    renderResults(data);
  } catch (err) {
    hideLoading(interval);
    showToast("Error: " + err.message);
    console.error(err);
  }
}

// ── Event Listeners ───────────────────────────────────────────────────────────

els.translateBtn.addEventListener("click", translate);
els.poemInput.addEventListener("keydown", e => { if (e.ctrlKey && e.key === "Enter") translate(); });

// ── Nav links ─────────────────────────────────────────────────────────────────

document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
    link.classList.add("active");
  });
});

// ── Load Chart.js ─────────────────────────────────────────────────────────────

(function () {
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js";
  s.onload = () => console.log("Chart.js ready");
  document.head.appendChild(s);
})();

// ── Google Noto Sans Devanagari for proper Hindi/Marathi rendering ────────────

(function () {
  const link = document.createElement("link");
  link.rel  = "stylesheet";
  link.href = "https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@300;400;500&display=swap";
  document.head.appendChild(link);
})();

// ── Health check ──────────────────────────────────────────────────────────────

window.addEventListener("load", async () => {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    if (d.status !== "ok") throw new Error();
    console.log("✓ Poetry NMT API online");
  } catch {
    showToast("Warning: API may be offline. Start Flask with: python app.py");
  }
});
