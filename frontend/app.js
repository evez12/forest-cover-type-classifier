"use strict";

/* =========================================================================
   Configuration
   ========================================================================= */
// Same origin when served by FastAPI; fallback for opening index.html from disk.
const API_BASE = window.location.protocol.startsWith("http")
  ? window.location.origin
  : "http://127.0.0.1:8000";

// Continuous features — names match the notebook / API exactly.
// min/max = dataset range (df.describe()), value = training median.
// hardMin/hardMax = API validation bounds (typed values may exceed the slider range).
const NUMERIC_GROUPS = [
  {
    title: "Topography",
    features: [
      { name: "Elevation", label: "Elevation", unit: "m", min: 1859, max: 3858, value: 2996, hardMin: 0, hardMax: 9000 },
      { name: "Aspect", label: "Aspect", unit: "° azimuth", min: 0, max: 360, value: 127, hardMin: 0, hardMax: 360 },
      { name: "Slope", label: "Slope", unit: "°", min: 0, max: 66, value: 13, hardMin: 0, hardMax: 90 },
    ],
  },
  {
    title: "Hydrology",
    features: [
      { name: "Horizontal_Distance_To_Hydrology", label: "Horizontal distance to water", unit: "m", min: 0, max: 1397, value: 218, hardMin: 0, hardMax: 100000 },
      { name: "Vertical_Distance_To_Hydrology", label: "Vertical distance to water", unit: "m", min: -173, max: 601, value: 30, hardMin: -2000, hardMax: 2000 },
    ],
  },
  {
    title: "Roads & wildfire",
    features: [
      { name: "Horizontal_Distance_To_Roadways", label: "Distance to roadways", unit: "m", min: 0, max: 7117, value: 1997, hardMin: 0, hardMax: 100000 },
      { name: "Horizontal_Distance_To_Fire_Points", label: "Distance to fire ignition points", unit: "m", min: 0, max: 7173, value: 1710, hardMin: 0, hardMax: 100000 },
    ],
  },
  {
    title: "Hillshade · summer solstice",
    features: [
      { name: "Hillshade_9am", label: "Hillshade 9am", unit: "index 0–255", min: 0, max: 254, value: 218, hardMin: 0, hardMax: 255 },
      { name: "Hillshade_Noon", label: "Hillshade noon", unit: "index 0–255", min: 0, max: 254, value: 226, hardMin: 0, hardMax: 255 },
      { name: "Hillshade_3pm", label: "Hillshade 3pm", unit: "index 0–255", min: 0, max: 254, value: 143, hardMin: 0, hardMax: 255 },
    ],
  },
];
const NUMERIC_FEATURES = NUMERIC_GROUPS.flatMap((g) => g.features);

const WILDERNESS_AREAS = ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"];

const SOIL_TYPES = [
  "2702 · Cathedral family – Rock outcrop complex, extremely stony",
  "2703 · Vanet – Ratake families complex, very stony",
  "2704 · Haploborolis – Rock outcrop complex, rubbly",
  "2705 · Ratake family – Rock outcrop complex, rubbly",
  "2706 · Vanet family – Rock outcrop complex, rubbly",
  "2717 · Vanet – Wetmore families – Rock outcrop complex, stony",
  "3501 · Gothic family",
  "3502 · Supervisor – Limber families complex",
  "4201 · Troutville family, very stony",
  "4703 · Bullwark – Catamount families – Rock outcrop complex, rubbly",
  "4704 · Bullwark – Catamount families – Rock land complex, rubbly",
  "4744 · Legault family – Rock land complex, stony",
  "4758 · Catamount family – Rock land – Bullwark family complex, rubbly",
  "5101 · Pachic Argiborolis – Aquolis complex",
  "5151 · Unspecified in the USFS Soil and ELU Survey",
  "6101 · Cryaquolis – Cryoborolis complex",
  "6102 · Gateview family – Cryaquolis complex",
  "6731 · Rogert family, very stony",
  "7101 · Typic Cryaquolis – Borohemists complex",
  "7102 · Typic Cryaquepts – Typic Cryaquolls complex",
  "7103 · Typic Cryaquolls – Leighcan family, till substratum complex",
  "7201 · Leighcan family, till substratum, extremely bouldery",
  "7202 · Leighcan family, till substratum – Typic Cryaquolls complex",
  "7700 · Leighcan family, extremely stony",
  "7701 · Leighcan family, warm, extremely stony",
  "7702 · Granile – Catamount families complex, very stony",
  "7709 · Leighcan family, warm – Rock outcrop complex, extremely stony",
  "7710 · Leighcan family – Rock outcrop complex, extremely stony",
  "7745 · Como – Legault families complex, extremely stony",
  "7746 · Como family – Rock land – Legault family complex, extremely stony",
  "7755 · Leighcan – Catamount families complex, extremely stony",
  "7756 · Catamount family – Rock outcrop – Leighcan family complex, extremely stony",
  "7757 · Leighcan – Catamount families – Rock outcrop complex, extremely stony",
  "7790 · Cryorthents – Rock land complex, extremely stony",
  "8703 · Cryumbrepts – Rock outcrop – Cryaquepts complex",
  "8707 · Bross family – Rock land – Cryumbrepts complex, extremely stony",
  "8708 · Rock outcrop – Cryumbrepts – Cryorthents complex, extremely stony",
  "8771 · Leighcan – Moran families – Cryaquolls complex, extremely stony",
  "8772 · Moran family – Cryorthents – Leighcan family complex, extremely stony",
  "8776 · Moran family – Cryorthents – Rock land complex, extremely stony",
];

// Most frequent categories in the dataset
const DEFAULT_WILDERNESS = 0; // Rawah
const DEFAULT_SOIL = 28; // 7745 · Como – Legault

const FEATURE_ORDER = [
  ...NUMERIC_FEATURES.map((f) => f.name),
  ...WILDERNESS_AREAS.map((_, i) => `Wilderness_Area_${i}`),
  ...SOIL_TYPES.map((_, i) => `Soil_Type_${i}`),
];

const classColor = (idx) => `var(--class-${idx})`;
const pct = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/* =========================================================================
   DOM refs & state
   ========================================================================= */
const $ = (sel) => document.querySelector(sel);
const els = {
  groups: $("#numeric-groups"),
  wilderness: $("#wilderness-options"),
  soil: $("#soil-select"),
  form: $("#predict-form"),
  predictBtn: $("#predict-btn"),
  formError: $("#form-error"),
  resetBtn: $("#reset-btn"),
  liveToggle: $("#live-toggle"),
  samples: $("#samples"),
  samplesList: $("#samples-list"),
  statusPill: $("#status-pill"),
  statusText: $("#status-text"),
  docsLink: $("#docs-link"),
  themeToggle: $("#theme-toggle"),
  resultEmpty: $("#result-empty"),
  result: $("#result"),
  heroSwatch: $("#hero-swatch"),
  heroClass: $("#hero-class"),
  heroMeta: $("#hero-meta"),
  heroConf: $("#hero-conf"),
  heroMeter: $("#hero-meter"),
  probList: $("#prob-list"),
  warnings: $("#warnings"),
  latency: $("#latency"),
  rawRequest: $("#raw-request"),
  rawResponse: $("#raw-response"),
  modelStats: $("#model-stats"),
  statsList: $("#stats-list"),
  tooltip: $("#tooltip"),
};

const state = {
  defaults: Object.fromEntries(NUMERIC_FEATURES.map((f) => [f.name, f.value])),
  controller: null,
  liveTimer: null,
};

/* =========================================================================
   Form rendering
   ========================================================================= */
function renderNumericGroups() {
  const frag = document.createDocumentFragment();
  for (const group of NUMERIC_GROUPS) {
    const fs = document.createElement("fieldset");
    fs.className = "group";
    fs.innerHTML = `<legend class="group__title">${group.title}</legend><div class="grid"></div>`;
    const grid = fs.querySelector(".grid");

    for (const f of group.features) {
      const field = document.createElement("div");
      field.className = "field";
      field.innerHTML = `
        <div class="field__top">
          <label class="field__label" for="num-${f.name}">${f.label}</label>
          <span class="field__unit">${f.unit}</span>
        </div>
        <div class="field__row">
          <input class="slider" type="range" id="rng-${f.name}" data-feature="${f.name}"
                 min="${f.min}" max="${f.max}" step="1" value="${f.value}" aria-label="${f.label} slider" />
          <input class="num" type="number" id="num-${f.name}" name="${f.name}" data-feature="${f.name}"
                 min="${f.hardMin}" max="${f.hardMax}" step="any" value="${f.value}" inputmode="decimal" />
        </div>
        <div class="field__range"><span data-min>${f.min}</span><span data-max>${f.max}</span></div>`;
      grid.appendChild(field);
    }
    frag.appendChild(fs);
  }
  els.groups.appendChild(frag);

  els.groups.addEventListener("input", (e) => {
    const target = e.target;
    const name = target.dataset.feature;
    if (!name) return;
    if (target.type === "range") {
      setNumeric(name, Number(target.value));
    } else {
      validateNumber(target);
      syncSlider(name, Number(target.value));
    }
    scheduleLivePredict();
  });
}

function renderWilderness() {
  els.wilderness.innerHTML = WILDERNESS_AREAS.map(
    (name, i) => `
      <label class="radio-card">
        <input type="radio" name="wilderness" value="${i}" ${i === DEFAULT_WILDERNESS ? "checked" : ""} />
        <span class="radio-card__body">
          <span class="radio-card__name">${name}</span>
          <span class="radio-card__code">Wilderness_Area_${i}</span>
        </span>
      </label>`,
  ).join("");
  els.wilderness.addEventListener("change", scheduleLivePredict);
}

function renderSoil() {
  els.soil.innerHTML = SOIL_TYPES.map(
    (desc, i) => `<option value="${i}" ${i === DEFAULT_SOIL ? "selected" : ""}>Soil_Type_${i} — ${desc}</option>`,
  ).join("");
  els.soil.addEventListener("change", scheduleLivePredict);
}

/* =========================================================================
   Value helpers
   ========================================================================= */
function syncSlider(name, value) {
  const slider = document.getElementById(`rng-${name}`);
  if (!slider || Number.isNaN(value)) return;
  const min = Number(slider.min);
  const max = Number(slider.max);
  const clamped = Math.min(max, Math.max(min, value));
  slider.value = String(clamped);
  slider.style.setProperty("--pct", `${((clamped - min) / (max - min)) * 100}%`);
}

function setNumeric(name, value) {
  const input = document.getElementById(`num-${name}`);
  input.value = String(value);
  validateNumber(input);
  syncSlider(name, value);
}

function validateNumber(input) {
  const v = input.valueAsNumber;
  const ok = input.value !== "" && Number.isFinite(v) && v >= Number(input.min) && v <= Number(input.max);
  input.classList.toggle("is-invalid", !ok);
  return ok;
}

function setWilderness(idx) {
  const radio = els.wilderness.querySelector(`input[value="${idx}"]`);
  if (radio) radio.checked = true;
}

function resetForm() {
  for (const f of NUMERIC_FEATURES) setNumeric(f.name, state.defaults[f.name]);
  setWilderness(DEFAULT_WILDERNESS);
  els.soil.value = String(DEFAULT_SOIL);
  markActiveSample(null);
  hideError();
}

/** Build the exact 54-feature JSON payload expected by POST /predict. */
function collectPayload() {
  const invalid = [];
  const payload = {};
  for (const f of NUMERIC_FEATURES) {
    const input = document.getElementById(`num-${f.name}`);
    if (!validateNumber(input)) invalid.push(f.label);
    payload[f.name] = input.valueAsNumber;
  }
  const wildIdx = Number(els.wilderness.querySelector("input:checked")?.value ?? DEFAULT_WILDERNESS);
  const soilIdx = Number(els.soil.value);
  WILDERNESS_AREAS.forEach((_, i) => (payload[`Wilderness_Area_${i}`] = i === wildIdx ? 1 : 0));
  SOIL_TYPES.forEach((_, i) => (payload[`Soil_Type_${i}`] = i === soilIdx ? 1 : 0));

  if (invalid.length) {
    throw new Error(`Invalid value for: ${invalid.join(", ")}`);
  }
  return Object.fromEntries(FEATURE_ORDER.map((k) => [k, payload[k]]));
}

/* =========================================================================
   API
   ========================================================================= */
async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function formatApiError(status, body) {
  if (body && Array.isArray(body.detail)) {
    return body.detail
      .map((d) => `${(d.loc || []).filter((l) => l !== "body").join(".") || "input"}: ${d.msg}`)
      .join(" · ");
  }
  if (body && typeof body.detail === "string") return body.detail;
  return `Request failed with status ${status}`;
}

async function predict() {
  hideError();
  let payload;
  try {
    payload = collectPayload();
  } catch (err) {
    showError(err.message);
    return;
  }

  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  setLoading(true);
  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error(formatApiError(res.status, body));
    renderResult(body, payload, performance.now() - t0);
  } catch (err) {
    if (err.name === "AbortError") return;
    showError(err instanceof TypeError ? `Cannot reach the API at ${API_BASE}. Is the server running?` : err.message);
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      setLoading(false);
    }
  }
}

function scheduleLivePredict() {
  markActiveSample(null);
  if (!els.liveToggle.checked) return;
  clearTimeout(state.liveTimer);
  state.liveTimer = setTimeout(predict, 220);
}

/* =========================================================================
   Result rendering
   ========================================================================= */
function renderResult(data, payload, roundTripMs) {
  els.resultEmpty.hidden = true;
  els.result.hidden = false;

  const color = classColor(data.class_index);
  els.heroSwatch.style.background = color;
  els.heroMeter.style.background = color;
  els.heroClass.textContent = data.class_name;
  els.heroMeta.textContent = `Cover_Type ${data.cover_type} · class index ${data.class_index}`;
  els.heroConf.textContent = pct(data.confidence);
  requestAnimationFrame(() => (els.heroMeter.style.width = pct(data.confidence, 2)));

  els.probList.innerHTML = data.probabilities
    .map(
      (p, rank) => `
      <li class="prob ${rank === 0 ? "is-top" : ""}" data-index="${p.class_index}" data-cover="${p.cover_type}"
          data-name="${p.class_name}" data-prob="${p.probability}" tabindex="0"
          aria-label="${p.class_name}: ${pct(p.probability, 2)}">
        <span class="prob__name"><span class="prob__dot" style="background:${classColor(p.class_index)}"></span><span>${p.class_name}</span></span>
        <span class="prob__track"><span class="prob__bar" style="background:${classColor(p.class_index)}"></span></span>
        <span class="prob__value">${pct(p.probability)}</span>
      </li>`,
    )
    .join("");
  requestAnimationFrame(() => {
    els.probList.querySelectorAll(".prob").forEach((li) => {
      li.querySelector(".prob__bar").style.width = pct(Number(li.dataset.prob), 3);
    });
  });

  if (data.warnings?.length) {
    els.warnings.innerHTML = data.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
    els.warnings.hidden = false;
  } else {
    els.warnings.hidden = true;
  }

  const server = data.latency_ms != null ? ` · model ${data.latency_ms.toFixed(1)} ms` : "";
  els.latency.textContent = `${roundTripMs.toFixed(0)} ms round-trip${server}`;
  els.rawRequest.textContent = JSON.stringify(payload, null, 2);
  els.rawResponse.textContent = JSON.stringify(data, null, 2);

  if (window.matchMedia("(max-width: 960px)").matches && !els.liveToggle.checked) {
    els.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/* ---- Tooltip on probability rows ---- */
function initTooltip() {
  const show = (li, x, y) => {
    els.tooltip.innerHTML = `<strong>${escapeHtml(li.dataset.name)}</strong>
      Cover_Type ${li.dataset.cover} · index ${li.dataset.index}<br/>
      p = ${Number(li.dataset.prob).toFixed(6)}`;
    els.tooltip.hidden = false;
    const { width, height } = els.tooltip.getBoundingClientRect();
    const left = Math.min(window.innerWidth - width - 8, x + 14);
    const top = y - height - 12 < 8 ? y + 16 : y - height - 12;
    els.tooltip.style.left = `${Math.max(8, left)}px`;
    els.tooltip.style.top = `${top}px`;
  };
  const hide = () => (els.tooltip.hidden = true);

  els.probList.addEventListener("pointermove", (e) => {
    const li = e.target.closest(".prob");
    li ? show(li, e.clientX, e.clientY) : hide();
  });
  els.probList.addEventListener("pointerleave", hide);
  els.probList.addEventListener("focusin", (e) => {
    const li = e.target.closest(".prob");
    if (!li) return;
    const r = li.getBoundingClientRect();
    show(li, r.left + r.width / 2, r.top);
  });
  els.probList.addEventListener("focusout", hide);
}

/* =========================================================================
   UI state helpers
   ========================================================================= */
function setLoading(loading) {
  els.predictBtn.disabled = loading;
  els.predictBtn.classList.toggle("is-loading", loading);
}

function showError(msg) {
  els.formError.textContent = msg;
  els.formError.hidden = false;
}

function hideError() {
  els.formError.hidden = true;
}

function setStatus(kind, text) {
  els.statusPill.className = `pill pill--${kind}`;
  els.statusText.textContent = text;
  els.statusPill.title = text;
}

function markActiveSample(idx) {
  els.samplesList.querySelectorAll(".chip").forEach((c) => c.classList.toggle("is-active", c.dataset.index === String(idx)));
}

/* =========================================================================
   Backend metadata: health, model info (training medians/ranges), samples
   ========================================================================= */
async function loadHealth() {
  try {
    const h = await apiGet("/health");
    if (h.model_loaded) setStatus("ok", `Model ready · ${h.device}`);
    else setStatus("error", "Model not loaded");
    if (!h.model_loaded && h.detail) showError(h.detail);
    return h.model_loaded;
  } catch {
    setStatus("error", "API offline");
    showError(`Cannot reach the API at ${API_BASE}. Start it with: uvicorn backend.main:app --reload`);
    return false;
  }
}

async function loadModelInfo() {
  try {
    const info = await apiGet("/model-info");

    // Use the exact training-split medians / ranges exported by train_and_export.py
    for (const f of NUMERIC_FEATURES) {
      const s = info.feature_stats?.[f.name];
      if (!s) continue;
      const slider = document.getElementById(`rng-${f.name}`);
      slider.min = String(Math.floor(s.min));
      slider.max = String(Math.ceil(s.max));
      const range = slider.closest(".field").querySelector(".field__range");
      range.querySelector("[data-min]").textContent = String(Math.floor(s.min));
      range.querySelector("[data-max]").textContent = String(Math.ceil(s.max));
      state.defaults[f.name] = Math.round(s.median);
    }
    resetForm();

    const m = info.test_metrics || {};
    const stats = [
      ["Accuracy", m.accuracy],
      ["Macro F1", m.macro_f1],
      ["Balanced acc.", m.balanced_accuracy],
    ].filter(([, v]) => typeof v === "number");
    if (stats.length) {
      els.statsList.innerHTML = stats
        .map(([k, v]) => `<div class="stat"><dt>${k}</dt><dd>${pct(v)}</dd></div>`)
        .join("");
      els.modelStats.hidden = false;
    }
  } catch {
    /* keep hard-coded defaults */
  }
}

async function loadExamples() {
  try {
    const examples = await apiGet("/examples");
    if (!examples.length) return;
    els.samplesList.innerHTML = examples
      .map(
        (ex) => `<button type="button" class="chip" data-index="${ex.class_index}"
                   title="Representative test-set row with true label ${escapeHtml(ex.class_name)}">
                   <span class="chip__dot" style="background:${classColor(ex.class_index)}"></span>${escapeHtml(ex.class_name)}
                 </button>`,
      )
      .join("");
    els.samples.hidden = false;

    els.samplesList.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      const ex = examples.find((x) => String(x.class_index) === chip.dataset.index);
      applyFeatures(ex.features);
      markActiveSample(ex.class_index);
      predict();
    });
  } catch {
    /* samples are optional */
  }
}

function applyFeatures(features) {
  for (const f of NUMERIC_FEATURES) setNumeric(f.name, features[f.name]);
  const w = WILDERNESS_AREAS.findIndex((_, i) => features[`Wilderness_Area_${i}`] === 1);
  const s = SOIL_TYPES.findIndex((_, i) => features[`Soil_Type_${i}`] === 1);
  if (w >= 0) setWilderness(w);
  if (s >= 0) els.soil.value = String(s);
  hideError();
}

/* =========================================================================
   Theme
   ========================================================================= */
function initTheme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("covtype-theme");
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch {
    /* storage unavailable */
  }
  els.themeToggle.addEventListener("click", () => {
    const current = root.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("covtype-theme", next);
    } catch {
      /* storage unavailable */
    }
  });
}

/* =========================================================================
   Init
   ========================================================================= */
async function init() {
  initTheme();
  renderNumericGroups();
  renderWilderness();
  renderSoil();
  resetForm();
  initTooltip();

  els.docsLink.href = `${API_BASE}/docs`;
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    predict();
  });
  els.resetBtn.addEventListener("click", () => {
    resetForm();
    scheduleLivePredict();
  });
  els.liveToggle.addEventListener("change", () => els.liveToggle.checked && predict());

  const ready = await loadHealth();
  if (ready) await Promise.all([loadModelInfo(), loadExamples()]);
}

document.addEventListener("DOMContentLoaded", init);
