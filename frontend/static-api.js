"use strict";

/* =========================================================================
   Static mode: in-browser inference (GitHub Pages build only)

   Emulates the FastAPI endpoints used by app.js by intercepting fetch():
     GET  /health, /model-info, /examples   -> pre-exported JSON (api/*.json)
     POST /predict, /predict/batch          -> MLP forward pass in JavaScript

   The model is exported by scripts/build_static_site.py with BatchNorm layers
   folded into the preceding Linear layers (exact in eval mode), so inference is
   a chain of affine transforms + ReLU followed by softmax. Preprocessing is the
   same StandardScaler (mean/scale) fitted on the training split.
   ========================================================================= */
(() => {
  const ASSET_BASE = new URL("api/", document.currentScript.src);
  const REPO_URL = "https://github.com/evez12/forest-cover-type-classifier";
  const nativeFetch = window.fetch.bind(window);

  const loadJson = async (name) => {
    const res = await nativeFetch(new URL(name, ASSET_BASE));
    if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
    return res.json();
  };

  const decodeFloat32 = (b64) => {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  };

  // Lazily loaded once, shared by all requests
  let modelPromise = null;
  const getModel = () =>
    (modelPromise ??= loadJson("model.json").then((m) => ({
      ...m,
      layers: m.layers.map((l) => ({ ...l, weight: decodeFloat32(l.weight), bias: decodeFloat32(l.bias) })),
    })));

  const jsonResponse = (body, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

  const validationError = (loc, msg) => ({ loc: ["body", ...loc], msg, type: "value_error" });

  function validate(record, model, index = null) {
    const prefix = index === null ? [] : ["instances", index];
    const errors = [];
    const allowed = new Set(model.feature_names);
    for (const key of Object.keys(record)) {
      if (!allowed.has(key)) errors.push(validationError([...prefix, key], "Extra inputs are not permitted"));
    }
    for (const [name, [lo, hi]] of Object.entries(model.numeric_bounds)) {
      const v = record[name];
      if (typeof v !== "number" || !Number.isFinite(v)) errors.push(validationError([...prefix, name], "Field required (number)"));
      else if ((lo !== null && v < lo) || (hi !== null && v > hi)) {
        errors.push(validationError([...prefix, name], `Value must be in [${lo ?? "-inf"}, ${hi ?? "inf"}]`));
      }
    }
    for (const name of model.binary_features) {
      const v = record[name] ?? 0;
      if (v !== 0 && v !== 1) errors.push(validationError([...prefix, name], "Input should be 0 or 1"));
    }
    for (const [group, prefixName] of [["wilderness_features", "Wilderness_Area"], ["soil_features", "Soil_Type"]]) {
      const active = model[group].reduce((acc, f) => acc + (record[f] ?? 0), 0);
      if (active !== 1) {
        errors.push(validationError(prefix, `Exactly one ${prefixName}_* field must be 1 (got ${active})`));
      }
    }
    return errors;
  }

  function forward(record, model) {
    // Preprocess: StandardScaler on numeric features, passthrough on one-hot
    let x = Float32Array.from(model.feature_names, (name, i) => {
      const v = record[name] ?? 0;
      return i < model.scaler.mean.length ? (v - model.scaler.mean[i]) / model.scaler.scale[i] : v;
    });
    for (const layer of model.layers) {
      const [out, inp] = layer.shape;
      const y = new Float32Array(out);
      for (let o = 0; o < out; o++) {
        let acc = layer.bias[o];
        const row = o * inp;
        for (let i = 0; i < inp; i++) acc += layer.weight[row + i] * x[i];
        y[o] = layer.relu && acc < 0 ? 0 : acc;
      }
      x = y;
    }
    const maxLogit = Math.max(...x);
    const exps = Array.from(x, (v) => Math.exp(v - maxLogit));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / total);
  }

  function toPrediction(record, model) {
    const proba = forward(record, model);
    const top = proba.indexOf(Math.max(...proba));
    const warnings = [];
    for (const [name, s] of Object.entries(model.feature_stats)) {
      const v = record[name];
      if (v < s.min || v > s.max) {
        warnings.push(`${name}=${v} is outside the training range [${s.min}, ${s.max}] — prediction is an extrapolation.`);
      }
    }
    const cls = model.classes[top];
    return {
      class_index: cls.class_index,
      cover_type: cls.cover_type,
      class_name: cls.class_name,
      confidence: proba[top],
      probabilities: model.classes
        .map((c) => ({ class_index: c.class_index, cover_type: c.cover_type, class_name: c.class_name, probability: proba[c.class_index] }))
        .sort((a, b) => b.probability - a.probability),
      warnings,
    };
  }

  async function handle(path, init) {
    switch (path) {
      case "/health":
        await getModel();
        return jsonResponse({ status: "ok", model_loaded: true, device: "in-browser", detail: null });
      case "/model-info":
        return jsonResponse(await loadJson("model-info.json"));
      case "/examples":
        return jsonResponse(await loadJson("examples.json"));
      case "/predict": {
        const model = await getModel();
        const t0 = performance.now();
        const record = JSON.parse(init?.body ?? "{}");
        const errors = validate(record, model);
        if (errors.length) return jsonResponse({ detail: errors }, 422);
        return jsonResponse({ ...toPrediction(record, model), latency_ms: performance.now() - t0 });
      }
      case "/predict/batch": {
        const model = await getModel();
        const t0 = performance.now();
        const { instances = [] } = JSON.parse(init?.body ?? "{}");
        const errors = instances.flatMap((r, i) => validate(r, model, i));
        if (!instances.length) errors.push(validationError(["instances"], "List should have at least 1 item"));
        if (errors.length) return jsonResponse({ detail: errors }, 422);
        const predictions = instances.map((r) => ({ ...toPrediction(r, model), latency_ms: null }));
        return jsonResponse({ predictions, count: predictions.length, latency_ms: performance.now() - t0 });
      }
      default:
        return null;
    }
  }

  window.fetch = async (input, init) => {
    const url = new URL(typeof input === "string" ? input : input.url, window.location.href);
    if (url.origin === window.location.origin) {
      const response = await handle(url.pathname.replace(/\/+$/, ""), init);
      if (response) return response;
    }
    return nativeFetch(input, init);
  };

  // No Swagger UI in static mode: point the header link to the repository instead
  window.addEventListener("load", () => {
    const link = document.getElementById("docs-link");
    if (link) {
      link.href = REPO_URL;
      link.textContent = "GitHub";
    }
  });
})();
