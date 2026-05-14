"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const connDot = $("connDot");
  const connText = $("connText");
  const modeBadge = $("modeBadge");
  const logBody = $("logBody");
  const logCount = $("logCount");
  const micBtn = $("micBtn");
  const estopBtn = $("estopBtn");
  const padStop = $("padStop");
  const clock = $("clock");
  const sessionId = $("sessionId");
  const video = $("video");
  const zoomIn = $("zoomIn");
  const zoomOut = $("zoomOut");
  const zoomVal = $("zoomVal");
  const modelPicker = $("modelPicker");
  const distanceReadout = $("distanceReadout");
  const trainToggle = $("trainToggle");
  const trainPanel = $("trainPanel");
  const trainClose = $("trainClose");
  const captureBtn = $("captureBtn");
  const captureList = $("captureList");
  const capturePending = $("capturePending");
  const placeNamesList = $("placeNames");
  const routeStatus = $("routeStatus");
  const routeFromInput = $("routeFromInput");
  const routeStartBtn = $("routeStartBtn");
  const routeToInput = $("routeToInput");
  const routeStopBtn = $("routeStopBtn");
  const routeCancelBtn = $("routeCancelBtn");
  const memoryRefresh = $("memoryRefresh");
  const placesList = $("placesList");
  const placesCount = $("placesCount");
  const routesList = $("routesList");
  const routesCount = $("routesCount");

  const state = {
    sock: null,
    connected: false,
    mode: "idle",
    activeMove: null,
    moveTimer: null,
    recorder: null,
    audioChunks: [],
    audioStream: null,
    audioUnlocked: false,
    entries: 0,
    models: [],
    currentModel: "",
    train: {
      open: false,
      pending: new Map(),
      placeNames: [],
      pollTimer: null,
      routeActive: false,
    },
  };

  sessionId.textContent =
    "SID " + Math.random().toString(36).slice(2, 10).toUpperCase();

  /* -------------------------------------------------------------- WS */

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws`;
    state.sock = new WebSocket(url);

    state.sock.addEventListener("open", () => {
      state.connected = true;
      connDot.setAttribute("data-on", "");
      connText.textContent = "ONLINE";
      sendSelectedModel();
    });

    state.sock.addEventListener("message", (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      handle(msg);
    });

    state.sock.addEventListener("close", () => {
      state.connected = false;
      connDot.removeAttribute("data-on");
      connText.textContent = "OFFLINE";
      setMode("idle");
      setTimeout(connect, 1000);
    });

    state.sock.addEventListener("error", () => {
      // close handler will fire right after; no-op here
    });
  }

  function send(msg) {
    if (state.connected && state.sock.readyState === WebSocket.OPEN) {
      state.sock.send(JSON.stringify(msg));
    }
  }

  setInterval(() => send({ type: "ping" }), 1000);

  /* --------------------------------------------------------- Inbound */

  function handle(msg) {
    switch (msg.type) {
      case "mode":
        setMode(msg.state);
        break;
      case "model":
        setCurrentModel(msg.model || "");
        break;
      case "transcript":
        appendLog("usr", msg.text ? `"${msg.text}"` : "(silence)");
        break;
      case "tool_call": {
        const args = msg.arguments && Object.keys(msg.arguments).length
          ? ` ${JSON.stringify(msg.arguments)}`
          : "";
        appendLog("tol", `${msg.name}${args}`);
        break;
      }
      case "tool_result": {
        const text = (msg.content || [])
          .filter((c) => c && c.type === "text")
          .map((c) => c.text)
          .join(" ");
        appendLog("res", text || "ok");
        break;
      }
      case "final":
        if (msg.text) appendLog("sys", `"${msg.text}"`, formatTelemetry(msg));
        break;
      case "audio_reply":
        playAudio(msg.data);
        break;
      case "error":
        appendLog("err", msg.text || "error");
        break;
      default:
        break;
    }
  }

  function setMode(m) {
    state.mode = m;
    modeBadge.dataset.mode = m;
    modeBadge.textContent = m.toUpperCase();
    micBtn.disabled = m === "ai";
  }

  /* -------------------------------------------------------- Log view */

  function appendLog(kind, text, detail = "") {
    const now = new Date();
    const t =
      String(now.getHours()).padStart(2, "0") + ":" +
      String(now.getMinutes()).padStart(2, "0") + ":" +
      String(now.getSeconds()).padStart(2, "0");

    const el = document.createElement("div");
    el.className = `entry entry--${kind}`;
    const tag = kind.toUpperCase();
    el.innerHTML =
      `<span class="entry__time">${t}</span>` +
      `<span class="entry__tag">${tag}</span>` +
      `<span class="entry__text"></span>`;
    el.querySelector(".entry__text").textContent = text;
    if (detail) {
      const detailEl = document.createElement("span");
      detailEl.className = "entry__detail";
      detailEl.textContent = detail;
      el.appendChild(detailEl);
    }

    logBody.appendChild(el);
    logBody.scrollTop = logBody.scrollHeight;

    while (logBody.childElementCount > 200) {
      logBody.removeChild(logBody.firstChild);
    }
    state.entries += 1;
    logCount.textContent = String(state.entries).padStart(3, "0");
  }

  /* ------------------------------------------------------ Model picker */

  async function loadModels() {
    try {
      const res = await fetch("/api/models", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      state.models = Array.isArray(payload.models) ? payload.models : [];
      renderModelOptions(payload.default_model || "");
      sendSelectedModel();
    } catch (err) {
      modelPicker.innerHTML = "";
      const opt = document.createElement("option");
      opt.textContent = "Models unavailable";
      modelPicker.appendChild(opt);
      modelPicker.disabled = true;
      appendLog("err", `model list failed: ${err.message || err}`);
    }
  }

  function renderModelOptions(defaultModel) {
    modelPicker.innerHTML = "";
    state.models.forEach((model) => {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.display_name || model.id;
      modelPicker.appendChild(opt);
    });
    const selected = state.models.some((m) => m.id === defaultModel)
      ? defaultModel
      : (state.models[0] && state.models[0].id) || "";
    modelPicker.value = selected;
    state.currentModel = selected;
    modelPicker.disabled = !selected;
  }

  function setCurrentModel(model) {
    if (!model) return;
    state.currentModel = model;
    if (Array.from(modelPicker.options).some((opt) => opt.value === model)) {
      modelPicker.value = model;
    }
  }

  function sendSelectedModel() {
    const model = modelPicker.value || state.currentModel;
    if (model) send({ type: "set_model", model });
  }

  function modelLabel(modelId) {
    const model = state.models.find((m) => m.id === modelId);
    return (model && model.display_name) || modelId || "model";
  }

  function formatTokens(n) {
    const value = Number(n) || 0;
    if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}K`;
    return String(value);
  }

  function formatTelemetry(msg) {
    if (!msg.model) return "";
    const seconds = ((Number(msg.latency_ms) || 0) / 1000).toFixed(1);
    const cost = (Number(msg.cost_usd) || 0).toFixed(4);
    return [
      modelLabel(msg.model),
      `${msg.iterations || 0} iter`,
      `${seconds}s`,
      `${formatTokens(msg.input_tokens)} in / ${formatTokens(msg.output_tokens)} out`,
      `$${cost}`,
    ].join(" · ");
  }

  modelPicker.addEventListener("change", () => {
    state.currentModel = modelPicker.value;
    sendSelectedModel();
  });

  /* -------------------------------------------------------- Sensors */

  function formatDistance(cm) {
    if (cm >= 100) return `${(cm / 100).toFixed(2)}m`;
    return `${Math.round(cm)}cm`;
  }

  function updateDistance(reading) {
    if (!distanceReadout) return;
    if (!reading || typeof reading.distance_cm !== "number") {
      distanceReadout.textContent = "DIST --";
      distanceReadout.dataset.status = "stale";
      return;
    }

    distanceReadout.textContent = `DIST ${formatDistance(reading.distance_cm)}`;
    if (reading.stale) {
      distanceReadout.dataset.status = "stale";
    } else if (!reading.safe_for_forward) {
      distanceReadout.dataset.status = "warn";
    } else {
      distanceReadout.dataset.status = "safe";
    }
  }

  async function pollSensors() {
    try {
      const res = await fetch("/api/sensors", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      updateDistance(payload.distance);
    } catch (_) {
      if (distanceReadout) {
        distanceReadout.textContent = "DIST --";
        distanceReadout.dataset.status = "stale";
      }
    }
  }

  /* ------------------------------------------------------- Manual drive */

  function startMove(cmd) {
    if (state.activeMove === cmd) return;
    stopMove();
    state.activeMove = cmd;
    send({ type: "move", cmd });
    state.moveTimer = setInterval(() => send({ type: "move", cmd }), 60);
  }

  function stopMove() {
    if (state.moveTimer) {
      clearInterval(state.moveTimer);
      state.moveTimer = null;
    }
    if (state.activeMove) {
      send({ type: "move", cmd: "stop" });
      state.activeMove = null;
    }
  }

  document.querySelectorAll(".pad[data-cmd]").forEach((btn) => {
    const cmd = btn.dataset.cmd;
    const onDown = (e) => {
      e.preventDefault();
      btn.classList.add("active");
      startMove(cmd);
    };
    const onUp = (e) => {
      if (e) e.preventDefault();
      btn.classList.remove("active");
      stopMove();
    };
    btn.addEventListener("pointerdown", onDown);
    btn.addEventListener("pointerup", onUp);
    btn.addEventListener("pointerleave", onUp);
    btn.addEventListener("pointercancel", onUp);
  });

  padStop.addEventListener("click", () => {
    stopMove();
    send({ type: "move", cmd: "stop" });
  });

  /* Keyboard */
  const keyMap = {
    w: "forward", s: "backward", a: "left", d: "right",
    W: "forward", S: "backward", A: "left", D: "right",
    ArrowUp: "forward", ArrowDown: "backward",
    ArrowLeft: "left", ArrowRight: "right",
  };

  document.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    const cmd = keyMap[e.key];
    if (cmd) {
      e.preventDefault();
      startMove(cmd);
    }
  });
  document.addEventListener("keyup", (e) => {
    if (keyMap[e.key]) {
      e.preventDefault();
      stopMove();
    }
  });
  window.addEventListener("blur", stopMove);

  /* ------------------------------------------------------------ Mic */

  function pickMime() {
    const types = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4;codecs=mp4a.40.2",
      "audio/mp4",
      "audio/ogg;codecs=opus",
    ];
    if (typeof MediaRecorder === "undefined") return "";
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return "";
  }

  async function startRecording() {
    if (state.mode === "ai" || state.recorder) return;
    try {
      state.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      appendLog("err", `mic access denied: ${err.message || err}`);
      return;
    }
    const mimeType = pickMime();
    try {
      state.recorder = mimeType
        ? new MediaRecorder(state.audioStream, { mimeType })
        : new MediaRecorder(state.audioStream);
    } catch (err) {
      appendLog("err", `recorder init failed: ${err.message || err}`);
      return;
    }
    state.audioChunks = [];
    state.recorder.addEventListener("dataavailable", (e) => {
      if (e.data && e.data.size) state.audioChunks.push(e.data);
    });
    state.recorder.addEventListener("stop", async () => {
      if (state.audioStream) {
        state.audioStream.getTracks().forEach((t) => t.stop());
        state.audioStream = null;
      }
      const type = state.recorder ? state.recorder.mimeType || "audio/webm" : "audio/webm";
      const blob = new Blob(state.audioChunks, { type });
      state.recorder = null;
      state.audioChunks = [];
      if (!blob.size) return;
      const data = await blobToBase64(blob);
      send({ type: "audio", data, mime: type });
    });
    state.recorder.start();
    micBtn.classList.add("recording");
  }

  function stopRecording() {
    if (state.recorder && state.recorder.state === "recording") {
      try { state.recorder.stop(); } catch (_) {}
    }
    micBtn.classList.remove("recording");
  }

  function blobToBase64(blob) {
    return new Promise((resolve) => {
      const r = new FileReader();
      r.onloadend = () => {
        const s = r.result || "";
        const i = s.indexOf(",");
        resolve(i >= 0 ? s.slice(i + 1) : "");
      };
      r.readAsDataURL(blob);
    });
  }

  micBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    unlockAudio();
    startRecording();
  });
  micBtn.addEventListener("pointerup", (e) => {
    e.preventDefault();
    stopRecording();
  });
  micBtn.addEventListener("pointerleave", stopRecording);
  micBtn.addEventListener("pointercancel", stopRecording);

  /* ----------------------------------------------------------- E-stop */

  estopBtn.addEventListener("click", () => {
    send({ type: "estop" });
    stopMove();
    appendLog("err", "EMERGENCY STOP");
  });

  /* ---------------------------------------------------- Audio playback */

  // Safari requires a user gesture before any audio playback. We unlock
  // by priming a silent Audio element the first time the user taps anything.
  function unlockAudio() {
    if (state.audioUnlocked) return;
    state.audioUnlocked = true;
    try {
      const a = new Audio();
      a.muted = true;
      a.play().catch(() => {});
    } catch (_) {}
  }
  window.addEventListener("pointerdown", unlockAudio, { once: true });

  function playAudio(b64) {
    try {
      const audio = new Audio(`data:audio/mp3;base64,${b64}`);
      audio.play().catch((err) => console.warn("audio play failed", err));
    } catch (err) {
      console.warn("audio decode failed", err);
    }
  }

  /* ------------------------------------------------------------ Zoom */

  const ZOOM_MIN = 1.0;
  const ZOOM_MAX = 3.0;
  const ZOOM_STEP = 0.25;
  let zoom = ZOOM_MIN;

  function applyZoom() {
    video.style.transform = `scale(${zoom})`;
    zoomVal.textContent = zoom.toFixed(2) + "x";
    zoomIn.disabled = zoom >= ZOOM_MAX - 1e-9;
    zoomOut.disabled = zoom <= ZOOM_MIN + 1e-9;
  }

  zoomIn.addEventListener("click", () => {
    zoom = Math.min(ZOOM_MAX, zoom + ZOOM_STEP);
    applyZoom();
  });
  zoomOut.addEventListener("click", () => {
    zoom = Math.max(ZOOM_MIN, zoom - ZOOM_STEP);
    applyZoom();
  });
  applyZoom();

  /* ----------------------------------------------------------- Clock */

  function tick() {
    const d = new Date();
    clock.textContent =
      String(d.getHours()).padStart(2, "0") + ":" +
      String(d.getMinutes()).padStart(2, "0") + ":" +
      String(d.getSeconds()).padStart(2, "0");
  }
  setInterval(tick, 1000);
  tick();
  setInterval(pollSensors, 500);
  pollSensors();

  /* -------------------------------------------------------- Training */

  // The training panel is a parallel UI surface to the voice/agent flow:
  // user drives with WASD/D-pad (unchanged), clicks CAPTURE here, then
  // names the frame and saves. Routes work the same way -- this just hits
  // the existing route_recorder via HTTP instead of via Claude.

  function openTrain() {
    if (state.train.open) return;
    state.train.open = true;
    trainPanel.hidden = false;
    trainToggle.setAttribute("aria-pressed", "true");
    refreshMemory();
    pollRouteState();
    state.train.pollTimer = setInterval(() => {
      pollRouteState();
    }, 1500);
  }

  function closeTrain() {
    if (!state.train.open) return;
    state.train.open = false;
    trainPanel.hidden = true;
    trainToggle.setAttribute("aria-pressed", "false");
    if (state.train.pollTimer) {
      clearInterval(state.train.pollTimer);
      state.train.pollTimer = null;
    }
  }

  trainToggle.addEventListener("click", () => {
    if (state.train.open) closeTrain(); else openTrain();
  });
  trainClose.addEventListener("click", closeTrain);

  async function captureFrame() {
    if (!state.train.open) return;
    captureBtn.disabled = true;
    try {
      const res = await fetch("/api/training/captures", { method: "POST" });
      if (!res.ok) {
        const msg = await res.text().catch(() => res.statusText);
        appendLog("err", `capture failed (${res.status}): ${msg}`);
        return;
      }
      const payload = await res.json();
      addPendingCard(payload.capture_id);
      appendLog("trn", `captured frame ${payload.capture_id.slice(0, 8)}`);
    } catch (err) {
      appendLog("err", `capture failed: ${err.message || err}`);
    } finally {
      captureBtn.disabled = false;
    }
  }

  function addPendingCard(captureId) {
    const card = document.createElement("div");
    card.className = "capture-card";
    card.dataset.id = captureId;
    card.innerHTML =
      `<img class="capture-card__thumb" src="/api/training/captures/${captureId}" alt="pending capture">` +
      `<div class="capture-card__body">` +
      `  <input class="capture-card__input" list="placeNames" placeholder="place name" autocomplete="off">` +
      `  <div class="capture-card__actions">` +
      `    <button class="capture-card__btn capture-card__btn--save">SAVE</button>` +
      `    <button class="capture-card__btn capture-card__btn--discard">DISCARD</button>` +
      `  </div>` +
      `</div>`;

    const input = card.querySelector(".capture-card__input");
    const saveBtn = card.querySelector(".capture-card__btn--save");
    const discardBtn = card.querySelector(".capture-card__btn--discard");

    const doSave = () => savePending(captureId, input.value.trim(), card);
    saveBtn.addEventListener("click", doSave);
    discardBtn.addEventListener("click", () => discardPending(captureId, card));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        doSave();
      }
    });

    captureList.prepend(card);
    state.train.pending.set(captureId, card);
    updatePendingCount();
    setTimeout(() => input.focus(), 0);
  }

  async function savePending(captureId, name, card) {
    if (!name) {
      appendLog("err", "place name is required");
      return;
    }
    const saveBtn = card.querySelector(".capture-card__btn--save");
    const discardBtn = card.querySelector(".capture-card__btn--discard");
    card.dataset.saving = "true";
    saveBtn.disabled = true;
    discardBtn.disabled = true;
    try {
      const res = await fetch(
        `/api/training/captures/${captureId}/save_place`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        }
      );
      if (!res.ok) {
        const detail = await readErrorDetail(res);
        appendLog("err", `save failed (${res.status}): ${detail}`);
        return;
      }
      const payload = await res.json();
      appendLog(
        "trn",
        `saved ${payload.name} (now ${payload.view_count} view${
          payload.view_count === 1 ? "" : "s"
        })`
      );
      removeCard(captureId);
      refreshMemory();
    } catch (err) {
      appendLog("err", `save failed: ${err.message || err}`);
    } finally {
      delete card.dataset.saving;
      saveBtn.disabled = false;
      discardBtn.disabled = false;
    }
  }

  async function discardPending(captureId, card) {
    try {
      await fetch(`/api/training/captures/${captureId}`, { method: "DELETE" });
    } catch (_) {
      // even if delete fails, drop the card from the UI
    }
    removeCard(captureId);
    appendLog("trn", `discarded capture ${captureId.slice(0, 8)}`);
  }

  function removeCard(captureId) {
    const card = state.train.pending.get(captureId);
    if (card && card.parentNode) card.parentNode.removeChild(card);
    state.train.pending.delete(captureId);
    updatePendingCount();
  }

  function updatePendingCount() {
    const n = state.train.pending.size;
    capturePending.textContent = `${n} pending`;
  }

  captureBtn.addEventListener("click", captureFrame);

  document.addEventListener("keydown", (e) => {
    if (!state.train.open) return;
    if (e.key !== "c" && e.key !== "C") return;
    if (e.repeat) return;
    const tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    e.preventDefault();
    captureFrame();
  });

  /* --- Route recording controls --- */

  async function startRouteRecording() {
    const from = routeFromInput.value.trim();
    if (!from) {
      appendLog("err", "route: from_place is required");
      return;
    }
    routeStartBtn.disabled = true;
    try {
      const res = await fetch("/api/training/routes/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from_place: from }),
      });
      if (!res.ok) {
        const detail = await readErrorDetail(res);
        appendLog("err", `route start failed (${res.status}): ${detail}`);
        return;
      }
      appendLog("trn", `route recording started from ${from}`);
      pollRouteState();
    } catch (err) {
      appendLog("err", `route start failed: ${err.message || err}`);
    } finally {
      routeStartBtn.disabled = false;
    }
  }

  async function stopRouteRecording() {
    const to = routeToInput.value.trim();
    if (!to) {
      appendLog("err", "route: to_place is required");
      return;
    }
    routeStopBtn.disabled = true;
    try {
      const res = await fetch("/api/training/routes/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ to_place: to }),
      });
      if (!res.ok) {
        const detail = await readErrorDetail(res);
        appendLog("err", `route stop failed (${res.status}): ${detail}`);
        return;
      }
      const payload = await res.json();
      appendLog(
        "trn",
        `route saved: ${payload.from_place} -> ${payload.to_place} (${payload.step_count} steps)`
      );
      routeFromInput.value = "";
      routeToInput.value = "";
      pollRouteState();
      refreshMemory();
    } catch (err) {
      appendLog("err", `route stop failed: ${err.message || err}`);
    } finally {
      routeStopBtn.disabled = false;
    }
  }

  async function cancelRouteRecording() {
    try {
      const res = await fetch("/api/training/routes/cancel", { method: "POST" });
      if (res.ok) {
        const payload = await res.json();
        if (payload.cancelled) appendLog("trn", "route recording cancelled");
      }
    } catch (_) {}
    pollRouteState();
  }

  routeStartBtn.addEventListener("click", startRouteRecording);
  routeStopBtn.addEventListener("click", stopRouteRecording);
  routeCancelBtn.addEventListener("click", cancelRouteRecording);

  async function pollRouteState() {
    try {
      const res = await fetch("/api/training/routes/state", { cache: "no-store" });
      if (!res.ok) return;
      const payload = await res.json();
      renderRouteState(payload);
    } catch (_) {
      // network blip; UI keeps its previous state.
    }
  }

  function renderRouteState(s) {
    const active = !!s.active;
    const overflowed = !!s.overflowed;
    state.train.routeActive = active;

    if (overflowed) {
      routeStatus.dataset.state = "overflowed";
      routeStatus.textContent = `OVERFLOW ${s.step_count}`;
    } else if (active) {
      routeStatus.dataset.state = "recording";
      routeStatus.textContent = `REC ${s.from_place} \u00b7 ${s.step_count}`;
    } else {
      routeStatus.dataset.state = "idle";
      routeStatus.textContent = "IDLE";
    }

    routeFromInput.disabled = active;
    routeStartBtn.disabled = active;
    routeToInput.disabled = !active;
    routeStopBtn.disabled = !active;
    routeCancelBtn.disabled = !active;
  }

  /* --- Memory lists + autocomplete --- */

  memoryRefresh.addEventListener("click", refreshMemory);

  async function refreshMemory() {
    await Promise.all([refreshPlaces(), refreshRoutes()]);
  }

  async function refreshPlaces() {
    try {
      const res = await fetch("/api/places", { cache: "no-store" });
      if (!res.ok) return;
      const payload = await res.json();
      const places = Array.isArray(payload.places) ? payload.places : [];
      state.train.placeNames = places.map((p) => p.name);
      renderPlaceNames();
      renderPlaces(places);
    } catch (_) {}
  }

  function renderPlaceNames() {
    placeNamesList.innerHTML = "";
    state.train.placeNames.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      placeNamesList.appendChild(opt);
    });
  }

  function renderPlaces(places) {
    placesCount.textContent = String(places.length);
    placesList.innerHTML = "";
    if (places.length === 0) {
      const li = document.createElement("li");
      li.className = "train__list--empty";
      li.textContent = "no places taught yet";
      placesList.appendChild(li);
      return;
    }
    places.forEach((p) => {
      const li = document.createElement("li");
      li.innerHTML =
        `<span class="train__list-name"></span>` +
        `<span class="train__list-meta"></span>`;
      li.querySelector(".train__list-name").textContent = p.name;
      li.querySelector(".train__list-meta").textContent = `${p.view_count} view${
        p.view_count === 1 ? "" : "s"
      }`;
      placesList.appendChild(li);
    });
  }

  async function refreshRoutes() {
    try {
      const res = await fetch("/api/routes", { cache: "no-store" });
      if (!res.ok) return;
      const payload = await res.json();
      const routes = Array.isArray(payload.routes) ? payload.routes : [];
      renderRoutes(routes);
    } catch (_) {}
  }

  function renderRoutes(routes) {
    routesCount.textContent = String(routes.length);
    routesList.innerHTML = "";
    if (routes.length === 0) {
      const li = document.createElement("li");
      li.className = "train__list--empty";
      li.textContent = "no routes recorded yet";
      routesList.appendChild(li);
      return;
    }
    routes.forEach((r) => {
      const li = document.createElement("li");
      li.innerHTML =
        `<span class="train__list-name"></span>` +
        `<span class="train__list-meta"></span>`;
      li.querySelector(".train__list-name").textContent =
        `${r.from_place} \u2192 ${r.to_place}`;
      li.querySelector(".train__list-meta").textContent = `${r.step_count} step${
        r.step_count === 1 ? "" : "s"
      }`;
      routesList.appendChild(li);
    });
  }

  async function readErrorDetail(res) {
    try {
      const body = await res.json();
      if (body && body.detail) return body.detail;
    } catch (_) {}
    try {
      return await res.text();
    } catch (_) {
      return res.statusText;
    }
  }

  /* ----------------------------------------------------------- Boot */

  loadModels();
  connect();
})();
