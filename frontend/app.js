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
        if (msg.text) appendLog("sys", `"${msg.text}"`);
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

  function appendLog(kind, text) {
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

    logBody.appendChild(el);
    logBody.scrollTop = logBody.scrollHeight;

    while (logBody.childElementCount > 200) {
      logBody.removeChild(logBody.firstChild);
    }
    state.entries += 1;
    logCount.textContent = String(state.entries).padStart(3, "0");
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

  /* ----------------------------------------------------------- Boot */

  connect();
})();
