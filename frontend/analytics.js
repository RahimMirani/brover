"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const POLL_MS = 2000;

  const refs = {
    connDot: $("connDot"),
    connText: $("connText"),
    throttleBanner: $("throttleBanner"),
    throttleTitle: $("throttleTitle"),
    throttleText: $("throttleText"),
    cpuTemp: $("cpuTemp"),
    cpuTempBar: $("cpuTempBar"),
    cpuTotal: $("cpuTotal"),
    cpuCores: $("cpuCores"),
    load1: $("load1"),
    load5: $("load5"),
    load15: $("load15"),
    armClock: $("armClock"),
    coreV: $("coreV"),
    cpuFreqMax: $("cpuFreqMax"),
    memPercent: $("memPercent"),
    memBar: $("memBar"),
    memUsed: $("memUsed"),
    memTotal: $("memTotal"),
    memCached: $("memCached"),
    swapPercent: $("swapPercent"),
    swapBar: $("swapBar"),
    swapUsed: $("swapUsed"),
    swapTotal: $("swapTotal"),
    diskPercent: $("diskPercent"),
    diskBar: $("diskBar"),
    diskUsed: $("diskUsed"),
    diskTotal: $("diskTotal"),
    diskRead: $("diskRead"),
    diskWrite: $("diskWrite"),
    netRx: $("netRx"),
    netTx: $("netTx"),
    netRxTotal: $("netRxTotal"),
    netTxTotal: $("netTxTotal"),
    wifiSignal: $("wifiSignal"),
    wifiIface: $("wifiIface"),
    wifiQuality: $("wifiQuality"),
    camFps: $("camFps"),
    camFrameKb: $("camFrameKb"),
    camDropped: $("camDropped"),
    camSubs: $("camSubs"),
    wsClients: $("wsClients"),
    motorDuty: $("motorDuty"),
    motorDutyBar: $("motorDutyBar"),
    motorLast: $("motorLast"),
    modeStack: $("modeStack"),
    modeIdle: $("modeIdle"),
    modeManual: $("modeManual"),
    modeAi: $("modeAi"),
    errCount: $("errCount"),
    estopCount: $("estopCount"),
    toolCalls: $("toolCalls"),
    selfCpu: $("selfCpu"),
    selfPid: $("selfPid"),
    selfRss: $("selfRss"),
    selfThreads: $("selfThreads"),
    camCpu: $("camCpu"),
    camPid: $("camPid"),
    camRss: $("camRss"),
    appUptime: $("appUptime"),
    sysUptime: $("sysUptime"),
    lastUpdate: $("lastUpdate"),
    sparkCpu: $("sparkCpu"),
    sparkTemp: $("sparkTemp"),
    sparkMem: $("sparkMem"),
    sparkRx: $("sparkRx"),
    sparkTx: $("sparkTx"),
    sparkDiskW: $("sparkDiskW"),
  };

  /* -------------------------------------------------------- formatters */

  const NBSP = "\u00A0";

  function fmtBytes(n) {
    if (n == null || Number.isNaN(n)) return "--";
    const abs = Math.abs(n);
    if (abs < 1024) return n.toFixed(0) + "B";
    if (abs < 1024 * 1024) return (n / 1024).toFixed(1) + "KB";
    if (abs < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + "MB";
    return (n / 1024 / 1024 / 1024).toFixed(2) + "GB";
  }

  function fmtRate(n) {
    if (n == null || Number.isNaN(n)) return "--";
    if (n < 1024) return n.toFixed(0) + "B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + "KB";
    return (n / 1024 / 1024).toFixed(2) + "MB";
  }

  function fmtNum(n, digits = 1) {
    if (n == null || Number.isNaN(n)) return "--";
    return Number(n).toFixed(digits);
  }

  function fmtInt(n) {
    if (n == null || Number.isNaN(n)) return "--";
    return Math.round(Number(n)).toString();
  }

  function fmtDuration(seconds) {
    if (seconds == null || Number.isNaN(seconds)) return "--";
    const s = Math.floor(seconds);
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${sec}s`;
    return `${sec}s`;
  }

  function fmtHz(hz) {
    if (hz == null) return "--";
    return (hz / 1_000_000).toFixed(0);
  }

  /* -------------------------------------------------------- sparkline */

  function renderSpark(svg, values) {
    if (!svg) return;
    const arr = (values || []).filter((v) => v != null && !Number.isNaN(v));
    svg.innerHTML = "";
    if (arr.length < 2) return;
    const max = Math.max(1e-6, ...arr);
    const min = Math.min(...arr);
    const range = Math.max(1e-6, max - min);
    const w = 100;
    const h = 24;
    const pad = 2;
    const iw = w - pad * 2;
    const ih = h - pad * 2;
    const n = arr.length;
    const pts = arr.map((v, i) => {
      const x = pad + (i * iw) / (n - 1);
      const y = pad + ih - ((v - min) / range) * ih;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    path.setAttribute("points", pts.join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1");
    path.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(path);
  }

  /* -------------------------------------------------------- render */

  function setGauge(el, percent, max) {
    if (!el) return;
    const p = Math.max(0, Math.min(100, (percent / (max || 100)) * 100));
    el.style.width = p.toFixed(1) + "%";
    el.removeAttribute("data-severity");
    if (p >= 90) el.setAttribute("data-severity", "crit");
    else if (p >= 75) el.setAttribute("data-severity", "warn");
  }

  function setCpuCores(container, cores) {
    if (!container) return;
    while (container.childElementCount < cores.length) {
      const b = document.createElement("div");
      b.className = "cpu-bar";
      const inner = document.createElement("div");
      inner.className = "cpu-bar__fill";
      b.appendChild(inner);
      container.appendChild(b);
    }
    while (container.childElementCount > cores.length) {
      container.removeChild(container.lastChild);
    }
    cores.forEach((v, i) => {
      const fill = container.children[i].firstChild;
      const p = Math.max(0, Math.min(100, v || 0));
      fill.style.height = p.toFixed(1) + "%";
      fill.removeAttribute("data-severity");
      if (p >= 90) fill.setAttribute("data-severity", "crit");
      else if (p >= 70) fill.setAttribute("data-severity", "warn");
    });
  }

  function setStack(el, share) {
    if (!el) return;
    const segs = el.querySelectorAll(".stackbar__seg");
    const order = ["idle", "manual", "ai"];
    order.forEach((k, i) => {
      const v = Math.max(0, Math.min(100, (share && share[k]) || 0));
      segs[i].style.width = v.toFixed(2) + "%";
    });
  }

  function setThrottleBanner(t) {
    const banner = refs.throttleBanner;
    if (!banner) return;
    if (!t) {
      banner.setAttribute("data-sev", "unknown");
      refs.throttleTitle.textContent = "PMIC STATUS";
      refs.throttleText.textContent = "vcgencmd not available on this host";
      return;
    }
    banner.setAttribute("data-sev", t.severity || "unknown");
    const flags = t.flags || {};
    const nowFlags = Object.entries(flags)
      .filter(([k, v]) => v && k.endsWith("_now"))
      .map(([k]) => k.replace("_now", "").replace(/_/g, " "));
    const pastFlags = Object.entries(flags)
      .filter(([k, v]) => v && k.endsWith("_past"))
      .map(([k]) => k.replace("_past", "").replace(/_/g, " "));

    if (nowFlags.length) {
      refs.throttleTitle.textContent = "THROTTLED / UNDER-VOLTAGE";
      refs.throttleText.textContent = "active: " + nowFlags.join(", ");
    } else if (pastFlags.length) {
      refs.throttleTitle.textContent = "PAST EVENT SINCE BOOT";
      refs.throttleText.textContent = "seen: " + pastFlags.join(", ");
    } else {
      refs.throttleTitle.textContent = "PMIC OK";
      refs.throttleText.textContent = "no throttle or under-voltage flags";
    }
  }

  function renderToolCalls(obj) {
    const el = refs.toolCalls;
    if (!el) return;
    el.innerHTML = "";
    const entries = Object.entries(obj || {}).sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
      const e = document.createElement("div");
      e.className = "kv__empty";
      e.textContent = "no calls yet";
      el.appendChild(e);
      return;
    }
    for (const [k, v] of entries) {
      const row = document.createElement("div");
      row.className = "kv__row";
      const key = document.createElement("span");
      key.className = "kv__k";
      key.textContent = k;
      const val = document.createElement("span");
      val.className = "kv__v";
      val.textContent = v.toString();
      row.appendChild(key);
      row.appendChild(val);
      el.appendChild(row);
    }
  }

  function render(snap) {
    refs.connDot.setAttribute("data-on", "");
    refs.connText.textContent = "ONLINE";

    const host = snap.host || {};
    const hist = snap.history || {};
    const app = snap.app || {};

    setThrottleBanner(host.throttled);

    refs.cpuTemp.textContent = host.cpu_temp_c != null ? fmtNum(host.cpu_temp_c, 1) : "--";
    setGauge(refs.cpuTempBar, host.cpu_temp_c || 0, 90);
    refs.cpuTotal.textContent = host.cpu_percent != null ? fmtNum(host.cpu_percent, 0) : "--";
    setCpuCores(refs.cpuCores, host.cpu_per_core || []);

    const la = host.loadavg || [null, null, null];
    refs.load1.textContent = la[0] != null ? fmtNum(la[0], 2) : "--";
    refs.load5.textContent = la[1] != null ? fmtNum(la[1], 2) : "--";
    refs.load15.textContent = la[2] != null ? fmtNum(la[2], 2) : "--";

    refs.armClock.textContent = host.arm_clock_hz != null ? fmtHz(host.arm_clock_hz) : (host.cpu_freq_mhz != null ? fmtNum(host.cpu_freq_mhz, 0) : "--");
    refs.coreV.textContent = host.core_voltage_v != null ? fmtNum(host.core_voltage_v, 3) : "--";
    refs.cpuFreqMax.textContent = host.cpu_freq_max_mhz != null ? fmtNum(host.cpu_freq_max_mhz, 0) : "--";

    refs.memPercent.textContent = host.mem_percent != null ? fmtNum(host.mem_percent, 0) : "--";
    setGauge(refs.memBar, host.mem_percent || 0, 100);
    refs.memUsed.textContent = fmtBytes(host.mem_used_bytes);
    refs.memTotal.textContent = fmtBytes(host.mem_total_bytes);
    refs.memCached.textContent = fmtBytes(host.mem_cached_bytes);

    const swapPct = host.swap_total_bytes
      ? (100 * host.swap_used_bytes) / host.swap_total_bytes
      : 0;
    refs.swapPercent.textContent = fmtNum(swapPct, 0);
    setGauge(refs.swapBar, swapPct, 100);
    refs.swapUsed.textContent = fmtBytes(host.swap_used_bytes);
    refs.swapTotal.textContent = fmtBytes(host.swap_total_bytes);

    refs.diskPercent.textContent = host.disk_percent != null ? fmtNum(host.disk_percent, 0) : "--";
    setGauge(refs.diskBar, host.disk_percent || 0, 100);
    refs.diskUsed.textContent = fmtBytes(host.disk_used_bytes);
    refs.diskTotal.textContent = fmtBytes(host.disk_total_bytes);
    refs.diskRead.textContent = fmtRate(host.disk_read_bytes_per_s);
    refs.diskWrite.textContent = fmtRate(host.disk_write_bytes_per_s);

    refs.netRx.textContent = fmtRate(host.net_rx_bytes_per_s);
    refs.netTx.textContent = fmtRate(host.net_tx_bytes_per_s);
    refs.netRxTotal.textContent = fmtBytes(host.net_total_rx_bytes);
    refs.netTxTotal.textContent = fmtBytes(host.net_total_tx_bytes);

    const wifi = host.wifi || {};
    refs.wifiSignal.textContent = wifi.signal_dbm != null ? fmtNum(wifi.signal_dbm, 0) : "--";
    refs.wifiIface.textContent = wifi.iface || "--";
    refs.wifiQuality.textContent = wifi.link_quality != null ? fmtNum(wifi.link_quality, 0) : "--";

    refs.camFps.textContent = app.camera_fps != null ? fmtNum(app.camera_fps, 1) : "--";
    refs.camFrameKb.textContent =
      app.camera_last_frame_bytes != null
        ? (app.camera_last_frame_bytes / 1024).toFixed(1)
        : "--";
    refs.camDropped.textContent = fmtInt(app.camera_frames_dropped);
    refs.camSubs.textContent = fmtInt(app.camera_subscribers);
    refs.wsClients.textContent = fmtInt(app.ws_clients);

    refs.motorDuty.textContent = app.motor_duty_percent_60s != null ? fmtNum(app.motor_duty_percent_60s, 1) : "--";
    setGauge(refs.motorDutyBar, app.motor_duty_percent_60s || 0, 100);
    refs.motorLast.textContent = app.motor_last_cmd || "--";

    setStack(refs.modeStack, app.mode_time_share_5m_percent);
    const share = app.mode_time_share_5m_percent || {};
    refs.modeIdle.textContent = fmtNum(share.idle, 0);
    refs.modeManual.textContent = fmtNum(share.manual, 0);
    refs.modeAi.textContent = fmtNum(share.ai, 0);

    refs.errCount.textContent = fmtInt(app.error_count);
    refs.estopCount.textContent = fmtInt(app.estop_count);
    renderToolCalls(app.tool_calls);

    const sp = host.self_proc || {};
    refs.selfCpu.textContent = sp.cpu_percent != null ? fmtNum(sp.cpu_percent, 0) : "--";
    refs.selfPid.textContent = sp.pid != null ? sp.pid.toString() : "--";
    refs.selfRss.textContent = fmtBytes(sp.rss_bytes);
    refs.selfThreads.textContent = sp.num_threads != null ? sp.num_threads.toString() : "--";

    const cp = host.camera_proc || {};
    refs.camCpu.textContent = cp.cpu_percent != null ? fmtNum(cp.cpu_percent, 0) : "--";
    refs.camPid.textContent = cp.pid != null ? cp.pid.toString() : "--";
    refs.camRss.textContent = fmtBytes(cp.rss_bytes);

    refs.appUptime.textContent = fmtDuration(snap.app_uptime_seconds);
    refs.sysUptime.textContent = fmtDuration(host.uptime_seconds);

    renderSpark(refs.sparkCpu, hist.cpu_percent);
    renderSpark(refs.sparkTemp, hist.cpu_temp_c);
    renderSpark(refs.sparkMem, hist.mem_percent);
    renderSpark(refs.sparkRx, hist.net_rx_bytes_per_s);
    renderSpark(refs.sparkTx, hist.net_tx_bytes_per_s);
    renderSpark(refs.sparkDiskW, hist.disk_write_bytes_per_s);

    const now = new Date();
    refs.lastUpdate.textContent =
      String(now.getHours()).padStart(2, "0") + ":" +
      String(now.getMinutes()).padStart(2, "0") + ":" +
      String(now.getSeconds()).padStart(2, "0");
  }

  function setOffline(reason) {
    refs.connDot.removeAttribute("data-on");
    refs.connText.textContent = reason || "OFFLINE";
  }

  /* -------------------------------------------------------- poll loop */

  let timer = null;

  async function tick() {
    if (document.visibilityState === "hidden") return;
    try {
      const resp = await fetch("/api/metrics", { cache: "no-store" });
      if (!resp.ok) {
        setOffline("HTTP " + resp.status);
        return;
      }
      const snap = await resp.json();
      render(snap);
    } catch (err) {
      setOffline("OFFLINE");
    }
  }

  function start() {
    if (timer) return;
    tick();
    timer = setInterval(tick, POLL_MS);
  }

  function stop() {
    if (!timer) return;
    clearInterval(timer);
    timer = null;
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      stop();
    } else {
      start();
    }
  });

  start();

  void NBSP;
})();
