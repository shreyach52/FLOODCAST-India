"use strict";
const $ = (s, el = document) => el.querySelector(s);
const BLUE = "#4a6fa5", DARK = "#1f2933", AMBER = "#a8691a", GREY = "#9aa5b1";
Chart.defaults.font.family = 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif';
Chart.defaults.font.size = 12;
Chart.defaults.color = "#6b7684";
Chart.defaults.animation = false;

const S = { summary: null, sid: null, horizon: 1, charts: {}, perf: { split: "A_seen_stations", horizon: 1 } };
const SPLIT_NAMES = { A_seen_stations: "Seen stations (later dates)", B_unseen_stations: "Unseen stations (never trained on)" };
const MODEL_NAMES = { lightgbm: "LightGBM", lstm: "LSTM", persistence: "Persistence (baseline)", climatology: "Climatology (baseline)" };

async function api(path) {
  const r = await fetch("/api/" + path);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 1) => v == null ? "—" : Number(v).toLocaleString("en-IN", { maximumFractionDigits: d });
const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
const badge = r => `<span class="badge ${r || "none"}">${r || "No data"}</span>`;
const stName = s => `${s.id}${s.river ? " — " + s.river : ""}${s.state ? ", " + s.state : ""}`;

function draw(id, cfg) {
  S.charts[id]?.destroy();
  const el = document.getElementById(id);
  if (el) S.charts[id] = new Chart(el, cfg);
}
const axisY = t => ({ title: { display: true, text: t }, grid: { color: "#eef0f4" }, beginAtZero: true });
const axisX = { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 } };

/* shades the part of the x axis after the last observation = the forecast period */
const shade = {
  id: "shade",
  beforeDatasetsDraw(c, _a, o) {
    if (o.from == null) return;
    const x = c.scales.x, a = c.chartArea;
    const left = x.getPixelForValue(o.from);
    c.ctx.save(); c.ctx.fillStyle = "rgba(74,111,165,.09)";
    c.ctx.fillRect(left, a.top, a.right - left, a.bottom - a.top);
    c.ctx.fillStyle = BLUE; c.ctx.font = "12px system-ui"; c.ctx.fillText("Forecast period", left + 6, a.top + 14);
    c.ctx.restore();
  }
};

/* ---------- shared: forecast panel ---------- */
function forecastPanel(title) {
  return `<section class="panel">
    <div class="panel-head"><h2>${title}</h2><div class="controls">
      <select id="sel" aria-label="Station"></select>
      <div class="seg" id="seg">${[1, 3, 7].map(h => `<button data-h="${h}">${h} day${h > 1 ? "s" : ""}</button>`).join("")}</div>
    </div></div>
    <div class="meta" id="fmeta"></div>
    <div class="chartbox"><canvas id="cFc"></canvas></div>
    <div id="fmsg"></div></section>`;
}
function fillSelect() {
  const sel = $("#sel");
  sel.innerHTML = S.summary.stations.map(s => `<option value="${esc(s.id)}">${esc(stName(s))}</option>`).join("");
  sel.value = S.sid;
  sel.onchange = () => { S.sid = sel.value; if (location.hash.startsWith("#/station")) location.hash = "#/station/" + encodeURIComponent(S.sid); else loadStation(); };
  document.querySelectorAll("#seg button").forEach(b => {
    b.classList.toggle("on", +b.dataset.h === S.horizon);
    b.onclick = () => { S.horizon = +b.dataset.h; document.querySelectorAll("#seg button").forEach(x => x.classList.toggle("on", x === b)); renderForecast(); };
  });
}
let CUR = null; // {st, fc, wx}
async function loadStation() {
  const sid = S.sid;
  const [st, fc, wx] = await Promise.all([api(`station/${encodeURIComponent(sid)}?days=365`), api(`forecasts/${encodeURIComponent(sid)}`), api(`weather/${encodeURIComponent(sid)}`)]);
  if (sid !== S.sid) return;
  CUR = { st, fc, wx };
  renderForecast(); renderWeather(); if ($("#info")) renderStationExtras();
}
function renderForecast() {
  if (!CUR || !$("#cFc")) return;
  const { st, fc } = CUR, h = S.horizon, a = fc.available ? fc.horizons[h] : null;
  const thr = st.high;
  $("#fmsg").innerHTML = "";
  let meta = `<b>${esc(stName(st))}</b>`;
  if (a) {
    const n = a.dates.length;
    const ds = [
      { label: "Observed", data: a.actual, borderColor: DARK, borderWidth: 1.5, pointRadius: 0, spanGaps: false },
      { label: `Predicted (${h}-day)`, data: a.q_pred, borderColor: BLUE, borderWidth: 1.8, borderDash: [5, 3], pointRadius: 0, spanGaps: false }
    ];
    if (a.q_low.some(v => v != null) && a.q_high.some(v => v != null)) {
      ds.push({ label: "Lower bound", data: a.q_low, borderColor: "rgba(74,111,165,.25)", borderWidth: 1, pointRadius: 0, fill: false });
      ds.push({ label: "Upper bound", data: a.q_high, borderColor: "rgba(74,111,165,.25)", backgroundColor: "rgba(74,111,165,.10)", borderWidth: 1, pointRadius: 0, fill: "-1" });
    }
    if (thr != null) ds.push({ label: "High-flow threshold", data: Array(n).fill(thr), borderColor: AMBER, borderWidth: 1, borderDash: [2, 3], pointRadius: 0 });
    const fi = a.dates.findIndex(d => d > st.last_date);
    draw("cFc", { type: "line", data: { labels: a.dates, datasets: ds },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 14 } }, shade: { from: fi < 0 ? null : fi } },
        scales: { x: axisX, y: axisY("Discharge (m³/s)") } }, plugins: [shade] });
    meta += ` · latest stored ${h}-day forecast: <b>${fmt(a.last_q_pred)} m³/s</b> for ${a.last_target_date} (issued ${a.last_issue_date})`;
    if (a.split) meta += ` · <span class="small">${esc(SPLIT_NAMES[a.split] || a.split)}</span>`;
    if (fi < 0) $("#fmsg").innerHTML = `<div class="msg">All stored forecasts for this station are back-tests on dates that already have observations, so there is no future forecast period to shade.</div>`;
  } else {
    draw("cFc", { type: "line", data: { labels: st.dates, datasets: [
        { label: "Observed", data: st.discharge, borderColor: DARK, borderWidth: 1.5, pointRadius: 0, spanGaps: false },
        ...(thr != null ? [{ label: "High-flow threshold", data: Array(st.dates.length).fill(thr), borderColor: AMBER, borderWidth: 1, borderDash: [2, 3], pointRadius: 0 }] : [])] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 14 } } },
        scales: { x: axisX, y: axisY("Discharge (m³/s)") } } });
    $("#fmsg").innerHTML = `<div class="msg warn">Predictions unavailable: ${esc(fc.reason || "no forecast data")}. Showing observed discharge only (last 365 days up to ${esc(st.last_date)}).</div>`;
  }
  $("#fmeta").innerHTML = meta;
  if ($("#fcCards")) renderFcCards();
}

/* ---------- weather ---------- */
function weatherBlock() {
  return `<div id="wx"><div class="grid3">
    <div><h3>Recent rainfall (30 days)</h3><div class="chartbox sm"><canvas id="cRain"></canvas></div></div>
    <div><h3>Soil moisture (90 days)</h3><div class="chartbox sm"><canvas id="cSoil"></canvas></div></div>
    <div><h3>Annual rainfall (complete years)</h3><div class="chartbox sm"><canvas id="cAnn"></canvas></div></div></div>
    <div id="wxmsg"></div></div>`;
}
function renderWeather() {
  if (!CUR || !$("#wx")) return;
  const w = CUR.wx;
  if (!w.available) {
    $("#wx").innerHTML = `<div class="msg warn">Weather unavailable: ${esc(w.reason)}</div>`;
    return;
  }
  if (!$("#cRain")) { $("#wx").outerHTML = weatherBlock(); }
  const bar = (id, labels, data, label, ytitle) => draw(id, { type: "bar", data: { labels, datasets: [{ label, data, backgroundColor: BLUE }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: axisX, y: axisY(ytitle) } } });
  bar("cRain", w.recent.dates, w.recent.rain_mm, "Rain", "mm per day");
  bar("cAnn", w.annual.years.map(String), w.annual.rain_mm, "Annual rain", "mm per year");
  draw("cSoil", { type: "line", data: { labels: w.soil.dates, datasets: [
      { label: "0–7 cm", data: w.soil.soil1, borderColor: BLUE, borderWidth: 1.5, pointRadius: 0 },
      { label: "7–28 cm (unavailable)", data: [], borderColor: GREY, borderWidth: 1.5, pointRadius: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { x: axisX, y: { ...axisY("m³/m³"), beginAtZero: false } } } });
  $("#wxmsg").innerHTML = `<div class="small" style="margin-top:8px">ERA5-Land, nearest land cell. Last weather date ${esc(w.last_date)}; rain in the last 30 days: ${fmt(w.rain_30d_total_mm)} mm.</div>`;
}

/* ---------- dashboard ---------- */
async function pageDashboard() {
  const s = S.summary, v = $("#view");
  const hv = (S.summary.stations.find(x => x.id === S.sid) || {});
  v.innerHTML = `
  <div class="cards">
    <div class="card"><div class="k">Monitoring stations</div><div class="v">${s.n_stations}</div><div class="n">${s.n_reporting} reported in the last ${s.fresh_days} days</div></div>
    <div class="card"><div class="k">Above high-flow threshold</div><div class="v">${s.above_high}</div><div class="n">of ${s.n_reporting} reporting stations, latest observation</div></div>
    <div class="card"><div class="k">Latest observation</div><div class="v">${esc(s.data_last_date)}</div><div class="n">${s.n_on_last_date} station(s) reported that day</div></div>
    <div class="card"><div class="k">Forecast horizons</div><div class="v">1 · 3 · 7 days</div><div class="n">${s.forecasts_available ? "Forecast file found" : "Forecast file not found"}</div></div>
  </div>
  ${forecastPanel("Forecast: observed vs predicted discharge")}
  <div class="grid2">
    <section class="panel"><div class="panel-head"><h2>Weather at selected station</h2></div>${weatherBlock()}</section>
    <section class="panel"><div class="panel-head"><h2>Risk monitoring</h2></div>
      <div class="risk">${["Normal", "Elevated", "High"].map(k => `<div class="tile ${k}"><div class="c">${s.counts[k]}</div>${k}</div>`).join("")}</div>
      <div class="small">Indicative only. Latest observed discharge compared with each station's own history up to ${esc(s.train_end)}:
      High ≥ ${Math.round(s.high_quantile * 100)}th percentile (same as the backend's high-flow level), Elevated ≥ ${Math.round(s.elevated_quantile * 100)}th percentile (configured in configs/frontend.json).
      Only stations that reported in the last ${s.fresh_days} days are classified; ${s.n_stations - s.n_reporting} are not, and ${s.n_without_threshold} have too little early history for a threshold.</div>
      <div class="scroll" style="margin-top:10px">${riskTable()}</div></section>
  </div>`;
  fillSelect();
  await loadStation();
}
function riskTable() {
  const rows = S.summary.stations.filter(x => x.risk === "High" || x.risk === "Elevated");
  if (!rows.length) return `<div class="small">No reporting station is currently Elevated or High.</div>`;
  return `<table><tr><th>Station</th><th>Latest (m³/s)</th><th>Level</th></tr>${rows.map(r =>
    `<tr><td><a href="#/station/${encodeURIComponent(r.id)}">${esc(r.id)}</a></td><td>${fmt(r.last_q)}</td><td>${badge(r.risk)}</td></tr>`).join("")}</table>`;
}

/* ---------- station page ---------- */
async function pageStation() {
  $("#view").innerHTML = `
  <section class="panel"><div class="panel-head"><h1 id="stTitle">Station details</h1></div><div class="info" id="info"></div></section>
  <section class="panel"><h2 style="margin-bottom:10px">Predictions</h2><div class="grid3" id="fcCards"></div><div id="fcNote"></div></section>
  ${forecastPanel("Actual vs predicted discharge")}
  <section class="panel"><h2 style="margin-bottom:10px">Recent discharge history (last 365 days)</h2><div class="chartbox"><canvas id="cHist"></canvas></div></section>
  <section class="panel"><h2 style="margin-bottom:10px">Weather</h2>${weatherBlock()}</section>`;
  fillSelect();
  await loadStation();
}
function renderStationExtras() {
  const { st, fc } = CUR;
  $("#stTitle").textContent = stName(st);
  $("#info").innerHTML = [
    ["River", st.river || "—"], ["State", st.state || "—"],
    ["Location", st.lat != null ? `${st.lat.toFixed(4)}° N, ${st.lon.toFixed(4)}° E` : "—"],
    ["Record", `${st.first_date} to ${st.last_date} (${fmt(st.n_obs, 0)} days)`],
    ["Latest observation", st.last_q != null ? `${fmt(st.last_q)} m³/s` : "—"],
    ["High-flow threshold", st.high != null ? `${fmt(st.high)} m³/s` : "Unavailable (too little history)"],
    ["Elevated level (configured)", st.elevated != null ? `${fmt(st.elevated)} m³/s` : "—"],
    ["Indicative level", badge(st.risk)]
  ].map(([k, v]) => `<div><span>${k}</span>${v}</div>`).join("");
  renderFcCards();
  draw("cHist", { type: "line", data: { labels: st.dates, datasets: [
      { label: "Observed", data: st.discharge, borderColor: DARK, borderWidth: 1.5, pointRadius: 0, spanGaps: false },
      ...(st.high != null ? [{ label: "High-flow threshold", data: Array(st.dates.length).fill(st.high), borderColor: AMBER, borderWidth: 1, borderDash: [2, 3], pointRadius: 0 }] : [])] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 14 } } }, scales: { x: axisX, y: axisY("Discharge (m³/s)") } } });
}
function renderFcCards() {
  const { fc } = CUR;
  $("#fcCards").innerHTML = [1, 3, 7].map(h => {
    const a = fc.available ? fc.horizons[h] : null;
    return `<div class="fc-card"><div class="small">${h}-day forecast</div>${a
      ? `<div class="v">${fmt(a.last_q_pred)} <span class="small">m³/s</span></div><div class="small">for ${a.last_target_date}, issued ${a.last_issue_date}</div>`
      : `<div class="v">—</div><div class="small">Not available</div>`}</div>`;
  }).join("");
  $("#fcNote").innerHTML = fc.available ? "" : `<div class="msg warn">Predictions unavailable: ${esc(fc.reason)}</div>`;
}

/* ---------- performance ---------- */
async function pagePerf() {
  const v = $("#view");
  v.innerHTML = `<section class="panel"><div class="panel-head"><h1>Model performance</h1><div class="controls">
    <select id="pSplit">${Object.entries(SPLIT_NAMES).map(([k, n]) => `<option value="${k}">${n}</option>`).join("")}</select>
    <div class="seg" id="pSeg">${[1, 3, 7].map(h => `<button data-h="${h}">${h} day${h > 1 ? "s" : ""}</button>`).join("")}</div></div></div>
    <div class="msg"><b>How to read these numbers.</b> NSE (Nash–Sutcliffe efficiency) is <b>not a percentage and not an accuracy score</b>:
    1 is perfect, 0 means no better than always predicting a station's average flow, and negative values are worse than that.
    Here it is the median over stations. RMSE and MAE are errors in m³/s (lower is better). Recall and precision are event-detection rates (shown as %), a different quantity again.</div>
    <div id="pBody"></div></section>`;
  let m;
  try { m = await api("metrics"); } catch (e) { $("#pBody").innerHTML = `<div class="msg err">${esc(e.message)}</div>`; return; }
  if (!m.available) { $("#pBody").innerHTML = `<div class="msg warn">Metrics unavailable: ${esc(m.reason)}</div>`; return; }
  const sel = $("#pSplit"); sel.value = S.perf.split;
  sel.onchange = () => { S.perf.split = sel.value; render(); };
  document.querySelectorAll("#pSeg button").forEach(b => {
    b.classList.toggle("on", +b.dataset.h === S.perf.horizon);
    b.onclick = () => { S.perf.horizon = +b.dataset.h; document.querySelectorAll("#pSeg button").forEach(x => x.classList.toggle("on", x === b)); render(); };
  });
  const order = ["lightgbm", "lstm", "persistence", "climatology"];
  function render() {
    const rows = m.rows.filter(r => r.split === S.perf.split && r.horizon === S.perf.horizon)
      .sort((a, b) => order.indexOf(a.model) - order.indexOf(b.model));
    if (!rows.length) { $("#pBody").innerHTML = `<div class="msg warn">No results for this split and horizon.</div>`; return; }
    const best = (k, lo) => { const vals = rows.map(r => r[k]).filter(x => x != null); return vals.length ? (lo ? Math.min(...vals) : Math.max(...vals)) : null; };
    const cell = (r, k, f, lo) => `<td class="${r[k] != null && r[k] === best(k, lo) ? "best" : ""}">${f(r[k])}</td>`;
    const nm = r => MODEL_NAMES[r.model] || r.model;
    const main = `<h3 style="margin-top:16px">Forecast error, ${S.perf.horizon}-day horizon · ${SPLIT_NAMES[S.perf.split]}</h3><div class="scroll"><table>
      <tr><th>Model</th><th>NSE (median station)</th><th>RMSE (m³/s)</th><th>MAE (m³/s)</th><th>High-flow RMSE (m³/s)</th><th>High-flow recall</th><th>High-flow precision</th><th>Test rows</th></tr>
      ${rows.map(r => `<tr><td>${nm(r)}</td>${cell(r, "median_station_nse", x => fmt(x, 3), false)}${cell(r, "rmse", x => fmt(x), true)}${cell(r, "mae", x => fmt(x), true)}${cell(r, "hi_rmse", x => fmt(x), true)}${cell(r, "hi_recall", pct, false)}${cell(r, "hi_precision", pct, false)}<td>${fmt(r.n, 0)}</td></tr>`).join("")}</table></div>`;
    const onset = `<h3 style="margin-top:18px">Flood-onset detection</h3><div class="small" style="margin-bottom:8px">Onset = the flow is below the high-flow level today and at or above it at the forecast horizon. Recall = share of onsets that were forecast; precision = share of onset alarms that were real.</div>
      <div class="scroll"><table><tr><th>Model</th><th>Onset events</th><th>Recall</th><th>Alarms raised</th><th>Precision</th></tr>
      ${rows.map(r => `<tr><td>${nm(r)}</td><td>${fmt(r.onset_n, 0)}</td>${cell(r, "onset_recall", pct, false)}<td>${fmt(r.onset_alarms, 0)}</td>${cell(r, "onset_precision", pct, false)}</tr>`).join("")}</table></div>`;
    const notes = [];
    if (!m.models.includes("lstm")) notes.push("LSTM results were not found in the configured metrics files, so only LightGBM and the two baselines are compared. Add the LSTM metrics CSV (same columns) to <code>metrics_files</code> in configs/frontend.json and it will appear here.");
    $("#pBody").innerHTML = main + onset + notes.map(n => `<div class="msg warn">${n}</div>`).join("") + `<div class="small" style="margin-top:10px">Best value in each column is highlighted. Results are copied from the backend's evaluation files, not recomputed.</div>`;
  }
  render();
}

/* ---------- router ---------- */
async function route() {
  const h = location.hash.replace(/^#\/?/, "").split("/");
  document.querySelectorAll("nav a").forEach(a => a.classList.remove("on"));
  try {
    if (!S.summary) {
      S.summary = await api("summary");
      const st = S.summary.stations;
      S.sid = (st.find(x => x.fresh) || st[0] || {}).id;
      $("#updated").innerHTML = `Latest observation: ${esc(S.summary.data_last_date)}<br>Data file updated: ${esc(S.summary.file_updated)}`;
    }
    if (h[0] === "performance") { $("[data-nav=perf]").classList.add("on"); await pagePerf(); }
    else if (h[0] === "station") {
      if (h[1]) S.sid = decodeURIComponent(h[1]);
      $("[data-nav=station]").classList.add("on"); await pageStation();
    } else { $("[data-nav=dash]").classList.add("on"); await pageDashboard(); }
  } catch (e) {
    $("#view").innerHTML = `<div class="msg err">Could not load data from the backend: ${esc(e.message)}. Is <code>python dashboard/server.py</code> running?</div>`;
  }
}
window.addEventListener("hashchange", route);
route();
