"use strict";
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLORS = () => ({ ink: css("--ink"), muted: css("--muted"), rule: css("--rule"), alice: css("--alice"),
  bob: css("--bob"), eve: css("--eve"), card: css("--card"), lock: css("--lock") });
const MODE_NAMES = { conventional: "Conventional", paper: "Paper 1-h" };
const HINTS = {
  conventional: "Every update moves a weight by 1. Slowest, the paper's baseline.",
  paper: "Odd rounds move weights by hop, even rounds by 1, fixed in advance (Algorithm 1)."
};
let mode = "paper";

/* ---------- mode ---------- */
function setMode(m) {
  mode = m;
  document.querySelectorAll("#modeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  $("modeHint").textContent = HINTS[m];
}
document.querySelectorAll("#modeSeg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
setMode("paper");

const FIELDS = ["K", "N", "L", "B", "MAX", "hop", "ecc_modulus", "count_mode",
  "attack", "M", "seed", "input_mode", "limit", "P"];
function params(extra = {}) {
  const p = { mode };
  FIELDS.forEach((f) => p[f] = $(f).value);
  return Object.assign(p, extra);
}
const qs = (o) => new URLSearchParams(o).toString();
const fmt = (n, d = 0) => Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (x, d = 1) => (x == null ? "n/a" : fmt(100 * x, d) + "%");

/* ---------- canvas helpers ---------- */
function fitCanvas(c, h) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth;
  const H = h || c.height / (c._dpr || 1);
  c._dpr = dpr; c.width = w * dpr; c.height = H * dpr; c.style.height = H + "px";
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h: H };
}
function drawLines(c, series, o) {
  const C = COLORS(), { g, w, h } = fitCanvas(c);
  const pad = { l: 44, r: 10, t: 10, b: 28 };
  g.clearRect(0, 0, w, h);
  let xmax = 1;
  series.forEach((s) => s.data.forEach((p) => xmax = Math.max(xmax, p[0])));
  const X = (x) => pad.l + (x / xmax) * (w - pad.l - pad.r);
  const Y = (y) => pad.t + (1 - (y - o.ymin) / (o.ymax - o.ymin)) * (h - pad.t - pad.b);
  g.font = "11px " + getComputedStyle(document.body).fontFamily; g.fillStyle = C.muted; g.strokeStyle = C.rule; g.lineWidth = 1;
  o.yticks.forEach((t) => {
    g.beginPath(); g.moveTo(pad.l, Y(t)); g.lineTo(w - pad.r, Y(t)); g.stroke();
    g.textAlign = "right"; g.fillText(o.yfmt ? o.yfmt(t) : t, pad.l - 6, Y(t) + 4);
  });
  g.textAlign = "left"; g.fillText("0", pad.l, h - 8);
  g.textAlign = "right"; g.fillText(fmt(xmax) + " " + o.xlabel, w - pad.r, h - 8);
  series.forEach((s) => {
    if (!s.data.length) return;
    g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []); g.beginPath();
    s.data.forEach((p, i) => {
      const x = X(p[0]), y = Y(p[1]);
      if (i && s.step) g.lineTo(x, Y(s.data[i - 1][1]));
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke(); g.setLineDash([]);
  });
}

/* ---------- live exchange ---------- */
let es = null, live = null, raf = 0;

function drawGrid() {
  const c = $("grid"), C = COLORS();
  const K = live ? live.K : +$("K").value, N = live ? live.N : +$("N").value;
  const P = live ? live.P : +$("P").value, rows = P * K;
  const rowH = Math.max(8, Math.min(34, 150 / rows)), gap = 4, pairGap = P > 1 ? 14 : gap;
  const H = rows * rowH + (rows - P) * gap + (P - 1) * pairGap;
  const { g, w, h } = fitCanvas(c, H);
  g.clearRect(0, 0, w, h);
  const cw = w / N, agree = live && live.last ? live.last.agree : "";
  let y = 0;
  for (let r = 0; r < rows; r++) {
    g.fillStyle = C.rule; g.fillRect(0, y, w, rowH);
    g.fillStyle = C.lock;
    for (let n = 0; n < N; n++) {
      if (agree && agree[r * N + n] === "1") g.fillRect(n * cw, y, Math.max(cw - (cw > 4 ? 1 : 0), 0.6), rowH);
    }
    y += rowH + ((r + 1) % K === 0 ? pairGap : gap);
  }
}

function render() {
  raf = 0;
  if (!live) return;
  const s = live.snaps, last = live.last;
  if (last) {
    $("gComm").textContent = fmt(last.comm_rounds);
    $("gMatch").textContent = pct(last.match_rate, 1);
    $("gHop").textContent = last.hop;
    $("gEve").textContent = last.eve_best == null ? "n/a" : pct(last.eve_best, 1);
  }
  drawGrid();
  const C = COLORS();
  const ser = [
    { data: s.map((d) => [d.comm_rounds, d.rho_true]), color: C.alice },
  ];
  if (s.length && s[0].eve_best != null)
    ser.push({ data: s.map((d) => [d.comm_rounds, d.eve_best]), color: C.eve, width: 1.5 });
  drawLines($("chartOverlap"), ser,
    { ymin: 0, ymax: 1, yticks: [0, .25, .5, .75, 1], xlabel: "rounds", yfmt: (t) => t.toFixed(2) });
  const hmax = Math.max(1, ...s.map((d) => d.hop));
  drawLines($("chartHop"), [{ data: s.map((d) => [d.comm_rounds, d.hop]), color: C.bob, step: true }],
    { ymin: 0, ymax: hmax + 1, yticks: Array.from({ length: hmax + 2 }, (_, i) => i), xlabel: "rounds" });
}
const schedule = () => { if (!raf) raf = requestAnimationFrame(render); };

$("runBtn").addEventListener("click", () => {
  if (es) {
    es.close(); es = null;
    $("runBtn").textContent = "Start exchange";
    $("liveStatus").textContent = "Stopped.";
    return;
  }
  const p = params({ every: +$("N").value * +$("K").value > 900 ? 5 : 2 });
  live = { K: +p.K, N: +p.N, P: +p.P, snaps: [], last: null };
  $("liveResult").hidden = true;
  $("runBtn").textContent = "Stop";
  $("liveStatus").textContent = `Synchronising with the ${MODE_NAMES[mode]} schedule...`;
  es = new EventSource("/api/run?" + qs(p));
  const t0 = Date.now();
  es.addEventListener("snap", (e) => {
    const d = JSON.parse(e.data);
    if (d.agree === undefined && live.last) d.agree = live.last.agree;   // grid is sent less often
    live.snaps.push(d); live.last = d; schedule();
    $("liveStatus").textContent = `Synchronising with the ${MODE_NAMES[mode]} schedule: ${fmt(d.comm_rounds)} rounds, `
      + (live.P > 1 ? `${d.pairs_done} of ${live.P} pairs done, ` : "")
      + `${fmt((Date.now() - t0) / 1000, 0)} s. Press Stop to end early.`;
  });
  es.addEventListener("done", (e) => {
    es.close(); es = null; $("runBtn").textContent = "Start exchange";
    const r = JSON.parse(e.data);
    if (r.error) { $("liveStatus").textContent = "The run failed: " + r.error; return; }
    if (live.last) {
      live.last.agree = "1".repeat(Math.round(r.match_post_ecc * live.P * live.K * live.N)).padEnd(live.P * live.K * live.N, "0");
    }
    $("gMatch").textContent = pct(r.match_post_ecc, 1);
    $("gComm").textContent = fmt(r.comm_rounds);
    if (r.eve_best != null) $("gEve").textContent = pct(r.eve_best, 1);
    schedule();
    showResult(r);
    $("liveStatus").textContent = r.keys_match ? "Keys agree." : r.timed_out
      ? "Stopped at the round limit before the machines synchronised. Raise the round limit or use a smaller L."
      : "Keys differ. Try a larger MAX or reconciliation modulus.";
  });
  es.onerror = () => {
    if (es) { es.close(); es = null; }
    $("runBtn").textContent = "Start exchange";
    $("liveStatus").textContent = "Lost the connection to the server. Check that server.py is running.";
  };
});

function showResult(r) {
  const el = $("liveResult");
  el.hidden = false;
  el.classList.toggle("bad", !r.keys_match || r.eve_success);
  const hops = Object.entries(r.hop_usage).map(([k, v]) => `step ${k}: ${fmt(v)}`).join(", ");
  const rows = [
    ["Keys", r.keys_match ? "Alice and Bob derived the same key" : "Keys differ"],
    ["Key (Alice)", r.key_a],
    ["Communication rounds", fmt(r.comm_rounds) + (r.timed_out ? " (hit the round limit)" : "")],
    ["Learning rounds / updates", `${fmt(r.learning_rounds)} / ${fmt(r.updates)}`],
    ["Step usage", hops || "none"],
    ["Bits on the wire", `${fmt(r.bits_on_wire)} (Alice + Bob, excluding reconciliation)`],
    ["Equal weights before / after reconciliation", `${pct(r.match_pre_ecc, 2)} / ${pct(r.match_post_ecc, 2)}`],
    ["Reconciliation leak", `${fmt(r.leak_bits)} bits leaked, ${fmt(r.residual_entropy_bits)} bits left`
      + (r.residual_entropy_bits < 384 ? " (too few for a 256-bit key; raise N or L)" : "")],
  ];
  if (r.P > 1)
    rows.splice(3, 0, ["Parallel pairs", `${r.P} pairs, finished at rounds ${r.pair_finish_rounds.map((x) => x == null ? "not finished" : fmt(x)).join(", ")}`]);
  if (r.eve_best != null) {
    const per = r.eve_per_pair.map((b, i) => `pair ${i + 1}: ${pct(b, 1)}`).join(", ");
    const verdict = r.eve_success ? "every pair reached 98%, attack succeeded"
      : `${r.eve_pairs_broken} of ${r.P} pairs broken, attack failed (needs all ${r.P})`;
    rows.push(["Attacker", (r.P > 1 ? per + "; " : `best agreement ${pct(r.eve_best, 1)}, `) + verdict]);
  }
  el.innerHTML = `<h2>${r.keys_match && !r.eve_success ? "Exchange complete" : r.keys_match ? "Keys agree, but the attacker kept up" : "Exchange did not converge"}</h2><dl>`
    + rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("") + "</dl>";
}

/* ---------- init ---------- */
["K", "N"].forEach((id) => $(id).addEventListener("change", () => { if (!es) { live = null; drawGrid(); } }));
addEventListener("resize", () => { drawGrid(); schedule(); });
live = null;
drawGrid();