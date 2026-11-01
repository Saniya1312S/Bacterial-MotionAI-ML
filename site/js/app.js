/* ═══════════════════════════════════════════════════════════════════════════
   BacMotionAI Dashboard — app.js
   Loads exported JSON, renders charts, controls trajectory explorer.
   No external libraries.
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  const DATA = {};
  const COLORS = {
    none: "#3b82f6", attractant: "#10b981",
    repellent: "#fb7185", antibiotic: "#a78bfa",
  };

  /* ── Bootstrap ────────────────────────────────────────────────────────── */
  async function init() {
    try {
      const names = ["overview", "model_metrics", "class_metrics", "error_analysis", "trajectories"];
      const results = await Promise.all(
        names.map(n => fetch("data/" + n + ".json").then(r => { if (!r.ok) throw n; return r.json(); }))
      );
      names.forEach((n, i) => (DATA[n] = results[i]));
    } catch (e) {
      console.error("Failed to load dashboard data:", e);
      document.getElementById("heroStats").innerHTML =
        '<p style="color:var(--danger)">Could not load dashboard data. Serve this directory with a local HTTP server.</p>';
      return;
    }
    renderHeroStats();
    renderDataset();
    renderModelChart("test_acc");
    renderConfusionMatrix();
    renderClassTable();
    initMetricSelector();
    initExplorer();
    initInsightTabs();
    initNav();
    initHeroBg();
  }

  /* ── Hero Stats ───────────────────────────────────────────────────────── */
  function renderHeroStats() {
    const o = DATA.overview;
    const best = DATA.model_metrics.models[0]; // sorted by cv_macro_f1 desc
    const el = document.getElementById("heroStats");
    el.innerHTML = [
      stat((best.test_acc * 100).toFixed(1) + "%", best.name + " test accuracy"),
      stat(best.roc_auc.toFixed(3), "ROC-AUC (macro)"),
      stat(o.rows.toLocaleString(), "Trajectories"),
      stat(o.feature_count, "Features"),
      stat(o.classes.length, "Classes"),
    ].join("");
  }
  function stat(val, lbl) {
    return '<div class="hero-stat"><span class="hero-stat-val">' + val +
           '</span><span class="hero-stat-lbl">' + lbl + '</span></div>';
  }

  /* ── Dataset Explorer ─────────────────────────────────────────────────── */
  function renderDataset() {
    const o = DATA.overview;
    // Pills
    document.getElementById("datasetPills").innerHTML = [
      pill(o.rows.toLocaleString(), "Samples"),
      pill(o.feature_count, "Features"),
      pill(o.classes.length, "Classes"),
      pill(o.channel_widths.join(", "), "Widths (um)"),
      pill(o.split, "Split"),
    ].join("");
    // Class bars
    const maxCount = Math.max(...Object.values(o.class_counts));
    const barsEl = document.getElementById("classBars");
    barsEl.innerHTML = o.classes.map(c => {
      const count = o.class_counts[c] || 0;
      const pct = (count / maxCount * 100).toFixed(0);
      return '<div class="class-bar-row"><span class="class-bar-label">' + c +
        '</span><div class="class-bar-track"><div class="class-bar-fill" data-cond="' + c +
        '" style="width:' + pct + '%"></div></div><span class="class-bar-val">' + count + '</span></div>';
    }).join("");
    // Feature table
    const tbody = document.getElementById("featureTableBody");
    tbody.innerHTML = o.features.map(f => {
      const s = o.feature_stats[f];
      const desc = o.feature_descriptions[f] || "";
      return "<tr><td>" + f + "</td><td>" + desc + "</td><td>" +
        s.mean.toFixed(3) + "</td><td>" + s.std.toFixed(3) + "</td><td>" +
        s.min.toFixed(3) + "</td><td>" + s.max.toFixed(3) + "</td></tr>";
    }).join("");
  }
  function pill(val, lbl) {
    return '<div class="stat-pill"><span class="stat-pill-val">' + val +
           '</span><span class="stat-pill-lbl">' + lbl + '</span></div>';
  }

  /* ── Model Lab ────────────────────────────────────────────────────────── */
  const METRIC_INFO = {
    test_acc: "Test accuracy — fraction of correctly classified test samples. Higher is better.",
    test_macro_f1: "Macro F1 — harmonic mean of precision and recall, averaged equally across classes.",
    roc_auc: "ROC-AUC (macro, one-vs-rest) — area under the ROC curve. 1.0 is perfect.",
    cv_macro_f1: "Cross-validation macro-F1 — 5-fold stratified CV score on training data only.",
    train_test_gap: "Overfit gap — difference between train and test accuracy. Lower is better.",
  };

  function renderModelChart(metric) {
    const mm = DATA.model_metrics;
    const container = document.getElementById("modelChart");
    const bestModel = mm.best_model.best_model;
    const isGap = metric === "train_test_gap";

    // Compute scale
    let maxVal = 0;
    mm.models.forEach(m => { if (m[metric] > maxVal) maxVal = m[metric]; });
    // Add baseline for accuracy
    let baselineVal = null;
    if (metric === "test_acc") {
      const speedBase = mm.baselines.find(b => b.name.includes("speed features"));
      if (speedBase) baselineVal = speedBase.accuracy;
    }
    if (metric === "test_macro_f1") {
      const speedBase = mm.baselines.find(b => b.name.includes("speed features"));
      if (speedBase) baselineVal = speedBase.macro_f1;
    }

    const scaleMax = isGap ? Math.max(maxVal * 1.3, 0.15) : 1.0;

    // CI keys
    const ciMap = {
      test_acc: ["test_acc_ci", "test_acc"],
      test_macro_f1: ["test_f1_ci", "test_macro_f1"],
    };

    let html = "";
    mm.models.forEach((m, rowIdx) => {
      const val = m[metric];
      const pct = (val / scaleMax * 100).toFixed(1);
      const isBest = m.name === bestModel;
      let ciHtml = "";
      if (ciMap[metric]) {
        const ci = m[ciMap[metric][0]];
        const v = m[ciMap[metric][1]];
        const ciLeftPct = (ci[0] / scaleMax * 100).toFixed(1);
        const ciRightPct = (ci[1] / scaleMax * 100).toFixed(1);
        const ciWidthPct = (ciRightPct - ciLeftPct).toFixed(1);
        ciHtml = '<div class="bar-ci" style="left:' + ciLeftPct + '%;width:' + ciWidthPct + '%"></div>';
      }

      html += '<div class="bar-row">' +
        '<div class="bar-label">' + m.name + (isBest ? " *" : "") + '</div>' +
        '<div class="bar-track">' +
          '<div class="bar-fill' + (isBest ? " best" : "") + '" style="width:' + pct + '%"></div>' + ciHtml +
          (baselineVal ? '<div class="baseline-line" style="left:' + (baselineVal / scaleMax * 100).toFixed(1) + '%">' + (rowIdx === 0 ? '<span class="baseline-tag">speed-only baseline</span>' : '') + '</div>' : '') +
        '</div>' +
        '<div class="bar-value">' + (isGap ? (val * 100).toFixed(1) + "%" : val.toFixed(3)) + '</div></div>';
    });
    if (!isGap) {
      // Baselines
      mm.baselines.forEach(b => {
        const val = metric === "test_acc" ? b.accuracy : b.macro_f1;
        if (metric === "roc_auc" || metric === "cv_macro_f1") return;
        const pct = (val / scaleMax * 100).toFixed(1);
        html += '<div class="bar-row" style="opacity:.5">' +
          '<div class="bar-label" style="font-size:.75rem">' + b.name + '</div>' +
          '<div class="bar-track"><div class="bar-fill" style="width:' + pct + '%;background:var(--muted)"></div></div>' +
          '<div class="bar-value">' + val.toFixed(3) + '</div></div>';
      });
    }
    container.innerHTML = html;
    document.getElementById("metricExplanation").textContent =
      METRIC_INFO[metric] + " * = selected by CV.";
  }

  function initMetricSelector() {
    document.getElementById("metricSelector").addEventListener("click", function (e) {
      const btn = e.target.closest(".metric-btn");
      if (!btn) return;
      this.querySelectorAll(".metric-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      renderModelChart(btn.dataset.metric);
    });
  }

  /* ── Confusion Matrix ─────────────────────────────────────────────────── */
  function renderConfusionMatrix() {
    const cm = DATA.class_metrics.confusion_matrix;
    const labels = cm.labels;
    const matrix = cm.matrix;
    const maxVal = Math.max(...matrix.flat());
    const n = labels.length;

    let html = '<div style="overflow-x:auto">';
    html += '<div class="cm-grid" style="grid-template-columns:30px repeat(' + n + ', 1fr); grid-template-rows: auto repeat(' + n + ', 1fr); max-width:' + (30 + n * 73) + 'px; margin:0 auto">';
    // Corner
    html += '<div></div>';
    // Column headers
    labels.forEach(l => { html += '<div class="cm-label col-label">' + l + '</div>'; });
    // Rows
    for (let i = 0; i < n; i++) {
      html += '<div class="cm-label row-label">' + labels[i] + '</div>';
      for (let j = 0; j < n; j++) {
        const val = matrix[i][j];
        const intensity = val / maxVal;
        const bg = i === j
          ? "rgba(45,212,191," + (0.15 + intensity * 0.65) + ")"
          : "rgba(251,113,133," + (intensity * 0.5) + ")";
        html += '<div class="cm-cell" style="background:' + bg + '" title="True: ' + labels[i] + ', Predicted: ' + labels[j] + '">' + val + '</div>';
      }
    }
    html += '</div>';
    html += '<p class="cm-axis-label mt-1" style="text-align:center">Predicted &rarr;</p>';
    html += '</div>';
    document.getElementById("confusionMatrix").innerHTML = html;
  }

  /* ── Class Table ──────────────────────────────────────────────────────── */
  function renderClassTable() {
    const pc = DATA.class_metrics.per_class;
    document.getElementById("classTableBody").innerHTML = pc.map(c => {
      return "<tr><td>" + c.class + '</td><td class="num">' + c.precision.toFixed(2) +
        '</td><td class="num">' + c.recall.toFixed(2) +
        '</td><td class="num">' + c.f1.toFixed(2) + "</td></tr>";
    }).join("");
  }

  /* ── Trajectory Explorer ──────────────────────────────────────────────── */
  function initExplorer() {
    const tData = DATA.trajectories;
    // Populate width selector
    const widthSel = document.getElementById("selWidth");
    tData.available_widths.forEach(w => {
      const opt = document.createElement("option");
      opt.value = w; opt.textContent = w + " um";
      if (w === 50) opt.selected = true;
      widthSel.appendChild(opt);
    });

    const selChem = document.getElementById("selChem");
    const selHydro = document.getElementById("selHydro");
    const selSeed = document.getElementById("selSeed");
    const seedVal = document.getElementById("seedVal");

    let current = null;
    let resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        if (current) drawTrajectory(current.traj, current.width);
      }, 150);
    });

    function update() {
      seedVal.textContent = (parseInt(selSeed.value) + 1);
      const chem = selChem.value;
      const width = parseInt(widthSel.value);
      const hydro = selHydro.value === "true";
      const si = parseInt(selSeed.value);

      const traj = tData.trajectories.find(t =>
        t.chemical === chem && t.channel_width === width &&
        t.hydro === hydro && t.seed_index === si
      );

      if (!traj) {
        document.getElementById("featBars").innerHTML = '<div class="empty-state"><div class="empty-icon">&#128683;</div>No precomputed trajectory for this combination.</div>';
        document.getElementById("predDisplay").innerHTML = "";
        return;
      }
      current = { traj: traj, width: width };
      drawTrajectory(traj, width);
      renderFeatures(traj.features);
      renderPrediction(traj);
    }

    [selChem, widthSel, selHydro, selSeed].forEach(el => el.addEventListener("change", update));
    selSeed.addEventListener("input", update);
    document.getElementById("btnReset").addEventListener("click", function () {
      selChem.value = "attractant"; widthSel.value = "50";
      selHydro.value = "false"; selSeed.value = "0";
      update();
    });
    update();
  }

  function drawTrajectory(traj, W) {
    const canvas = document.getElementById("trajCanvas");
    const wrap = canvas.parentElement;
    canvas.width = wrap.clientWidth * (window.devicePixelRatio || 1);
    canvas.height = wrap.clientHeight * (window.devicePixelRatio || 1);
    canvas.style.width = wrap.clientWidth + "px";
    canvas.style.height = wrap.clientHeight + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    const cw = wrap.clientWidth, ch = wrap.clientHeight;

    ctx.fillStyle = "#121e31";
    ctx.fillRect(0, 0, cw, ch);

    const xs = traj.xs, ys = traj.ys;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const pad = 40;
    const plotW = cw - 2 * pad, plotH = ch - 2 * pad;
    const xRange = Math.max(xMax - xMin, 1);

    function tx(x) { return pad + ((x - xMin) / xRange) * plotW; }
    function ty(y) { return pad + plotH - (y / W) * plotH; }

    // Grid
    ctx.strokeStyle = "#1a2a40"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const gy = pad + (g / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(cw - pad, gy); ctx.stroke();
    }

    // Walls
    ctx.strokeStyle = "#475569"; ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.beginPath(); ctx.moveTo(pad, ty(0)); ctx.lineTo(cw - pad, ty(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, ty(W)); ctx.lineTo(cw - pad, ty(W)); ctx.stroke();
    ctx.setLineDash([]);

    // Wall labels
    ctx.fillStyle = "#64748b"; ctx.font = "11px " + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = "right";
    ctx.fillText("y=0", pad - 6, ty(0) + 4);
    ctx.fillText("y=" + W, pad - 6, ty(W) + 4);

    // Trajectory
    const color = COLORS[traj.chemical] || "#3b82f6";
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = tx(xs[i]), py = ty(ys[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Start/end markers
    ctx.fillStyle = "#10b981";
    ctx.beginPath(); ctx.arc(tx(xs[0]), ty(ys[0]), 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#ef4444";
    ctx.beginPath(); ctx.arc(tx(xs[xs.length - 1]), ty(ys[ys.length - 1]), 5, 0, Math.PI * 2); ctx.fill();

    // Legend
    const legend = document.getElementById("trajLegend");
    legend.innerHTML =
      '<div class="traj-legend-row"><div class="traj-legend-dot" style="background:#10b981"></div> Start</div>' +
      '<div class="traj-legend-row"><div class="traj-legend-dot" style="background:#ef4444"></div> End</div>' +
      '<div class="traj-legend-row"><div class="traj-legend-dot" style="background:' + color + '"></div> ' + traj.chemical + '</div>';
  }

  function renderFeatures(feats) {
    const maxVals = {
      mean_speed: 30, std_speed: 2, cv_speed: 0.1, straightness: 1,
      mean_turn_angle: 0.15, turn_bias: 0.04, persistence_1s: 1,
      reorient_count: 10, mean_alignment: 1, near_wall_frac: 1, wall_contact_frac: 0.5,
    };
    const el = document.getElementById("featBars");
    el.innerHTML = Object.entries(feats).map(function (kv) {
      var k = kv[0], v = kv[1];
      var pct = Math.min(100, Math.abs(v) / (maxVals[k] || 30) * 100);
      return '<div class="feat-row"><span class="feat-name">' + k +
        '</span><div class="feat-track"><div class="feat-fill" style="width:' + pct.toFixed(0) +
        '%"></div></div><span class="feat-val">' + v.toFixed(3) + '</span></div>';
    }).join("");
  }

  function renderPrediction(traj) {
    const el = document.getElementById("predDisplay");
    const correct = traj.correct;
    const badge = correct
      ? '<span class="correct-badge">Correct</span>'
      : '<span class="wrong-badge">Wrong</span>';
    let html = '<div class="pred-result" style="color:' + (COLORS[traj.prediction] || "#e2e8f0") + '">' +
      traj.prediction + ' ' + badge + '</div>';
    html += '<p style="font-size:.78rem; color:var(--muted); margin-bottom:.75rem;">True: <strong>' +
      traj.chemical + '</strong></p>';

    // Probability bars
    const sorted = Object.entries(traj.probabilities).sort(function (a, b) { return b[1] - a[1]; });
    sorted.forEach(function (kv) {
      var cls = kv[0], prob = kv[1];
      var pct = (prob * 100).toFixed(0);
      html += '<div class="pred-row"><span class="pred-lbl">' + cls +
        '</span><div class="pred-track"><div class="pred-fill" data-cls="' + cls +
        '" style="width:' + pct + '%"></div></div><span class="pred-pct">' + pct + '%</span></div>';
    });
    el.innerHTML = html;
  }

  /* ── Insight Tabs ─────────────────────────────────────────────────────── */
  function initInsightTabs() {
    document.getElementById("insightTabs").addEventListener("click", function (e) {
      var btn = e.target.closest(".insight-tab");
      if (!btn) return;
      this.querySelectorAll(".insight-tab").forEach(function (b) {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      document.querySelectorAll(".insight-panel").forEach(function (p) { p.classList.remove("active"); });
      var panel = document.getElementById(btn.dataset.panel);
      if (panel) panel.classList.add("active");
    });
  }

  /* ── Navigation ───────────────────────────────────────────────────────── */
  function initNav() {
    // Mobile toggle
    var toggle = document.querySelector(".nav-toggle");
    var links = document.querySelector(".nav-links");
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open);
    });
    // Close on link click
    links.addEventListener("click", function (e) {
      if (e.target.tagName === "A") {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });

    // Scroll spy
    var sections = document.querySelectorAll("section[id]");
    var navAs = document.querySelectorAll(".nav-links a");
    var ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var scrollY = window.scrollY + 80;
        var current = "";
        sections.forEach(function (s) {
          if (s.offsetTop <= scrollY) current = s.id;
        });
        navAs.forEach(function (a) {
          a.classList.toggle("active", a.getAttribute("href") === "#" + current);
        });
        ticking = false;
      });
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  /* ── Hero Background Animation ────────────────────────────────────────── */
  function initHeroBg() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    var canvas = document.getElementById("heroBg");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var particles = [];

    function resize() {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      particles = [];
      for (var i = 0; i < 50; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 1.2,
          vy: (Math.random() - 0.5) * 1.2,
          r: 1.5 + Math.random() * 2,
          trail: [],
        });
      }
    }
    resize();
    window.addEventListener("resize", resize);

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(function (p) {
        p.trail.push({ x: p.x, y: p.y });
        if (p.trail.length > 12) p.trail.shift();
        p.x += p.vx; p.y += p.vy;
        // Tumble occasionally
        if (Math.random() < 0.005) {
          p.vx = (Math.random() - 0.5) * 1.5;
          p.vy = (Math.random() - 0.5) * 1.5;
        }
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        // Trail
        if (p.trail.length > 1) {
          ctx.beginPath();
          ctx.moveTo(p.trail[0].x, p.trail[0].y);
          p.trail.forEach(function (pt) { ctx.lineTo(pt.x, pt.y); });
          ctx.strokeStyle = "rgba(45,212,191,0.15)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        // Dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(45,212,191,0.5)";
        ctx.fill();
      });
      requestAnimationFrame(animate);
    }
    animate();
  }

  /* ── Go ───────────────────────────────────────────────────────────────── */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
