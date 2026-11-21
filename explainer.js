// =====================================================================
// explainer.js: the interactive optimizer explainer as a reusable widget.
//
//   createExplainer(container, config)
//
// Each call builds one widget inside `container`, with its own canvas,
// controls, optimizer state, play timer, arrows, legend and values
// panel, so several widgets on one page never affect each other (except
// that pressing Play in one pauses any other that is playing).
//
// config (every field optional):
//   size:     "full"  - fills the window height: controls with the values
//                       panel under them on the left, and the plot filling
//                       all the space to the right
//             "small" - square plot as wide as its container allows (up
//                       to 520px, and short enough to fit the window),
//                       controls in a compact row under it, values below
//             "stage" - fills its container (which must have a size, e.g. a
//                       full-height panel); the plot uses the whole area,
//                       with controls and values in floating cards on the
//                       right. Use setVisible() to show one of several.
//   controls: which controls to show, any of
//             "optimizers"        checkboxes (without them, only the
//                                 checked optimizers get a row)
//             "learningRates"     a slider per optimizer
//             "sameLearningRate"  the "Same learning rate for all" box
//             "play", "step", "reset" (or "playStep" for all three);
//                                 the step counter shows with any of them
//             "rotation", "elongation", "speed"
//             (default: all). Hidden controls keep their preset value.
//   values:   false hides the values panel (default true)
//   preset:   {
//     elongation:    20 | 50 | 100            (default 20)
//     rotation:      degrees, 0 to 90         (default 0)
//     start:         [x1, x2] in the bowl's own frame; it is rotated with
//                    the bowl                 (default [-1.5, 0.5])
//     optimizers:    ids that start checked   (default ["gd"])
//     learningRates: { gd, momentum, adam }, any subset; the rest use the
//                    defaults for the elongation
//     speed:         "normal" | "fast"        (default "normal")
//   }
//
// The math (loss, gradient, optimizers) is in optimizers.js; this file
// only draws and handles the controls.
// =====================================================================
import {
  OPTIMIZERS, OPTIMIZER_IDS, loss, gradient, rotateStart, defaultLearningRates,
  isDiverged, STOP_LOSS, DIVERGE_LOSS, MAX_STEPS, DEFAULT_START
} from "./optimizers.js";

// ---------------------------------------------------------------------
// How each optimizer is shown: name, color, arrows, legend, value rows.
// ---------------------------------------------------------------------
const COLORS = { gd: "#5b9cff", momentum: "#ff9f43", adam: "#3ddc84" };
const ARROW_NEW_GRADIENT = "#ff6b9d";  // Momentum: gradient push
const ARROW_CARRIED_OVER = "#4dd8f0";  // Momentum: carried over
const ARROW_ADAM_GRADIENT = "#ff6b9d"; // Adam: gradient direction
const ARROW_ADAM_AVERAGED = "#ffe066"; // Adam: averaged direction
const ARROW_ADAM_STEP = COLORS.adam;   // Adam: step taken

// Plot colors.
const PLOT_BG = "#171a21";
// Contours: a soft lavender gray, brightest near the minimum and fading
// outward (alpha goes from CONTOUR_ALPHA_INNER to CONTOUR_ALPHA_OUTER).
const CONTOUR_RGB = "205, 198, 240";
const CONTOUR_ALPHA_INNER = 0.42, CONTOUR_ALPHA_OUTER = 0.10;
const PLOT_BG_CENTER = "#1c2030";  // slightly lighter middle of the plot
const AXIS_COLOR = "rgba(230, 232, 238, 0.22)";
const AXIS_LABEL_COLOR = "#9aa3b2";
const MARKER_COLOR = "#e6e8ee";

// Adam's three arrows have very different true lengths, so each is drawn
// at the same fixed length (direction only). When -g and -m_hat point
// almost the same way (as on step 1), -g is drawn thicker so it stays
// visible underneath.
function adamArrows(r, g) {
  const gLen = Math.hypot(g[0], g[1]), mLen = Math.hypot(r.mHat[0], r.mHat[1]);
  const cosAngle = gLen > 0 && mLen > 0 ? (g[0] * r.mHat[0] + g[1] * r.mHat[1]) / (gLen * mLen) : 0;
  const overlap = cosAngle > Math.cos(3 * Math.PI / 180);  // within 3 degrees
  return [
    { vec: [-g[0], -g[1]], color: ARROW_ADAM_GRADIENT, fixedLength: true,
      width: overlap ? 6.5 : 4, head: overlap ? 19 : 14,
      label: { name: "Gradient direction", tex: "-\\nabla L(x_{t-1})" } },
    { vec: [-r.mHat[0], -r.mHat[1]], color: ARROW_ADAM_AVERAGED, fixedLength: true, width: 4, head: 14,
      label: { name: "Averaged direction", tex: "-\\hat{m}_t" } },
    { vec: r.step, color: ARROW_ADAM_STEP, fixedLength: true, width: 3.5, head: 12,
      label: { name: "Adam step", tex: "-\\eta\\, \\hat{m}_t \\,/\\, (\\sqrt{\\hat{s}_t} + \\epsilon)" } }
  ];
}

// Labels are { name, tex }: a short word name followed by a formula,
// rendered with KaTeX. Indexing: x_t is the current point (after t
// steps); the step just taken went from x_{t-1} to x_t, using the
// gradient at x_{t-1}.
const NOTE_DIRECTION = "Arrows show direction; lengths are scaled to be visible.";
const NOTE_TOGETHER = "Arrows are scaled up together, so their relative lengths are true.";

// Values panel rows shared by all optimizers.
const ROW_STEP = { name: "Step", tex: "t" };
const ROW_POSITION = { name: "Position", tex: "x_t" };
const ROW_GRADIENT = { name: "Gradient", tex: "\\nabla L(x_{t-1})" };
const ROW_STEP_TAKEN = { name: "Step taken", tex: "\\Delta x_t" };
const ROW_LOSS = { name: "Loss", tex: "L(x_t)" };

const OPTIMIZER_VIEWS = {
  gd: {
    name: "GD",
    color: COLORS.gd,
    showLegend: true,
    legendNote: NOTE_DIRECTION,
    arrows: function (r) {
      return [{ vec: r.step, color: COLORS.gd,
                label: { name: "Step", tex: "-\\eta\\, \\nabla L(x_{t-1})" } }];
    }
  },
  momentum: {
    name: "Momentum",
    color: COLORS.momentum,
    showLegend: true,
    legendNote: NOTE_TOGETHER,
    arrows: function (r) {
      return [
        { vec: r.gradientPart, color: ARROW_NEW_GRADIENT,
          label: { name: "Gradient push", tex: "-\\eta\\, \\nabla L(x_{t-1})" } },
        { vec: r.carriedPart, color: ARROW_CARRIED_OVER,
          label: { name: "Momentum carried over", tex: "-\\eta\\, \\beta\\, v_{t-1}" } },
        { vec: r.step, color: COLORS.momentum,
          label: { name: "Step taken", tex: "-\\eta\\, v_t" } }
      ];
    }
  },
  adam: {
    name: "Adam",
    color: COLORS.adam,
    showLegend: true,
    legendNote: NOTE_DIRECTION,
    arrows: adamArrows,
    // Values panel rows between Position and Loss, each an (x1, x2) pair.
    detailLabels: [
      ROW_GRADIENT,
      { name: "Averaged gradient", tex: "\\hat{m}_t" },
      { name: "Gradient scale", tex: "\\sqrt{\\hat{s}_t}" },
      { name: "Effective step size", tex: "\\eta \\,/ \\sqrt{\\hat{s}_t}", note: "(per coordinate)" },
      ROW_STEP_TAKEN
    ],
    details: function (r, g) {
      return [g, r.mHat, r.sqrtSHat, r.effectiveLr, r.step];
    }
  }
};

// Render a { name, tex, note } label into an element: the word name, then
// the formula with KaTeX (as plain text if KaTeX didn't load).
const mathCache = new Map();  // tex -> KaTeX HTML
function renderLabel(target, label) {
  const name = document.createElement("span");
  name.className = "ex-label-name";
  name.textContent = label.name;
  target.appendChild(name);
  const math = document.createElement("span");
  math.className = "ex-label-math";
  if (window.katex) {
    // Panels redraw every frame while playing, so typeset each formula once.
    if (!mathCache.has(label.tex)) {
      mathCache.set(label.tex, window.katex.renderToString(label.tex, { throwOnError: false }));
    }
    math.innerHTML = mathCache.get(label.tex);
  } else {
    math.textContent = label.tex;
  }
  target.appendChild(math);
  if (label.note) {
    const note = document.createElement("span");
    note.className = "ex-label-note";
    note.textContent = label.note;
    target.appendChild(note);
  }
}

// ---------------------------------------------------------------------
// Constants shared by all widgets
// ---------------------------------------------------------------------
const ALL_CONTROLS = [
  "optimizers", "learningRates", "sameLearningRate", "play", "step", "reset",
  "rotation", "elongation", "speed"
];
const SPEEDS = { normal: 150, fast: 30 };  // ms per step while playing
// Plot range for the small (square) size: x1 and x2 from -2 to 2.
const SQUARE_RANGE = { x1Min: -2, x1Max: 2, x2Min: -2, x2Max: 2 };
// Stage and full: the plot fills its whole area at the same scale on both
// axes. The scale is set so a circle of this radius around the minimum
// fits the shorter side; that keeps the start point (about 1.58 from the
// minimum) in view at every rotation, and the zoom doesn't change when
// the valley is rotated. The longer side shows more.
const STAGE_RADIUS = 1.75;
const MINIMUM = [0, 0];
const PIXEL_SCALE = 2;       // backing-store scale for sharp lines
const ARROW_TARGET_PX = 90;  // longest arrow of a step is drawn at least this long
// Logical drawing size of the small widget's square canvas (text and
// markers are sized in these units). Full and stage use their real size.
const CANVAS_SIZE = { small: 460 };
// Small: the plot fills the container's width, up to SMALL_PLOT_MAX, but
// is short enough that the whole widget (with its controls and values)
// fits in the window, less SMALL_WINDOW_MARGIN, since it stays in view
// while the reader scrolls.
const SMALL_PLOT_MAX = 520, SMALL_PLOT_MIN = 240, SMALL_WINDOW_MARGIN = 32;
// Log-scale learning-rate sliders: the slider value is log10(lr), from
// -4 to 0 (lr 0.0001 to 1.0). Dragged values are rounded to 3
// significant figures.
const LR_LOG_MIN = -4, LR_LOG_MAX = 0;

// Contour levels: fixed low levels near the minimum, plus levels that
// scale with K so the steep direction is covered at every elongation.
function contourLevels(K) {
  const levels = [0.1, 0.5, 1, 2, 4];
  for (const f of [0.1, 0.2, 0.35, 0.5, 0.7, 0.95, 1.3, 1.7, 2.2, 2.8, 3.5]) {
    if (f * K > 5.6) levels.push(f * K);
  }
  return levels;
}

// Numbers in the values panel: 4 decimals, or compact scientific form
// for huge values (a diverging run), so the panel keeps its width.
function fmt(n) {
  return Math.abs(n) >= 1e5 ? n.toExponential(2) : n.toFixed(4);
}
function fmtVec(v) {
  return "(" + fmt(v[0]) + ", " + fmt(v[1]) + ")";
}
function formatLr(lr) {
  return String(Number(lr.toPrecision(3)));
}

function makeSwatch(color) {
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = color;
  return swatch;
}

// Every widget on the page, so Play in one can pause the others.
const widgets = new Set();

const TEMPLATE = `
  <div class="ex-controls">
    <div class="ex-optimizers control-box"></div>
    <div class="ex-run control-box">
      <div class="ex-run-buttons">
        <button class="ex-play" type="button">Play</button>
        <button class="ex-step" type="button">Step (+1)</button>
        <button class="ex-reset" type="button">Reset</button>
      </div>
      <div class="ex-step-counter">Step: 0</div>
    </div>
    <div class="ex-speed control-box">
      <span>Speed:</span>
      <button type="button" data-speed="normal">Normal</button>
      <button type="button" data-speed="fast">Fast</button>
    </div>
    <label class="ex-rotation control-box">
      Rotation:
      <input class="ex-rotation-slider" type="range" min="0" max="90" step="5" value="0">
      <span class="ex-rotation-value">0&deg;</span>
    </label>
    <div class="ex-elongation control-box">
      <span class="ex-elongation-label">Elongation:</span>
      <button type="button" data-k="20">20:1</button>
      <button type="button" data-k="50">50:1</button>
      <button type="button" data-k="100">100:1</button>
    </div>
    <div class="ex-legend"></div>
  </div>
  <div class="ex-plot-panel">
    <canvas class="ex-plot" role="img"
      aria-label="Contour plot of the loss, with the optimizer paths and arrows"></canvas>
  </div>
  <div class="ex-text">
    <h2 class="ex-focus-name"></h2>
    <p class="ex-focus-description"></p>
    <table class="ex-values"></table>
    <p class="ex-status"></p>
  </div>
`;

export function createExplainer(container, config) {
  config = config || {};
  const sizeName = config.size === "small" || config.size === "stage" ? config.size : "full";
  const controls = new Set(config.controls || ALL_CONTROLS);
  if (controls.has("playStep")) {  // shorthand for all three buttons
    controls.add("play"); controls.add("step"); controls.add("reset");
  }
  const showValues = config.values !== false;
  const preset = Object.assign({
    elongation: 20,
    rotation: 0,
    start: DEFAULT_START,
    optimizers: ["gd"],
    learningRates: null,
    speed: "normal"
  }, config.preset || {});

  // -------------------------------------------------------------------
  // This widget's state
  // -------------------------------------------------------------------
  const land = { K: preset.elongation, thetaDeg: preset.rotation };
  const startInBowl = preset.start.slice();
  let start = rotateStart(startInBowl, land.thetaDeg);
  let lrs = Object.assign(defaultLearningRates(land.K), preset.learningRates || {});
  let checkOrder = preset.optimizers.slice();  // oldest first; last = most recently checked
  let playSpeed = preset.speed;
  const runs = {};      // id -> { x, state, path, stopped, diverged, last }
  let stepCount = 0;
  let playTimer = null;
  // Arrows for the latest step, one entry per optimizer that is still
  // moving: { id, origin, arrows }. Each new step replaces them.
  let arrowSets = [];

  // -------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------
  const root = document.createElement("div");
  root.className = "explainer explainer--" + sizeName;
  root.innerHTML = TEMPLATE;
  container.appendChild(root);
  const el = function (selector) { return root.querySelector(selector); };

  const plotPanel = el(".ex-plot-panel");
  const canvas = el(".ex-plot");
  // Logical drawing size (text and markers are sized in these units) and
  // the plot range. Fixed for the small size; stage and full set both in
  // fillArea().
  let CANVAS_W = CANVAS_SIZE[sizeName] || 640, CANVAS_H = CANVAS_SIZE[sizeName] || 640;
  let range = Object.assign({}, SQUARE_RANGE);
  let filledSize = null;  // "WxH" the canvas was last fitted to (stage, full)
  canvas.width = CANVAS_W * PIXEL_SCALE;
  canvas.height = CANVAS_H * PIXEL_SCALE;
  const ctx = canvas.getContext("2d");
  const background = document.createElement("canvas");
  background.width = canvas.width;
  background.height = canvas.height;

  // Hide the controls this widget doesn't show.
  const showRun = controls.has("play") || controls.has("step") || controls.has("reset");
  el(".ex-run").style.display = showRun ? "" : "none";
  el(".ex-play").style.display = controls.has("play") ? "" : "none";
  el(".ex-step").style.display = controls.has("step") ? "" : "none";
  el(".ex-reset").style.display = controls.has("reset") ? "" : "none";
  el(".ex-text").style.display = showValues ? "" : "none";
  el(".ex-speed").style.display = controls.has("speed") ? "" : "none";
  el(".ex-rotation").style.display = controls.has("rotation") ? "" : "none";
  el(".ex-elongation").style.display = controls.has("elongation") ? "" : "none";
  const showOptimizerBox = controls.has("optimizers") || controls.has("learningRates");
  el(".ex-optimizers").style.display = showOptimizerBox ? "" : "none";

  // -------------------------------------------------------------------
  // Simulation
  // -------------------------------------------------------------------
  function isChecked(id) {
    return checkOrder.indexOf(id) !== -1;
  }
  function checkedIds() {
    return OPTIMIZER_IDS.filter(isChecked);
  }
  function focusId() {
    return checkOrder.length ? checkOrder[checkOrder.length - 1] : null;
  }

  function resetAll() {
    pause();
    for (const id of OPTIMIZER_IDS) {
      runs[id] = {
        x: start.slice(),
        state: OPTIMIZERS[id].init(),
        path: [start.slice()],
        stopped: false,
        diverged: false,
        last: null
      };
    }
    stepCount = 0;
    arrowSets = [];  // no arrows until the first step
    render();
  }

  function allStopped() {
    return checkedIds().every(function (id) { return runs[id].stopped; });
  }

  // Advance every checked optimizer that is still moving by one step.
  // Returns false if nothing could move. Replaces the arrows with ones
  // for this step, drawn at each optimizer's new (current) point.
  function advance() {
    const ids = checkedIds();
    if (ids.length === 0 || allStopped()) return false;

    arrowSets = [];
    for (const id of ids) {
      const run = runs[id];
      if (run.stopped) continue;
      const view = OPTIMIZER_VIEWS[id];
      const g = gradient(run.x, land);
      const result = OPTIMIZERS[id].step(run.state, g, lrs[id]);
      run.x = [run.x[0] + result.step[0], run.x[1] + result.step[1]];
      // Only finite points go into the drawn path (a NaN point can't be drawn).
      if (isFinite(run.x[0]) && isFinite(run.x[1])) run.path.push(run.x.slice());
      run.last = {
        gradient: g,
        step: result.step,
        details: view.details ? view.details(result, g) : null
      };
      const L = loss(run.x, land);
      if (isDiverged(L)) {
        run.diverged = true;  // stop it; the others keep going
        run.stopped = true;
      } else if (L < STOP_LOSS) {
        run.stopped = true;
      }
      // Stopped or diverged optimizers get no arrows.
      if (!run.stopped) {
        // One optimizer checked: its full set of arrows. Two or three:
        // only the step taken (the last arrow), in its path color.
        const all = view.arrows(result, g);
        const arrows = ids.length === 1
          ? all
          : [Object.assign({}, all[all.length - 1], { color: view.color })];
        arrowSets.push({ id: id, origin: run.x.slice(), arrows: arrows });
      }
    }
    stepCount += 1;
    return true;
  }

  function play() {
    if (playTimer !== null) return;
    if (checkedIds().length === 0 || allStopped() || stepCount >= MAX_STEPS) return;
    // Only one widget animates at a time.
    for (const other of widgets) {
      if (other !== api && other.isPlaying()) other.pause();
    }
    playTimer = setInterval(playTick, SPEEDS[playSpeed]);
    render();
  }

  function playTick() {
    advance();
    if (allStopped() || stepCount >= MAX_STEPS) pause();
    render();
  }

  function pause() {
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = null;
    }
  }

  function stepOnce() {
    pause();
    advance();
    render();
  }

  // Switch between Normal and Fast; takes effect immediately if playing.
  function setSpeed(name) {
    playSpeed = name;
    for (const button of root.querySelectorAll(".ex-speed button")) {
      button.classList.toggle("selected", button.dataset.speed === playSpeed);
    }
    if (playTimer !== null) {
      clearInterval(playTimer);
      playTimer = setInterval(playTick, SPEEDS[playSpeed]);
    }
  }

  // -------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------
  function toPx(x) {
    return [
      (x[0] - range.x1Min) / (range.x1Max - range.x1Min) * CANVAS_W,
      (range.x2Max - x[1]) / (range.x2Max - range.x2Min) * CANVAS_H
    ];
  }

  // Contours are traced from loss() itself (marching squares), on a grid
  // of about 2 logical pixels per cell.
  function drawContours(c) {
    const nx = Math.max(64, Math.round(CANVAS_W / 2)), ny = Math.max(64, Math.round(CANVAS_H / 2));
    const dx = (range.x1Max - range.x1Min) / nx;
    const dy = (range.x2Max - range.x2Min) / ny;
    const vals = [];
    for (let j = 0; j <= ny; j++) {
      const row = [];
      for (let i = 0; i <= nx; i++) {
        row.push(loss([range.x1Min + i * dx, range.x2Min + j * dy], land));
      }
      vals.push(row);
    }

    function crossing(pa, va, pb, vb, level) {
      const t = (level - va) / (vb - va);
      return toPx([pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])]);
    }

    c.lineWidth = 1;
    const levels = contourLevels(land.K);
    for (let li = 0; li < levels.length; li++) {
      const level = levels[li];
      const t = levels.length > 1 ? li / (levels.length - 1) : 0;
      const alpha = CONTOUR_ALPHA_INNER + (CONTOUR_ALPHA_OUTER - CONTOUR_ALPHA_INNER) * t;
      c.strokeStyle = "rgba(" + CONTOUR_RGB + ", " + alpha.toFixed(3) + ")";
      c.beginPath();
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const x0 = range.x1Min + i * dx, x1 = x0 + dx;
          const y0 = range.x2Min + j * dy, y1 = y0 + dy;
          const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
          const v = [vals[j][i], vals[j][i + 1], vals[j + 1][i + 1], vals[j + 1][i]];
          const pts = [];
          for (let k = 0; k < 4; k++) {
            const k2 = (k + 1) % 4;
            if ((v[k] < level) !== (v[k2] < level)) {
              pts.push(crossing(corners[k], v[k], corners[k2], v[k2], level));
            }
          }
          for (let k = 0; k + 1 < pts.length; k += 2) {
            c.moveTo(pts[k][0], pts[k][1]);
            c.lineTo(pts[k + 1][0], pts[k + 1][1]);
          }
        }
      }
      c.stroke();
    }
  }

  // Light dashed x1 and x2 axis lines through (0, 0), with labels.
  function drawAxes(c) {
    const o = toPx([0, 0]);
    c.strokeStyle = AXIS_COLOR;
    c.lineWidth = 1;
    c.setLineDash([5, 5]);
    c.beginPath();
    c.moveTo(0, o[1]); c.lineTo(CANVAS_W, o[1]);
    c.moveTo(o[0], 0); c.lineTo(o[0], CANVAS_H);
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = AXIS_LABEL_COLOR;
    c.font = "italic 14px Inter, system-ui, sans-serif";
    c.fillText("x1", CANVAS_W - 22, o[1] - 7);
    c.fillText("x2", o[0] + 7, 16);
  }

  // The axes and contours only change when the landscape changes, so
  // they are drawn once per landscape and reused every frame.
  function redrawBackground() {
    const c = background.getContext("2d");
    c.setTransform(PIXEL_SCALE, 0, 0, PIXEL_SCALE, 0, 0);
    // Panel color with a slightly lighter glow around the minimum.
    const m = toPx(MINIMUM);
    const glow = c.createRadialGradient(m[0], m[1], 0, m[0], m[1], CANVAS_W * 0.7);
    glow.addColorStop(0, PLOT_BG_CENTER);
    glow.addColorStop(1, PLOT_BG);
    c.fillStyle = glow;
    c.fillRect(0, 0, CANVAS_W, CANVAS_H);
    drawAxes(c);
    drawContours(c);
  }

  // Minimum: a cross and label. Drawn on top of the paths, with an
  // outline in the plot's background color, so it stays visible at the
  // end of a run.
  function drawMinimumMarker() {
    const m = toPx(MINIMUM);
    const label = "minimum (0, 0)";
    ctx.font = "13px Inter, system-ui, sans-serif";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const pass of [{ color: PLOT_BG, width: 5 }, { color: MARKER_COLOR, width: 2 }]) {
      ctx.strokeStyle = pass.color;
      ctx.lineWidth = pass.width;
      ctx.beginPath();
      ctx.moveTo(m[0] - 6, m[1] - 6); ctx.lineTo(m[0] + 6, m[1] + 6);
      ctx.moveTo(m[0] - 6, m[1] + 6); ctx.lineTo(m[0] + 6, m[1] - 6);
      ctx.stroke();
    }
    ctx.strokeStyle = PLOT_BG;
    ctx.lineWidth = 3;
    ctx.strokeText(label, m[0] + 10, m[1] + 18);
    ctx.fillStyle = MARKER_COLOR;
    ctx.fillText(label, m[0] + 10, m[1] + 18);
    ctx.lineJoin = "miter";
    ctx.lineCap = "butt";
  }

  // Start: a ring and label.
  function drawStartMarker() {
    ctx.font = "13px Inter, system-ui, sans-serif";
    ctx.fillStyle = MARKER_COLOR;
    ctx.strokeStyle = MARKER_COLOR;
    ctx.lineWidth = 2;
    const s = toPx(start);
    ctx.beginPath();
    ctx.arc(s[0], s[1], 7, 0, 2 * Math.PI);
    ctx.stroke();
    const startLabel = "start (" + Number(start[0].toFixed(2)) + ", " + Number(start[1].toFixed(2)) + ")";
    // Keep the label inside the plot (it can be near the edge on narrow screens).
    const labelW = ctx.measureText(startLabel).width;
    const labelX = Math.max(4, Math.min(CANVAS_W - labelW - 4, s[0] - 30));
    ctx.fillText(startLabel, labelX, s[1] - 14);
  }

  function drawPath(id) {
    const path = runs[id].path;
    ctx.strokeStyle = OPTIMIZER_VIEWS[id].color;
    ctx.fillStyle = OPTIMIZER_VIEWS[id].color;
    ctx.lineWidth = 1.75;
    ctx.lineJoin = "round";
    // The line gets a soft glow in its own color.
    ctx.shadowColor = OPTIMIZER_VIEWS[id].color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i < path.length; i++) {
      const p = toPx(path[i]);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.lineJoin = "miter";
    for (let i = 0; i < path.length; i++) {
      const p = toPx(path[i]);
      ctx.beginPath();
      ctx.arc(p[0], p[1], 2.6, 0, 2 * Math.PI);
      ctx.fill();
    }
    // The current position: a larger dot with a ring around it.
    if (path.length > 1) {
      const p = toPx(path[path.length - 1]);
      ctx.beginPath();
      ctx.arc(p[0], p[1], 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = PLOT_BG;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p[0], p[1], 8.5, 0, 2 * Math.PI);
      ctx.strokeStyle = OPTIMIZER_VIEWS[id].color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  function drawArrow(from, to, color, width, headSize) {
    const dx = to[0] - from[0], dy = to[1] - from[1];
    const len = Math.hypot(dx, dy);
    if (len < 0.5) return;  // nothing to show (e.g. carried-over part on step 1)
    const ux = dx / len, uy = dy / len;
    const head = Math.min(headSize || 12, len * 0.6);
    const baseX = to[0] - ux * head, baseY = to[1] - uy * head;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    ctx.lineTo(baseX, baseY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(to[0], to[1]);
    ctx.lineTo(baseX - uy * head * 0.45, baseY + ux * head * 0.45);
    ctx.lineTo(baseX + uy * head * 0.45, baseY - ux * head * 0.45);
    ctx.closePath();
    ctx.fill();
  }

  // Draw one optimizer's arrows, starting at its current point.
  function drawArrowSet(set) {
    const pxPerUnit = CANVAS_W / (range.x1Max - range.x1Min);
    const origin = toPx(set.origin);

    // One common scale factor for all arrows of this step.
    let longest = 0;
    for (const a of set.arrows) {
      longest = Math.max(longest, Math.hypot(a.vec[0], a.vec[1]) * pxPerUnit);
    }
    if (!(longest > 0)) return;
    const factor = Math.max(1, ARROW_TARGET_PX / longest);

    set.arrows.forEach(function (a, i) {
      const isLast = i === set.arrows.length - 1;  // the step actually taken
      // Fixed-length arrows show direction only; the others share `factor`.
      const lengthPx = Math.hypot(a.vec[0], a.vec[1]) * pxPerUnit;
      if (lengthPx === 0) return;
      const scale = a.fixedLength ? ARROW_TARGET_PX / lengthPx : factor;
      const tip = [
        origin[0] + a.vec[0] * pxPerUnit * scale,
        origin[1] - a.vec[1] * pxPerUnit * scale
      ];
      drawArrow(origin, tip, a.color, a.width || (isLast ? 3.5 : 2.5), a.head);
    });
  }

  function drawPlot() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(background, 0, 0);
    ctx.setTransform(PIXEL_SCALE, 0, 0, PIXEL_SCALE, 0, 0);
    drawStartMarker();
    for (const id of checkedIds()) drawPath(id);
    drawMinimumMarker();  // on top of the paths
    for (const set of arrowSets) drawArrowSet(set);
  }

  // Arrow legend (not drawn on the plot), shown when exactly one
  // optimizer is checked.
  function renderLegend() {
    const legendEl = el(".ex-legend");
    legendEl.innerHTML = "";
    if (checkedIds().length !== 1 || arrowSets.length !== 1) return;
    const set = arrowSets[0];
    const view = OPTIMIZER_VIEWS[set.id];
    if (!view.showLegend) return;

    for (const a of set.arrows) {
      const row = document.createElement("div");
      row.className = "legend-row";
      row.innerHTML =
        '<svg width="30" height="12" viewBox="0 0 30 12">' +
        '<line x1="0" y1="6" x2="20" y2="6" stroke-width="2.5"></line>' +
        '<polygon points="30,6 19,1.5 19,10.5"></polygon></svg>';
      row.querySelector("line").setAttribute("stroke", a.color);
      row.querySelector("polygon").setAttribute("fill", a.color);
      const text = document.createElement("span");
      text.className = "legend-label";
      renderLabel(text, a.label);
      row.appendChild(text);
      legendEl.appendChild(row);
    }
    if (view.legendNote) {
      const noteRow = document.createElement("div");
      noteRow.className = "legend-note";
      noteRow.textContent = view.legendNote;
      legendEl.appendChild(noteRow);
    }
  }

  // Stage and full: the plot fills its whole area (see fillArea).
  // Small: a square as wide as the widget's inner width, up to
  // SMALL_PLOT_MAX, and short enough that the whole widget fits in the
  // window.
  function sizePlot() {
    if (sizeName === "stage") {
      fillArea(root);
      return;
    }
    if (sizeName === "full") {
      fillArea(plotPanel);  // the plot column, right of the controls
      return;
    }
    const style = getComputedStyle(root);
    const innerW = root.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    // Everything in the widget except the plot (controls, legend space,
    // values) keeps its height when the plot is resized.
    const otherH = root.offsetHeight - plotPanel.offsetHeight;
    const fitH = window.innerHeight - SMALL_WINDOW_MARGIN - otherH;
    const size = Math.floor(Math.min(innerW, SMALL_PLOT_MAX, Math.max(SMALL_PLOT_MIN, fitH)));
    plotPanel.style.width = size + "px";
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
  }

  // The canvas fills the given element (any shape), drawn at its real
  // pixel size. The range keeps equal scale on both axes, centered on the
  // minimum, with STAGE_RADIUS fitting the shorter side.
  function fillArea(area) {
    const W = area.clientWidth, H = area.clientHeight;
    if (W === 0 || H === 0) return;  // hidden: sized again when shown
    if (filledSize === W + "x" + H) return;  // unchanged
    filledSize = W + "x" + H;
    CANVAS_W = W;
    CANVAS_H = H;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    canvas.width = background.width = Math.round(W * PIXEL_SCALE);
    canvas.height = background.height = Math.round(H * PIXEL_SCALE);
    const unitsPerPx = (2 * STAGE_RADIUS) / Math.min(W, H);
    range = {
      x1Min: -W / 2 * unitsPerPx, x1Max: W / 2 * unitsPerPx,
      x2Min: -H / 2 * unitsPerPx, x2Max: H / 2 * unitsPerPx
    };
    redrawBackground();
    if (runs.gd) render();  // keep the current run; just redraw it
  }

  // -------------------------------------------------------------------
  // Values panel
  // -------------------------------------------------------------------
  function renderText() {
    const nameEl = el(".ex-focus-name");
    const descEl = el(".ex-focus-description");
    const valuesEl = el(".ex-values");
    const statusEl = el(".ex-status");
    const id = focusId();

    valuesEl.innerHTML = "";
    statusEl.textContent = "";
    nameEl.style.display = "";
    descEl.style.display = "";
    // A landscape-only widget (nothing checked, no checkboxes to check)
    // has nothing to show in the values panel, so hide it.
    el(".ex-text").style.display =
      !showValues || (id === null && !controls.has("optimizers")) ? "none" : "";
    if (id === null) {
      nameEl.textContent = "No optimizer selected";
      nameEl.style.color = "";
      descEl.textContent = "Check GD, Momentum, or Adam to begin.";
      return;
    }

    // Two or three checked: one compact row per checked optimizer.
    const ids = checkedIds();
    if (ids.length > 1) {
      nameEl.style.display = "none";
      descEl.style.display = "none";
      for (const cid of ids) {
        const r = runs[cid];
        const tr = document.createElement("tr");
        const nameTd = document.createElement("td");
        nameTd.appendChild(makeSwatch(OPTIMIZER_VIEWS[cid].color));
        nameTd.appendChild(document.createTextNode(OPTIMIZER_VIEWS[cid].name));
        nameTd.className = "ex-name-cell";
        tr.appendChild(nameTd);
        const cells = [
          "step " + (r.path.length - 1),
          "loss " + fmt(loss(r.x, land)),
          r.diverged ? "diverged" : r.stopped ? "stopped" : ""
        ];
        for (const cell of cells) {
          const td = document.createElement("td");
          td.textContent = cell;
          tr.appendChild(td);
        }
        valuesEl.appendChild(tr);
      }
      return;
    }

    const view = OPTIMIZER_VIEWS[id];
    const run = runs[id];
    nameEl.textContent = view.name;
    nameEl.style.color = view.color;
    descEl.style.display = "none";  // only used when nothing is checked

    const rows = [
      [ROW_STEP, String(run.path.length - 1)],
      [ROW_POSITION, fmtVec(run.x)]
    ];
    if (view.detailLabels) {
      // Optimizer-specific rows (Adam), each an (x1, x2) pair.
      view.detailLabels.forEach(function (label, i) {
        rows.push([label, run.last ? fmtVec(run.last.details[i]) : "-"]);
      });
    } else {
      rows.push([ROW_GRADIENT, run.last ? fmtVec(run.last.gradient) : "-"]);
      rows.push([ROW_STEP_TAKEN, run.last ? fmtVec(run.last.step) : "-"]);
    }
    rows.push([ROW_LOSS, fmt(loss(run.x, land))]);
    for (const row of rows) {
      const tr = document.createElement("tr");
      const labelTd = document.createElement("td");
      labelTd.className = "ex-row-label";
      renderLabel(labelTd, row[0]);
      const valueTd = document.createElement("td");
      valueTd.textContent = row[1];
      tr.appendChild(labelTd);
      tr.appendChild(valueTd);
      valuesEl.appendChild(tr);
    }
    if (run.diverged) {
      statusEl.textContent = "Diverged: loss went above " + DIVERGE_LOSS + " or became NaN.";
    } else if (run.stopped) {
      statusEl.textContent = "Stopped: loss is below " + STOP_LOSS + ".";
    }
  }

  function render() {
    drawPlot();
    renderLegend();
    renderText();
    el(".ex-step-counter").textContent = "Step: " + stepCount;
    el(".ex-play").textContent = playTimer !== null ? "Pause" : "Play";
  }

  // -------------------------------------------------------------------
  // Optimizer checkboxes and learning-rate sliders
  // -------------------------------------------------------------------
  const lrNotes = {}, lrSliders = {}, lrRows = {};
  let sameInput = null, sharedSlider = null, sharedRow = null;

  // Log-scale learning-rate slider.
  function makeLrSlider(onChange) {
    const row = document.createElement("div");
    row.className = "lr-row";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = LR_LOG_MIN;
    slider.max = LR_LOG_MAX;
    slider.step = "any";  // continuous, so the defaults sit exactly on it
    slider.addEventListener("input", function () {
      onChange(Number(Math.pow(10, Number(slider.value)).toPrecision(3)));
    });
    row.appendChild(slider);
    return { row: row, slider: slider };
  }

  function buildOptimizerBox() {
    const box = el(".ex-optimizers");
    // Without checkboxes the checked optimizers can't change, so only
    // those get a row (a GD-only widget shows only GD's slider).
    const listed = controls.has("optimizers") ? OPTIMIZER_IDS : checkedIds();
    for (const id of OPTIMIZER_IDS) {
      const view = OPTIMIZER_VIEWS[id];
      if (listed.indexOf(id) === -1) {
        lrNotes[id] = document.createElement("span");  // not shown
        continue;
      }
      const group = document.createElement("div");
      group.className = "ex-opt";
      const label = document.createElement("label");
      if (controls.has("optimizers")) {
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = isChecked(id);
        input.addEventListener("change", function () {
          checkOrder = checkOrder.filter(function (other) { return other !== id; });
          if (input.checked) checkOrder.push(id);
          resetAll();  // checking or unchecking a box also resets
        });
        label.appendChild(input);
      }
      label.appendChild(makeSwatch(view.color));
      label.appendChild(document.createTextNode(view.name));
      const lrNote = document.createElement("span");
      lrNote.className = "lr-note";
      label.appendChild(lrNote);
      lrNotes[id] = lrNote;
      group.appendChild(label);

      if (controls.has("learningRates")) {
        // Learning-rate slider under the checkbox.
        const s = makeLrSlider(function (lr) {
          lrs[id] = lr;
          syncLrControls();
          resetAll();  // changing a learning rate resets the optimizers
        });
        lrSliders[id] = s.slider;
        lrRows[id] = s.row;
        group.appendChild(s.row);
      }
      box.appendChild(group);
    }

    // "Same learning rate for all": one shared slider replaces the others.
    // Only shown with the sliders and two or more optimizers.
    if (!controls.has("sameLearningRate") || !controls.has("learningRates") || listed.length < 2) return;

    const same = document.createElement("div");
    same.className = "ex-same-lr";
    const sameLabel = document.createElement("label");
    sameLabel.className = "same-lr-label";
    sameInput = document.createElement("input");
    sameInput.type = "checkbox";
    sameInput.addEventListener("change", function () {
      if (sameInput.checked) {
        // Start the shared value from the optimizer in focus (or GD).
        const lr = lrs[focusId() || "gd"];
        for (const id of OPTIMIZER_IDS) lrs[id] = lr;
        syncLrControls();
        resetAll();
      } else {
        syncLrControls();  // each keeps the shared value until changed
      }
    });
    sameLabel.appendChild(sameInput);
    sameLabel.appendChild(document.createTextNode(" Same learning rate for all"));
    same.appendChild(sameLabel);

    const shared = makeLrSlider(function (lr) {
      for (const id of OPTIMIZER_IDS) lrs[id] = lr;
      syncLrControls();
      resetAll();
    });
    sharedSlider = shared.slider;
    sharedRow = shared.row;
    same.appendChild(sharedRow);
    box.appendChild(same);
  }

  // Update the learning-rate labels and sliders to the current values,
  // and show either the three sliders or the shared one.
  function syncLrControls() {
    const same = sameInput !== null && sameInput.checked;
    for (const id of OPTIMIZER_IDS) {
      lrNotes[id].textContent = "η = " + formatLr(lrs[id]);  // "η = 0.008"
      if (lrSliders[id]) {
        lrSliders[id].value = Math.log10(lrs[id]);
        lrRows[id].style.display = same ? "none" : "";
      }
    }
    if (sharedRow) {
      sharedSlider.value = Math.log10(lrs.gd);
      sharedRow.style.display = same ? "" : "none";
    }
  }

  // Back to the default learning rates for the current elongation, each
  // optimizer with its own slider.
  function resetLearningRates() {
    lrs = defaultLearningRates(land.K);
    if (sameInput) sameInput.checked = false;
    syncLrControls();
  }

  // Apply the current landscape: start point, contours and control
  // labels, then reset all optimizers. (Learning rates are not touched
  // here: rotating keeps them; elongation resets them separately.)
  function applyLandscape() {
    start = rotateStart(startInBowl, land.thetaDeg);
    redrawBackground();
    syncLrControls();
    el(".ex-rotation-slider").value = land.thetaDeg;
    el(".ex-rotation-value").textContent = land.thetaDeg + "°";
    for (const button of root.querySelectorAll(".ex-elongation button")) {
      button.classList.toggle("selected", Number(button.dataset.k) === land.K);
    }
    resetAll();
  }

  // -------------------------------------------------------------------
  // Wiring
  // -------------------------------------------------------------------
  el(".ex-rotation-slider").addEventListener("input", function (event) {
    land.thetaDeg = Number(event.target.value);
    applyLandscape();
  });
  for (const button of root.querySelectorAll(".ex-elongation button")) {
    button.addEventListener("click", function () {
      land.K = Number(button.dataset.k);
      resetLearningRates();  // changing the elongation sets the defaults again
      applyLandscape();
    });
  }
  for (const button of root.querySelectorAll(".ex-speed button")) {
    button.addEventListener("click", function () {
      setSpeed(button.dataset.speed);
    });
  }
  el(".ex-play").addEventListener("click", function () {
    if (playTimer !== null) {
      pause();
      render();
    } else {
      play();
    }
  });
  el(".ex-step").addEventListener("click", stepOnce);
  el(".ex-reset").addEventListener("click", resetAll);

  // What other widgets (and the page) can do with this one.
  const api = {
    isPlaying: function () { return playTimer !== null; },
    pause: function () { pause(); render(); },
    // Show or hide the widget (the stage shows one section's widget at a
    // time). A hidden widget is paused but keeps its run, so it is still
    // there when the reader scrolls back.
    setVisible: function (visible) {
      if (!visible) {
        pause();
        root.style.display = "none";
        render();
        return;
      }
      root.style.display = "";
      sizePlot();
      render();
    },
    destroy: function () {
      pause();
      widgets.delete(api);
      window.removeEventListener("resize", sizePlot);
      root.remove();
    }
  };
  widgets.add(api);

  // Small widgets with one optimizer: reserve room for the arrow legend
  // from the start, so the widget doesn't grow when it appears.
  function reserveLegendSpace() {
    const ids = checkedIds();
    if (sizeName !== "small" || ids.length !== 1 || controls.has("optimizers")) return;
    const view = OPTIMIZER_VIEWS[ids[0]];
    if (!view.showLegend) return;
    const zero = [0, 0];
    const dummy = { step: zero, gradientPart: zero, carriedPart: zero, mHat: zero };
    arrowSets = [{ id: ids[0], origin: zero, arrows: view.arrows(dummy, zero) }];
    renderLegend();
    const legendEl = el(".ex-legend");
    legendEl.style.minHeight = legendEl.offsetHeight + "px";
    arrowSets = [];
    renderLegend();
  }

  buildOptimizerBox();
  setSpeed(playSpeed);
  applyLandscape();  // draws everything
  reserveLegendSpace();
  sizePlot();
  window.addEventListener("resize", sizePlot);

  return api;
}
