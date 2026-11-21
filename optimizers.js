// =====================================================================
// optimizers.js: the math only. The loss landscape, its gradient, and
// the three optimizers. No drawing, no DOM, no colors.
// =====================================================================

export const STOP_LOSS = 0.001;   // an optimizer stops once its loss is below this
export const DIVERGE_LOSS = 1e6;  // ...or has diverged once its loss is above this (or NaN)
export const MAX_STEPS = 300;     // Play stops at this many steps

// Start point in the bowl's own frame; it is rotated with the bowl.
export const DEFAULT_START = [-1.5, 0.5];

// ---------------------------------------------------------------------
// Loss landscape: a bowl with elongation K, rotated by theta.
//   u =  cos(theta)*x1 + sin(theta)*x2
//   w = -sin(theta)*x1 + cos(theta)*x2
//   L = u^2 + K*w^2
// With K = 20 and theta = 0 this is x1^2 + 20*x2^2.
// `land` is { K, thetaDeg }.
// ---------------------------------------------------------------------
export function loss(x, land) {
  const theta = land.thetaDeg * Math.PI / 180;
  const c = Math.cos(theta), s = Math.sin(theta);
  const u = c * x[0] + s * x[1];
  const w = -s * x[0] + c * x[1];
  return u * u + land.K * w * w;
}

export function gradient(x, land) {
  const theta = land.thetaDeg * Math.PI / 180;
  const c = Math.cos(theta), s = Math.sin(theta);
  const K = land.K;
  const u = c * x[0] + s * x[1];
  const w = -s * x[0] + c * x[1];
  return [c * 2 * u - s * 2 * K * w, s * 2 * u + c * 2 * K * w];
}

// Rotate a point given in the bowl's own frame by thetaDeg, so the start
// point is always the same spot relative to the bowl.
export function rotateStart(p, thetaDeg) {
  const theta = thetaDeg * Math.PI / 180;
  const c = Math.cos(theta), s = Math.sin(theta);
  return [c * p[0] - s * p[1], s * p[0] + c * p[1]];
}

// Default learning rates for elongation K.
export function defaultLearningRates(K) {
  return {
    gd: 0.8 / K,        // 0.04 at 20:1
    momentum: 0.2 / K,  // 0.01 at 20:1
    adam: 0.1
  };
}

export function isDiverged(L) {
  return isNaN(L) || L > DIVERGE_LOSS;
}

// ---------------------------------------------------------------------
// Optimizers. Each has:
//   init()              -> fresh state (all zeros)
//   step(state, g, lr)  -> { step: [dx1, dx2], ...intermediate values }
// step() updates the optimizer's own state but not the position; the
// caller adds `step` to the position.
// ---------------------------------------------------------------------

// GD: x = x - lr*g
function gdInit() {
  return {};
}
function gdStep(state, g, lr) {
  return { step: [-lr * g[0], -lr * g[1]] };
}

// Momentum: v = beta*v + g;  x = x - lr*v
export const MOMENTUM_BETA = 0.9;
function momentumInit() {
  return { v: [0, 0] };
}
function momentumStep(state, g, lr) {
  const beta = MOMENTUM_BETA;
  const vPrev = state.v;
  const gradientPart = [-lr * g[0], -lr * g[1]];                       // -lr*g
  const carriedPart = [-lr * beta * vPrev[0], -lr * beta * vPrev[1]];  // -lr*beta*v_previous
  state.v = [beta * vPrev[0] + g[0], beta * vPrev[1] + g[1]];
  const step = [-lr * state.v[0], -lr * state.v[1]];                   // their sum
  return { step: step, gradientPart: gradientPart, carriedPart: carriedPart };
}

// Adam: m = b1*m + (1-b1)*g;  s = b2*s + (1-b2)*g^2  (per coordinate)
//       m_hat = m/(1-b1^t);   s_hat = s/(1-b2^t)
//       x = x - lr*m_hat/(sqrt(s_hat)+eps)
export const ADAM_PARAMS = { b1: 0.9, b2: 0.999, eps: 1e-8 };
function adamInit() {
  return { m: [0, 0], s: [0, 0], t: 0 };
}
function adamStep(state, g, lr) {
  const p = ADAM_PARAMS;
  state.t += 1;
  const step = [0, 0], mHat = [0, 0], sqrtSHat = [0, 0], effectiveLr = [0, 0];
  for (let i = 0; i < 2; i++) {
    state.m[i] = p.b1 * state.m[i] + (1 - p.b1) * g[i];
    state.s[i] = p.b2 * state.s[i] + (1 - p.b2) * g[i] * g[i];
    mHat[i] = state.m[i] / (1 - Math.pow(p.b1, state.t));
    sqrtSHat[i] = Math.sqrt(state.s[i] / (1 - Math.pow(p.b2, state.t)));
    effectiveLr[i] = lr / sqrtSHat[i];
    step[i] = -lr * mHat[i] / (sqrtSHat[i] + p.eps);
  }
  return { step: step, mHat: mHat, sqrtSHat: sqrtSHat, effectiveLr: effectiveLr };
}

export const OPTIMIZERS = {
  gd: { init: gdInit, step: gdStep },
  momentum: { init: momentumInit, step: momentumStep },
  adam: { init: adamInit, step: adamStep }
};
export const OPTIMIZER_IDS = ["gd", "momentum", "adam"];
