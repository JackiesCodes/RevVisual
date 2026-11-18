// Exemplar-based hole filling (PatchMatch with multi-scale voting, as in
// Wexler et al. and Barnes et al.; the idea behind "content-aware fill").
// Every hole pixel is rebuilt from real patches found elsewhere in the
// window, so the fill keeps the surroundings' texture and never invents
// colours that aren't there. The patch map is solved small, then used to
// copy source pixels at full resolution, so the result stays sharp.

const R = 3; // patch radius (7×7 patches)
/**
 * Preference for nearby source patches, per squared pixel of distance at the
 * base level: a matching texture next to the hole beats a slightly closer
 * match across the picture (road from road, sky from sky).
 */
const NEAR = 12;

interface Level {
  w: number;
  h: number;
  /** RGB, 0..255 */
  c: Float32Array;
  /** 1 = hole */
  hole: Uint8Array;
}

/**
 * Fills `hole` (s×s, 1 = fill) in the RGBA window `px` (s×s) and returns the
 * filled RGBA pixels (known pixels unchanged), or null when the window holds
 * too little intact background to copy from.
 */
export function patchFill(px: Uint8ClampedArray, hole: Uint8Array, s: number, seed = 1): Uint8ClampedArray | null {
  let rnd = seed >>> 0 || 1;
  const rand = () => ((rnd = (rnd * 1664525 + 1013904223) >>> 0) / 4294967296);

  // Base level at up to 160 px; coarser levels halve down to ~24 px.
  const n0 = Math.min(160, s);
  const levels: Level[] = [downsample(px, hole, s, n0)];
  while (Math.min(levels[levels.length - 1].w, levels[levels.length - 1].h) > 40) levels.push(half(levels[levels.length - 1]));

  let nnf: Int32Array | null = null; // per pixel: source index (y*w+x) for hole pixels
  for (let li = levels.length - 1; li >= 0; li--) {
    const L = levels[li];
    const { w, h } = L;
    const valid = validCenters(L);
    if (valid.length < 8) return null;
    // Initial guess for hole colours: coarse upsampled fill or, at the top, the mean of known pixels.
    if (nnf) {
      const P = levels[li + 1];
      const up = new Int32Array(w * h).fill(-1);
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          if (!L.hole[y * w + x]) continue;
          const px2 = Math.min(P.w - 1, x >> 1);
          const py2 = Math.min(P.h - 1, y >> 1);
          const src = nnf[py2 * P.w + px2];
          if (src < 0) continue;
          const sx = Math.min(w - 1 - R, Math.max(R, (src % P.w) * 2 + (x & 1)));
          const sy = Math.min(h - 1 - R, Math.max(R, ((src / P.w) | 0) * 2 + (y & 1)));
          up[y * w + x] = isValid(L, sx, sy) ? sy * w + sx : valid[(rand() * valid.length) | 0];
        }
      nnf = up;
      vote(L, nnf);
    } else {
      nnf = new Int32Array(w * h).fill(-1);
      smoothGuess(L);
    }
    for (let i = 0; i < w * h; i++) if (L.hole[i] && nnf[i] < 0) nnf[i] = valid[(rand() * valid.length) | 0];

    const iters = li === levels.length - 1 ? 6 : 3;
    const near = NEAR * Math.pow(4, li);
    for (let it = 0; it < iters; it++) {
      patchMatch(L, nnf, valid, rand, it, near);
      vote(L, nnf);
    }
  }

  // Full resolution: copy source pixels through the base-level patch map.
  const L = levels[0];
  const k = s / L.w;
  const out = px.slice();
  for (let Y = 0; Y < s; Y++)
    for (let X = 0; X < s; X++) {
      if (!hole[Y * s + X]) continue;
      const lx = Math.min(L.w - 1, Math.floor(X / k));
      const ly = Math.min(L.h - 1, Math.floor(Y / k));
      const src = nnf![ly * L.w + lx];
      const o = (Y * s + X) * 4;
      let sx = -1;
      let sy = -1;
      if (src >= 0) {
        sx = Math.round((src % L.w) * k + (X - lx * k));
        sy = Math.round(((src / L.w) | 0) * k + (Y - ly * k));
      }
      if (sx >= 0 && sy >= 0 && sx < s && sy < s && !hole[sy * s + sx]) {
        const so = (sy * s + sx) * 4;
        out[o] = px[so];
        out[o + 1] = px[so + 1];
        out[o + 2] = px[so + 2];
      } else {
        const i = (ly * L.w + lx) * 3;
        out[o] = L.c[i];
        out[o + 1] = L.c[i + 1];
        out[o + 2] = L.c[i + 2];
      }
    }
  return out;
}

function downsample(px: Uint8ClampedArray, hole: Uint8Array, s: number, n: number): Level {
  const k = s / n;
  const c = new Float32Array(n * n * 3);
  const h = new Uint8Array(n * n);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      let r = 0, g = 0, b = 0, m = 0, any = 0;
      for (let yy = Math.floor(y * k); yy < Math.min(s, Math.ceil((y + 1) * k)); yy++)
        for (let xx = Math.floor(x * k); xx < Math.min(s, Math.ceil((x + 1) * k)); xx++) {
          if (hole[yy * s + xx]) any = 1;
          const o = (yy * s + xx) * 4;
          r += px[o];
          g += px[o + 1];
          b += px[o + 2];
          m++;
        }
      const i = y * n + x;
      c[i * 3] = r / m;
      c[i * 3 + 1] = g / m;
      c[i * 3 + 2] = b / m;
      h[i] = any;
    }
  return { w: n, h: n, c, hole: h };
}

function half(L: Level): Level {
  const w = Math.ceil(L.w / 2);
  const h = Math.ceil(L.h / 2);
  const c = new Float32Array(w * h * 3);
  const hole = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let m = 0;
      const acc = [0, 0, 0];
      let any = 0;
      for (let dy = 0; dy < 2; dy++)
        for (let dx = 0; dx < 2; dx++) {
          const sx = Math.min(L.w - 1, x * 2 + dx);
          const sy = Math.min(L.h - 1, y * 2 + dy);
          const i = sy * L.w + sx;
          if (L.hole[i]) any = 1;
          for (let k = 0; k < 3; k++) acc[k] += L.c[i * 3 + k];
          m++;
        }
      const i = y * w + x;
      for (let k = 0; k < 3; k++) c[i * 3 + k] = acc[k] / m;
      hole[i] = any;
    }
  return { w, h, c, hole };
}

/** A source patch must lie wholly inside the window and contain no hole pixel. */
function isValid(L: Level, x: number, y: number) {
  if (x < R || y < R || x >= L.w - R || y >= L.h - R) return false;
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (L.hole[(y + dy) * L.w + x + dx]) return false;
  return true;
}

function validCenters(L: Level) {
  const out: number[] = [];
  for (let y = R; y < L.h - R; y++) for (let x = R; x < L.w - R; x++) if (isValid(L, x, y)) out.push(y * L.w + x);
  return out;
}

/** Sum of squared differences between the patch at p and the source patch at q (clipped at the borders). */
function dist(L: Level, p: number, q: number, best: number) {
  const { w, h, c } = L;
  const px = p % w;
  const py = (p / w) | 0;
  const qx = q % w;
  const qy = (q / w) | 0;
  let d = 0;
  for (let dy = -R; dy <= R; dy++) {
    const ay = py + dy;
    if (ay < 0 || ay >= h) continue;
    for (let dx = -R; dx <= R; dx++) {
      const ax = px + dx;
      if (ax < 0 || ax >= w) continue;
      const a = (ay * w + ax) * 3;
      const b = ((qy + dy) * w + qx + dx) * 3;
      const e0 = c[a] - c[b];
      const e1 = c[a + 1] - c[b + 1];
      const e2 = c[a + 2] - c[b + 2];
      d += e0 * e0 + e1 * e1 + e2 * e2;
    }
    if (d > best) return d;
  }
  return d;
}

function patchMatch(L: Level, nnf: Int32Array, valid: number[], rand: () => number, it: number, near: number) {
  const { w, h } = L;
  const cost = new Float64Array(w * h);
  const far = (p: number, q: number) => {
    const dx = (p % w) - (q % w);
    const dy = ((p / w) | 0) - ((q / w) | 0);
    return near * (dx * dx + dy * dy);
  };
  for (let i = 0; i < w * h; i++) if (L.hole[i]) cost[i] = dist(L, i, nnf[i], Infinity) + far(i, nnf[i]);
  const fwd = it % 2 === 0;
  const step = fwd ? 1 : -1;
  for (let yy = 0; yy < h; yy++) {
    const y = fwd ? yy : h - 1 - yy;
    for (let xx = 0; xx < w; xx++) {
      const x = fwd ? xx : w - 1 - xx;
      const i = y * w + x;
      if (!L.hole[i]) continue;
      const tryCand = (q: number) => {
        if (q < 0 || q === nnf[i]) return;
        const qx = q % w;
        const qy = (q / w) | 0;
        if (!isValid(L, qx, qy)) return;
        const f = far(i, q);
        if (f >= cost[i]) return;
        const d = dist(L, i, q, cost[i] - f) + f;
        if (d < cost[i]) {
          cost[i] = d;
          nnf[i] = q;
        }
      };
      // Propagation from the left/up (or right/down) neighbour's match, shifted by one.
      const nx = x - step;
      if (nx >= 0 && nx < w && L.hole[y * w + nx] && nnf[y * w + nx] >= 0) {
        const q = nnf[y * w + nx];
        if ((q % w) + step >= 0 && (q % w) + step < w) tryCand(q + step);
      }
      const ny = y - step;
      if (ny >= 0 && ny < h && L.hole[ny * w + x] && nnf[ny * w + x] >= 0) {
        const q = nnf[ny * w + x];
        const qy = ((q / w) | 0) + step;
        if (qy >= 0 && qy < h) tryCand(q + step * w);
      }
      // Random search around the current match, shrinking radius; plus one global sample.
      const cx = nnf[i] % w;
      const cy = (nnf[i] / w) | 0;
      for (let r = Math.max(w, h); r >= 1; r >>= 1) {
        const sx = Math.round(cx + (rand() * 2 - 1) * r);
        const sy = Math.round(cy + (rand() * 2 - 1) * r);
        if (sx >= 0 && sy >= 0 && sx < w && sy < h) tryCand(sy * w + sx);
      }
      tryCand(valid[(rand() * valid.length) | 0]);
    }
  }
}

/** Hole colours from the surroundings by push–pull (average down a pyramid, interpolate back up). */
function smoothGuess(L: Level) {
  const { w, h } = L;
  type P = { w: number; h: number; c: Float32Array; k: Float32Array };
  const base: P = { w, h, c: new Float32Array(L.c), k: new Float32Array(w * h) };
  for (let i = 0; i < w * h; i++) base.k[i] = L.hole[i] ? 0 : 1;
  const pyr: P[] = [base];
  while (pyr[pyr.length - 1].w > 1 || pyr[pyr.length - 1].h > 1) {
    const f = pyr[pyr.length - 1];
    const nw = Math.ceil(f.w / 2);
    const nh = Math.ceil(f.h / 2);
    const q: P = { w: nw, h: nh, c: new Float32Array(nw * nh * 3), k: new Float32Array(nw * nh) };
    for (let y = 0; y < f.h; y++)
      for (let x = 0; x < f.w; x++) {
        const a = y * f.w + x;
        if (!f.k[a]) continue;
        const b = (y >> 1) * nw + (x >> 1);
        for (let t = 0; t < 3; t++) q.c[b * 3 + t] += f.c[a * 3 + t] * f.k[a];
        q.k[b] += f.k[a];
      }
    for (let b = 0; b < nw * nh; b++) {
      if (q.k[b]) for (let t = 0; t < 3; t++) q.c[b * 3 + t] /= q.k[b];
      q.k[b] = Math.min(1, q.k[b]);
    }
    pyr.push(q);
  }
  for (let li = pyr.length - 2; li >= 0; li--) {
    const f = pyr[li];
    const c = pyr[li + 1];
    for (let y = 0; y < f.h; y++)
      for (let x = 0; x < f.w; x++) {
        const a = y * f.w + x;
        if (f.k[a] >= 1) continue;
        const b = Math.min(c.h - 1, y >> 1) * c.w + Math.min(c.w - 1, x >> 1);
        for (let t = 0; t < 3; t++) f.c[a * 3 + t] = f.c[a * 3 + t] * f.k[a] + c.c[b * 3 + t] * (1 - f.k[a]);
        f.k[a] = 1;
      }
  }
  for (let i = 0; i < w * h; i++) if (L.hole[i]) for (let t = 0; t < 3; t++) L.c[i * 3 + t] = base.c[i * 3 + t];
}

/** Each hole pixel becomes the average of what overlapping patches' matches say it should be. */
function vote(L: Level, nnf: Int32Array) {
  const { w, h, c } = L;
  const acc = new Float32Array(w * h * 3);
  const n = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if (!L.hole[p] || nnf[p] < 0) continue;
      const qx = nnf[p] % w;
      const qy = (nnf[p] / w) | 0;
      for (let dy = -R; dy <= R; dy++)
        for (let dx = -R; dx <= R; dx++) {
          const ax = x + dx;
          const ay = y + dy;
          if (ax < 0 || ay < 0 || ax >= w || ay >= h) continue;
          const a = ay * w + ax;
          if (!L.hole[a]) continue;
          const b = ((qy + dy) * w + qx + dx) * 3;
          acc[a * 3] += c[b];
          acc[a * 3 + 1] += c[b + 1];
          acc[a * 3 + 2] += c[b + 2];
          n[a]++;
        }
    }
  for (let i = 0; i < w * h; i++) if (L.hole[i] && n[i]) for (let k = 0; k < 3; k++) c[i * 3 + k] = acc[i * 3 + k] / n[i];
}
