/**
 * Scalar spatiotemporal blue noise (STBN, Wolfe et al. 2022, "Spatiotemporal Blue Noise
 * Masks") by void-and-cluster (Ulichney 1993) over a size × size × depth volume.
 *
 * The energy is separable, as in the STBN paper: two points interact spatially when they
 * are in the same slice and temporally when they are at the same pixel. Each slice is then
 * a 2D blue noise mask and every pixel's sequence over the slices is 1D blue noise, so a
 * pixel's samples over consecutive frames are well spread (fast temporal convergence) and
 * the error of one frame is high-frequency on screen (easy to filter).
 *
 * Returns ranks in [0, size²·depth) laid out as [t][y][x]; divide by the count for values
 * in [0, 1). Deterministic for a given seed.
 */
export function spatiotemporalBlueNoise(size: number, depth: number, seed: number, sigmaSpace = 1.9, sigmaTime = 1.9): Uint32Array {
  const area = size * size;
  const count = area * depth;
  const gs = new Float32Array(area);
  for (let dy = 0; dy < size; dy++) {
    for (let dx = 0; dx < size; dx++) {
      const x = Math.min(dx, size - dx);
      const y = Math.min(dy, size - dy);
      gs[dy * size + dx] = Math.exp(-(x * x + y * y) / (2 * sigmaSpace * sigmaSpace));
    }
  }
  const gt = new Float32Array(depth);
  for (let dt = 0; dt < depth; dt++) {
    const t = Math.min(dt, depth - dt);
    gt[dt] = Math.exp(-(t * t) / (2 * sigmaTime * sigmaTime));
  }

  /** Adds `sign` × the kernel of point i to `energy`. */
  const splat = (energy: Float32Array, i: number, sign: number) => {
    const t = Math.floor(i / area);
    const p = i - t * area;
    const px = p % size;
    const py = (p - px) / size;
    const base = t * area;
    for (let y = 0; y < size; y++) {
      const row = ((y - py + size) % size) * size;
      const out = base + y * size;
      for (let x = 0; x < size; x++) energy[out + x]! += sign * gs[row + ((x - px + size) % size)]!;
    }
    for (let s = 0; s < depth; s++) energy[s * area + p]! += sign * gt[(s - t + depth) % depth]!;
  };
  const extreme = (energy: Float32Array, pattern: Uint8Array, value: number, max: boolean) => {
    let best = -1;
    let bestE = max ? -Infinity : Infinity;
    for (let i = 0; i < count; i++) {
      if (pattern[i] !== value) continue;
      const e = energy[i]!;
      if (max ? e > bestE : e < bestE) {
        bestE = e;
        best = i;
      }
    }
    return best;
  };

  // Deterministic PRNG (mulberry32).
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let r = Math.imul(state ^ (state >>> 15), 1 | state);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };

  // Initial pattern: ~10 % random points, relaxed until no point moves.
  const initial = new Uint8Array(count);
  const energy = new Float32Array(count);
  const ones = Math.max(1, Math.floor(count / 10));
  for (let placed = 0; placed < ones; ) {
    const i = Math.floor(random() * count);
    if (initial[i]) continue;
    initial[i] = 1;
    splat(energy, i, 1);
    placed++;
  }
  for (;;) {
    const cluster = extreme(energy, initial, 1, true);
    initial[cluster] = 0;
    splat(energy, cluster, -1);
    const void_ = extreme(energy, initial, 0, false);
    initial[void_] = 1;
    splat(energy, void_, 1);
    if (void_ === cluster) break;
  }

  const rank = new Uint32Array(count);
  // Phase 1: ranks below the initial count, removing the tightest clusters.
  const pattern = initial.slice();
  const e1 = energy.slice();
  for (let r = ones - 1; r >= 0; r--) {
    const cluster = extreme(e1, pattern, 1, true);
    pattern[cluster] = 0;
    splat(e1, cluster, -1);
    rank[cluster] = r;
  }
  // Phase 2: up to half, filling the largest voids.
  pattern.set(initial);
  const e2 = energy;
  const half = Math.floor(count / 2);
  let r = ones;
  for (; r < half; r++) {
    const void_ = extreme(e2, pattern, 0, false);
    pattern[void_] = 1;
    splat(e2, void_, 1);
    rank[void_] = r;
  }
  // Phase 3: the rest, taking the tightest cluster of the remaining zeros (minority).
  const e3 = new Float32Array(count);
  for (let i = 0; i < count; i++) if (!pattern[i]) splat(e3, i, 1);
  for (; r < count; r++) {
    const cluster = extreme(e3, pattern, 0, true);
    pattern[cluster] = 1;
    splat(e3, cluster, -1);
    rank[cluster] = r;
  }
  return rank;
}
