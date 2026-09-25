/** Deterministic 32-bit integer hash of (seed, x, y, z); same result on every thread/machine. */
export function hash3(seed: number, x: number, y: number, z: number): number {
  let h = seed | 0;
  h = Math.imul(h ^ Math.imul(x | 0, 0x27d4eb2d), 0x85ebca6b);
  h = Math.imul(h ^ Math.imul(y | 0, 0x165667b1), 0xc2b2ae35);
  h = Math.imul(h ^ Math.imul(z | 0, 0x9e3779b1), 0x85ebca6b);
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
}

/** hash3 mapped to [0, 1). */
export function hash01(seed: number, x: number, y: number, z: number): number {
  return hash3(seed, x, y, z) / 4294967296;
}
