/**
 * Small float image toolkit for the texture packer. Pure (no I/O, no sharp) so it can be
 * unit tested. Images are row-major, `channels` floats per pixel, values usually in [0, 1].
 */

export interface FloatImage {
  width: number;
  height: number;
  channels: number;
  data: Float32Array;
}

export function createImage(width: number, height: number, channels: number, fill = 0): FloatImage {
  return { width, height, channels, data: new Float32Array(width * height * channels).fill(fill) };
}

export function fromBytes(bytes: Uint8Array, width: number, height: number, channels: number): FloatImage {
  const img = createImage(width, height, channels);
  for (let i = 0; i < img.data.length; i++) img.data[i] = bytes[i]! / 255;
  return img;
}

export function toBytes(img: FloatImage): Uint8Array {
  const out = new Uint8Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.round(Math.min(1, Math.max(0, img.data[i]!)) * 255);
  return out;
}

export function channel(img: FloatImage, c: number): FloatImage {
  const out = createImage(img.width, img.height, 1);
  for (let i = 0; i < img.width * img.height; i++) out.data[i] = img.data[i * img.channels + c]!;
  return out;
}

/** Circular shift by (dx, dy) pixels. A rolled tileable image stays tileable. */
export function roll(img: FloatImage, dx: number, dy: number): FloatImage {
  const { width: w, height: h, channels: c } = img;
  const out = createImage(w, h, c);
  for (let y = 0; y < h; y++) {
    const sy = (((y - dy) % h) + h) % h;
    for (let x = 0; x < w; x++) {
      const sx = (((x - dx) % w) + w) % w;
      const o = (y * w + x) * c;
      const s = (sy * w + sx) * c;
      for (let k = 0; k < c; k++) out.data[o + k] = img.data[s + k]!;
    }
  }
  return out;
}

/**
 * Makes an image tileable by blending it with a copy rolled by half its size: the rolled
 * copy wraps seamlessly at the borders, the original covers its own seam in the middle.
 * `band`: width of the cross-fade zone as a fraction of the size (≤ 0.5).
 * `isNormal`: channels 0-2 are a normal vector, renormalised after blending.
 */
export function seamBlend(img: FloatImage, band = 0.25, isNormal = false): FloatImage {
  const { width: w, height: h, channels: c } = img;
  const rolled = roll(img, w >> 1, h >> 1);
  const out = createImage(w, h, c);
  const ramp = (i: number, n: number) => Math.min(1, Math.min(i + 0.5, n - i - 0.5) / (band * n));
  for (let y = 0; y < h; y++) {
    const wy = ramp(y, h);
    for (let x = 0; x < w; x++) {
      // Smoothstep of the product avoids visible kinks at the edge of the fade zone.
      const t = Math.min(1, ramp(x, w) * wy);
      const wgt = t * t * (3 - 2 * t);
      const o = (y * w + x) * c;
      for (let k = 0; k < c; k++) out.data[o + k] = img.data[o + k]! * wgt + rolled.data[o + k]! * (1 - wgt);
      if (isNormal) normalize3(out.data, o);
    }
  }
  return out;
}

function normalize3(d: Float32Array, o: number): void {
  const len = Math.hypot(d[o]!, d[o + 1]!, d[o + 2]!) || 1;
  d[o] = d[o]! / len;
  d[o + 1] = d[o + 1]! / len;
  d[o + 2] = d[o + 2]! / len;
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Tileable fractal value noise in [0, 1] of size `size`², `cells` lattice cells across
 * (doubling per octave, so the pattern repeats exactly at the image border).
 */
export function tileableNoise(size: number, cells: number, octaves: number, seed: number): FloatImage {
  const out = createImage(size, size, 1);
  let amp = 1;
  let total = 0;
  for (let o = 0; o < octaves; o++) {
    const n = cells << o;
    const r = rng(seed * 7919 + o);
    const lattice = Float32Array.from({ length: n * n }, () => r());
    const at = (x: number, y: number) => lattice[(((y % n) + n) % n) * n + (((x % n) + n) % n)]!;
    for (let y = 0; y < size; y++) {
      const fy = (y / size) * n;
      const y0 = Math.floor(fy);
      const ty = smooth(fy - y0);
      for (let x = 0; x < size; x++) {
        const fx = (x / size) * n;
        const x0 = Math.floor(fx);
        const tx = smooth(fx - x0);
        const a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx;
        const b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
        out.data[y * size + x]! += (a + (b - a) * ty) * amp;
      }
    }
    total += amp;
    amp *= 0.5;
  }
  for (let i = 0; i < out.data.length; i++) out.data[i]! /= total;
  return out;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Tangent-space normal map (xyz in [-1, 1]) from a tileable height field. */
export function normalsFromHeight(height: FloatImage, strength: number): FloatImage {
  const { width: w, height: h } = height;
  const out = createImage(w, h, 3);
  const at = (x: number, y: number) => height.data[((y + h) % h) * w + ((x + w) % w)]!;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // Image y grows downwards; OpenGL-style normal maps have +Y pointing up.
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y - 1) - at(x, y + 1)) * strength;
      const o = (y * w + x) * 3;
      out.data[o] = -dx;
      out.data[o + 1] = -dy;
      out.data[o + 2] = 1;
      normalize3(out.data, o);
    }
  }
  return out;
}

/** A sprite to stamp: RGBA colour (straight alpha) plus aligned per-pixel extra maps. */
export interface Sprite {
  color: FloatImage; // 4 channels, alpha = opacity
  normal: FloatImage; // 3 channels, vector in [-1, 1]
  roughness: FloatImage; // 1 channel
}

export interface StampTarget {
  color: FloatImage; // 4 channels
  normal: FloatImage; // 3 channels
  roughness: FloatImage;
  ao: FloatImage;
  height: FloatImage;
}

/**
 * Stamps a sprite rotated by `angle` (radians) and scaled so its larger side spans
 * `size` pixels, centred at (cx, cy), wrapping around the target's edges (tileable).
 * Normal vectors are rotated with the sprite. `shade` multiplies colour, `level` is
 * written as height / ambient occlusion for the covered pixels.
 */
export function stamp(
  target: StampTarget,
  sprite: Sprite,
  cx: number,
  cy: number,
  size: number,
  angle: number,
  shade: [number, number, number],
  level: number,
): void {
  const { width: tw, height: th } = target.color;
  const { width: sw, height: sh } = sprite.color;
  const scale = size / Math.max(sw, sh);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const half = Math.ceil(size / 2) + 1;
  for (let dy = -half; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++) {
      // Inverse rotation into sprite space.
      const sx = Math.floor((cos * dx + sin * dy) / scale + sw / 2);
      const sy = Math.floor((-sin * dx + cos * dy) / scale + sh / 2);
      if (sx < 0 || sy < 0 || sx >= sw || sy >= sh) continue;
      const si = sy * sw + sx;
      const a = sprite.color.data[si * 4 + 3]!;
      if (a <= 0.01) continue;
      const tx = (((Math.round(cx) + dx) % tw) + tw) % tw;
      const ty = (((Math.round(cy) + dy) % th) + th) % th;
      const ti = ty * tw + tx;
      for (let k = 0; k < 3; k++) {
        const c = target.color.data[ti * 4 + k]!;
        target.color.data[ti * 4 + k] = c + (sprite.color.data[si * 4 + k]! * shade[k]! - c) * a;
      }
      target.color.data[ti * 4 + 3] = Math.max(target.color.data[ti * 4 + 3]!, a);
      // Rotate the sprite's normal (x right, y up) by the stamp angle (image y is down).
      const nx = sprite.normal.data[si * 3]!;
      const ny = sprite.normal.data[si * 3 + 1]!;
      const rx = cos * nx + sin * ny;
      const ry = -sin * nx + cos * ny;
      const n = target.normal.data;
      n[ti * 3] = n[ti * 3]! + (rx - n[ti * 3]!) * a;
      n[ti * 3 + 1] = n[ti * 3 + 1]! + (ry - n[ti * 3 + 1]!) * a;
      n[ti * 3 + 2] = n[ti * 3 + 2]! + (sprite.normal.data[si * 3 + 2]! - n[ti * 3 + 2]!) * a;
      normalize3(n, ti * 3);
      const r = target.roughness.data;
      r[ti] = r[ti]! + (sprite.roughness.data[si]! - r[ti]!) * a;
      target.ao.data[ti] = target.ao.data[ti]! + (level - target.ao.data[ti]!) * a;
      target.height.data[ti] = target.height.data[ti]! + (level - target.height.data[ti]!) * a;
    }
  }
}

/** LabPBR-style channel encodings used by the pack (see docs in build-textures.ts). */
export const labPbr = {
  /** Perceptual smoothness from linear roughness. */
  smoothness: (roughness: number) => 1 - Math.sqrt(Math.min(1, Math.max(0, roughness))),
  /** F0 channel: dielectric reflectance 0.04 (≈ 10/255) or 255 for metals. */
  f0: (metal: number) => (metal >= 0.5 ? 255 : Math.round(0.04 * 255)),
  /** Emission in 0..254; 255 means "no emission" (so an opaque-white alpha default is inert). */
  emission: (e: number) => (e <= 0 ? 255 : Math.min(254, Math.round(e * 254))),
};
