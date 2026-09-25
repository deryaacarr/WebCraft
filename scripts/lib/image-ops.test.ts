import { describe, expect, it } from 'vitest';
import { createImage, labPbr, normalsFromHeight, roll, seamBlend, stamp, tileableNoise, type Sprite } from './image-ops.ts';

/** Mean absolute difference between the last and first column (wrap-around seam). */
function seamError(img: { width: number; height: number; data: Float32Array }): number {
  let sum = 0;
  for (let y = 0; y < img.height; y++) sum += Math.abs(img.data[y * img.width]! - img.data[y * img.width + img.width - 1]!);
  return sum / img.height;
}

/** Mean absolute difference between neighbouring columns inside the image. */
function interiorStep(img: { width: number; height: number; data: Float32Array }): number {
  let sum = 0;
  for (let y = 0; y < img.height; y++) {
    for (let x = 1; x < img.width; x++) sum += Math.abs(img.data[y * img.width + x]! - img.data[y * img.width + x - 1]!);
  }
  return sum / (img.height * (img.width - 1));
}

describe('image ops', () => {
  it('seamBlend makes a non-tileable ramp tile without a visible edge', () => {
    const w = 64;
    const ramp = createImage(w, w, 1);
    for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) ramp.data[y * w + x] = x / (w - 1);
    expect(seamError(ramp)).toBeCloseTo(1); // the raw ramp jumps from 1 back to 0
    const tiled = seamBlend(ramp, 0.25);
    // The wrap-around step is now no larger than an ordinary step inside the image.
    expect(seamError(tiled)).toBeLessThanOrEqual(interiorStep(tiled) * 2);
  });

  it('seamBlend keeps normal vectors unit length', () => {
    const img = createImage(16, 16, 3);
    for (let i = 0; i < 256; i++) img.data.set(i % 2 ? [1, 0, 0] : [0, 0, 1], i * 3);
    const out = seamBlend(img, 0.25, true);
    for (let i = 0; i < 256; i++) expect(Math.hypot(out.data[i * 3]!, out.data[i * 3 + 1]!, out.data[i * 3 + 2]!)).toBeCloseTo(1, 5);
  });

  it('roll is a circular shift', () => {
    const img = createImage(4, 1, 1);
    img.data.set([1, 2, 3, 4]);
    expect([...roll(img, 1, 0).data]).toEqual([4, 1, 2, 3]);
  });

  it('tileableNoise repeats exactly across the border and is deterministic', () => {
    const a = tileableNoise(64, 4, 3, 7);
    const b = tileableNoise(64, 4, 3, 7);
    expect([...a.data]).toEqual([...b.data]);
    expect(seamError(a)).toBeLessThan(interiorStep(a) * 3);
    expect(Math.min(...a.data)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...a.data)).toBeLessThanOrEqual(1);
  });

  it('normalsFromHeight points +Y (OpenGL up) where height rises towards the top of the image', () => {
    const h = createImage(8, 8, 1);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) h.data[y * 8 + x] = (8 - y) / 8; // higher at the top
    const n = normalsFromHeight(h, 4);
    const o = (4 * 8 + 4) * 3;
    expect(n.data[o + 1]).toBeLessThan(0); // surface tilts away from the higher side
    expect(n.data[o + 2]).toBeGreaterThan(0);
  });

  it('stamp wraps around edges and rotates normals with the sprite', () => {
    const size = 8;
    const sprite: Sprite = { color: createImage(size, size, 4, 1), normal: createImage(size, size, 3), roughness: createImage(size, size, 1, 0.3) };
    for (let i = 0; i < size * size; i++) sprite.normal.data.set([1, 0, 0], i * 3); // points right
    const t = {
      color: createImage(32, 32, 4),
      normal: createImage(32, 32, 3),
      roughness: createImage(32, 32, 1, 1),
      ao: createImage(32, 32, 1, 1),
      height: createImage(32, 32, 1),
    };
    for (let i = 0; i < 32 * 32; i++) t.normal.data.set([0, 0, 1], i * 3);
    stamp(t, sprite, 0, 0, size, Math.PI / 2, [1, 1, 1], 0.5); // centred on the corner
    const px = (x: number, y: number) => (y * 32 + x) * 4 + 3;
    expect(t.color.data[px(31, 31)]).toBe(1); // wrapped to the opposite corner
    expect(t.color.data[px(16, 16)]).toBe(0);
    // Rotated 90° clockwise on screen (image y points down): "right" becomes "down" (−Y).
    const n = (1 * 32 + 1) * 3;
    expect(t.normal.data[n]).toBeCloseTo(0, 5);
    expect(t.normal.data[n + 1]).toBeCloseTo(-1, 5);
  });

  it('encodes LabPBR-style channels', () => {
    expect(labPbr.smoothness(0)).toBe(1);
    expect(labPbr.smoothness(1)).toBe(0);
    expect(labPbr.smoothness(0.25)).toBeCloseTo(0.5);
    expect(labPbr.f0(0)).toBe(10);
    expect(labPbr.f0(1)).toBe(255);
    expect(labPbr.emission(0)).toBe(255);
    expect(labPbr.emission(1)).toBe(254);
  });
});
