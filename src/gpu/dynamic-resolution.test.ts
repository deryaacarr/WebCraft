import { describe, expect, it } from 'vitest';
import { DynamicResolution, type DynamicResolutionConfig } from './dynamic-resolution';

const cfg: DynamicResolutionConfig = { enabled: true, targetMs: 16, minScale: 0.35, maxScale: 0.6 };

/** Runs `frames` frames at 30 fps with GPU time = k · scale² (+ optional spikes); returns
 *  the scale per frame. Measurements arrive `latency` frames late, like the profiler's. */
function simulate(dr: DynamicResolution, frames: number, k: (frame: number) => number, latency = 3, spike?: (frame: number) => number) {
  const history: number[] = [];
  const scales: number[] = [];
  for (let f = 0; f < frames; f++) {
    history.push(k(f) * dr.scale * dr.scale + (spike?.(f) ?? 0));
    const measured = history[f - latency];
    if (measured !== undefined) dr.update(measured, f / 30, cfg);
    scales.push(dr.scale);
  }
  return scales;
}

const directionChanges = (scales: number[]) => {
  let changes = 0;
  let last = 0;
  for (let i = 1; i < scales.length; i++) {
    const d = Math.sign(scales[i]! - scales[i - 1]!);
    if (d !== 0) {
      if (last !== 0 && d !== last) changes++;
      last = d;
    }
  }
  return changes;
};

describe('DynamicResolution', () => {
  it('drops the scale when the frame is over budget and settles under the target', () => {
    // 20.5 ms at 0.5 → k = 82.
    const dr = new DynamicResolution(0.5);
    const scales = simulate(dr, 300, () => 82);
    const final = scales[scales.length - 1]!;
    expect(final).toBeLessThan(0.5);
    expect(82 * final * final).toBeLessThanOrEqual(16 * 1.05);
    expect(82 * final * final).toBeGreaterThan(16 * 0.75);
    expect(directionChanges(scales)).toBe(0);
  });

  it('rises slowly when there is headroom, without overshooting into oscillation', () => {
    // 11.5 ms at 0.5 → k = 46: room up to ~0.59.
    const dr = new DynamicResolution(0.4);
    const scales = simulate(dr, 900, () => 46);
    // One step per second at most: 30 frames ≥ between rises.
    const rises = scales.map((s, i) => (i > 0 && s > scales[i - 1]! ? i : -1)).filter((i) => i >= 0);
    for (let i = 1; i < rises.length; i++) expect(rises[i]! - rises[i - 1]!).toBeGreaterThanOrEqual(30);
    const final = scales[scales.length - 1]!;
    expect(final).toBeGreaterThan(0.5);
    expect(46 * final * final).toBeLessThanOrEqual(16 * 1.05);
    expect(directionChanges(scales)).toBe(0);
  });

  it('follows a load step down and back up with one reversal only', () => {
    const dr = new DynamicResolution(0.4);
    // Light scene, then 10 s of a heavy one, then light again.
    const scales = simulate(dr, 1200, (f) => (f >= 300 && f < 600 ? 100 : 60));
    expect(scales[599]!).toBeLessThan(scales[299]!);
    expect(scales[1199]!).toBeGreaterThan(scales[599]!);
    expect(directionChanges(scales)).toBeLessThanOrEqual(2);
  });

  it('ignores single-frame spikes', () => {
    const dr = new DynamicResolution(0.45);
    // Under target (~12 ms) with one 60 ms spike every 2 s (a chunk upload).
    const scales = simulate(dr, 600, () => 60, 3, (f) => (f % 60 === 30 ? 60 : 0));
    expect(Math.min(...scales)).toBeGreaterThanOrEqual(0.45);
  });

  it('stays within the configured range and holds when disabled', () => {
    const dr = new DynamicResolution(0.5);
    const scales = simulate(dr, 600, () => 400); // hopeless: 100 ms at 0.5
    expect(Math.min(...scales)).toBeCloseTo(0.35);
    const off = new DynamicResolution(0.45);
    for (let f = 0; f < 100; f++) off.update(100, f / 30, { ...cfg, enabled: false });
    expect(off.scale).toBe(0.45);
  });
});
