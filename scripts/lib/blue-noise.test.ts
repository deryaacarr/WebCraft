import { describe, expect, it } from 'vitest';
import { spatiotemporalBlueNoise } from './blue-noise.ts';

describe('spatiotemporalBlueNoise', () => {
  const size = 16;
  const depth = 4;
  const rank = spatiotemporalBlueNoise(size, depth, 1234);
  const area = size * size;

  it('ranks every texel exactly once', () => {
    const sorted = Array.from(rank).sort((a, b) => a - b);
    expect(sorted).toEqual(Array.from({ length: area * depth }, (_, i) => i));
  });

  it('is deterministic for a seed', () => {
    expect(spatiotemporalBlueNoise(size, depth, 1234)).toEqual(rank);
  });

  it('anti-correlates neighbours in space and in time (blue noise)', () => {
    const v = (t: number, y: number, x: number) => rank[t * area + y * size + x]! / (area * depth) - 0.5;
    let space = 0;
    let time = 0;
    let norm = 0;
    for (let t = 0; t < depth; t++) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          space += v(t, y, x) * v(t, y, (x + 1) % size);
          time += v(t, y, x) * v((t + 1) % depth, y, x);
          norm += v(t, y, x) ** 2;
        }
      }
    }
    expect(space / norm).toBeLessThan(-0.05);
    expect(time / norm).toBeLessThan(-0.05);
  });
});
