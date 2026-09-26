import { describe, expect, it } from 'vitest';
import { kelvinToLinearRgb, planckianXy } from './color-temperature';

describe('colour temperature', () => {
  it('hits known points of the Planckian locus', () => {
    const [x, y] = planckianXy(6500);
    expect(x).toBeCloseTo(0.3135, 3);
    expect(y).toBeCloseTo(0.3236, 3);
    const [xa, ya] = planckianXy(2856); // CIE illuminant A
    expect(xa).toBeCloseTo(0.4476, 2);
    expect(ya).toBeCloseTo(0.4074, 2);
  });

  it('is near white around 6500 K, warm below and blue above', () => {
    const d65 = kelvinToLinearRgb(6500);
    for (const c of d65) {
      expect(c).toBeGreaterThan(0.85);
      expect(c).toBeLessThan(1.15);
    }
    const warm = kelvinToLinearRgb(3000);
    expect(warm[0]).toBeGreaterThan(warm[2] * 2);
    const cold = kelvinToLinearRgb(12000);
    expect(cold[2]).toBeGreaterThan(cold[0]);
  });
});
