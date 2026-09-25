import { describe, expect, it } from 'vitest';
import { Spline } from './spline';

describe('Spline', () => {
  const s = new Spline([[-1, 0], [0, 10], [1, 100]]);

  it('passes through control points and clamps outside the range', () => {
    expect(s.evaluate(-1)).toBe(0);
    expect(s.evaluate(0)).toBeCloseTo(10);
    expect(s.evaluate(1)).toBe(100);
    expect(s.evaluate(-5)).toBe(0);
    expect(s.evaluate(5)).toBe(100);
  });

  it('never overshoots monotone data', () => {
    const steps = new Spline([[0, 0], [1, 0], [2, 50], [3, 51], [4, 200]]);
    let prev = -Infinity;
    for (let x = 0; x <= 4; x += 0.01) {
      const y = steps.evaluate(x);
      expect(y).toBeGreaterThanOrEqual(prev - 1e-9);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(200);
      prev = y;
    }
  });

  it('rejects invalid control points', () => {
    expect(() => new Spline([[0, 0]])).toThrow();
    expect(() => new Spline([[0, 0], [0, 1]])).toThrow();
  });
});
