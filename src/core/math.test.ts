import { describe, expect, it } from 'vitest';
import {
  forwardFromYawPitch,
  halton,
  identity,
  invert,
  jitterProjection,
  multiply,
  perspective,
  transformVec4,
  viewRotation,
} from './math';

const close = (a: ArrayLike<number>, b: ArrayLike<number>, eps = 1e-5) => {
  for (let i = 0; i < b.length; i++) expect(a[i]).toBeCloseTo(b[i]!, -Math.log10(eps));
};

describe('math', () => {
  it('inverts matrices', () => {
    const m = multiply(perspective(1.2, 1.6, 0.1, 500), viewRotation(0.7, -0.3));
    close(multiply(m, invert(m)), identity(), 1e-4);
  });

  it('projects with WebGPU depth: near → 0, far → 1, looking down −Z', () => {
    const p = perspective(Math.PI / 2, 1, 1, 100);
    const n = transformVec4(p, [0, 0, -1, 1]);
    const f = transformVec4(p, [0, 0, -100, 1]);
    expect(n[2] / n[3]).toBeCloseTo(0);
    expect(f[2] / f[3]).toBeCloseTo(1);
  });

  it('builds a view rotation that maps forward to −Z and keeps up up', () => {
    for (const [yaw, pitch] of [[0, 0], [1.3, 0.4], [-2.5, -1.2]] as const) {
      const f = forwardFromYawPitch(yaw, pitch);
      const v = viewRotation(yaw, pitch);
      close(transformVec4(v, [...f, 0]), [0, 0, -1, 0]);
      const up = transformVec4(v, [0, 1, 0, 0]);
      expect(up[1]).toBeGreaterThan(0); // world up stays in the upper half of the view
    }
    close(forwardFromYawPitch(0, 0), [0, 0, -1]);
    close(forwardFromYawPitch(Math.PI / 2, 0), [-1, 0, 0]);
  });

  it('jitters by exactly the requested pixel offset', () => {
    const p = perspective(1, 2, 0.1, 100);
    const w = 200;
    const h = 100;
    const point: [number, number, number, number] = [0.3, -0.2, -5, 1];
    const a = transformVec4(p, point);
    const b = transformVec4(jitterProjection(p, 0.5, 0.25, w, h), point);
    // NDC → pixels: x right (w/2 per unit), y down (h/2 per unit).
    expect((b[0] / b[3] - a[0] / a[3]) * (w / 2)).toBeCloseTo(0.5);
    expect(-(b[1] / b[3] - a[1] / a[3]) * (h / 2)).toBeCloseTo(0.25);
  });

  it('generates the Halton sequence', () => {
    expect([1, 2, 3, 4].map((i) => halton(i, 2))).toEqual([0.5, 0.25, 0.75, 0.125]);
    const b3 = [1, 2, 3, 4].map((i) => halton(i, 3));
    close(b3, [1 / 3, 2 / 3, 1 / 9, 4 / 9]);
  });
});
