/** Control points `[input, output]`, sorted by input. */
export type SplinePoints = readonly (readonly [number, number])[];

/**
 * Monotone cubic (Fritsch–Carlson) interpolation through the control points.
 * Monotone data never overshoots, so a height curve cannot dip between two
 * rising points; inputs outside the range clamp to the end values.
 */
export class Spline {
  private readonly xs: Float64Array;
  private readonly ys: Float64Array;
  private readonly ms: Float64Array;

  constructor(points: SplinePoints) {
    if (points.length < 2) throw new Error('Spline needs at least 2 points');
    const n = points.length;
    this.xs = Float64Array.from(points, (p) => p[0]);
    this.ys = Float64Array.from(points, (p) => p[1]);
    for (let i = 1; i < n; i++) {
      if (this.xs[i]! <= this.xs[i - 1]!) throw new Error('Spline inputs must be strictly increasing');
    }

    const d = new Float64Array(n - 1);
    for (let i = 0; i < n - 1; i++) {
      d[i] = (this.ys[i + 1]! - this.ys[i]!) / (this.xs[i + 1]! - this.xs[i]!);
    }
    const m = new Float64Array(n);
    m[0] = d[0]!;
    m[n - 1] = d[n - 2]!;
    for (let i = 1; i < n - 1; i++) {
      m[i] = d[i - 1]! * d[i]! <= 0 ? 0 : (d[i - 1]! + d[i]!) / 2;
    }
    for (let i = 0; i < n - 1; i++) {
      if (d[i] === 0) {
        m[i] = 0;
        m[i + 1] = 0;
        continue;
      }
      const a = m[i]! / d[i]!;
      const b = m[i + 1]! / d[i]!;
      const s = a * a + b * b;
      if (s > 9) {
        const t = 3 / Math.sqrt(s);
        m[i] = t * a * d[i]!;
        m[i + 1] = t * b * d[i]!;
      }
    }
    this.ms = m;
  }

  evaluate(x: number): number {
    const { xs, ys, ms } = this;
    const n = xs.length;
    if (x <= xs[0]!) return ys[0]!;
    if (x >= xs[n - 1]!) return ys[n - 1]!;
    let i = 0;
    while (x > xs[i + 1]!) i++;
    const h = xs[i + 1]! - xs[i]!;
    const t = (x - xs[i]!) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return (
      (2 * t3 - 3 * t2 + 1) * ys[i]! +
      (t3 - 2 * t2 + t) * h * ms[i]! +
      (-2 * t3 + 3 * t2) * ys[i + 1]! +
      (t3 - t2) * h * ms[i + 1]!
    );
  }
}
