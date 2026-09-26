import type { Vec3 } from './math';

/** CIE 1931 xy chromaticity of a black body at `kelvin` (Kim et al. 2002, 1667–25000 K). */
export function planckianXy(kelvin: number): [number, number] {
  const t = Math.min(25000, Math.max(1667, kelvin));
  const t2 = t * t;
  const t3 = t2 * t;
  const x =
    t <= 4000
      ? -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.17991
      : -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.24039;
  const x2 = x * x;
  const x3 = x2 * x;
  const y =
    t <= 2222
      ? -1.1063814 * x3 - 1.3481102 * x2 + 2.18555832 * x - 0.20219683
      : t <= 4000
        ? -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
        : 3.081758 * x3 - 5.8733867 * x2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}

/** Linear sRGB colour of a black-body illuminant, normalised to luminance 1. */
export function kelvinToLinearRgb(kelvin: number): Vec3 {
  const [x, y] = planckianXy(kelvin);
  const X = x / y;
  const Z = (1 - x - y) / y;
  const r = 3.2404542 * X - 1.5371385 - 0.4985314 * Z;
  const g = -0.969266 * X + 1.8760108 + 0.041556 * Z;
  const b = 0.0556434 * X - 0.2040259 + 1.0572252 * Z;
  const rgb: Vec3 = [Math.max(r, 1e-4), Math.max(g, 1e-4), Math.max(b, 1e-4)];
  const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  return [rgb[0] / lum, rgb[1] / lum, rgb[2] / lum];
}
