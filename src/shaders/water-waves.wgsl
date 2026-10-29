// Water surface waves (Aşama 9c), evaluated where a primary ray meets the top water voxel
// (primary.wgsl): the surface lies at `level` inside that voxel plus this height field.
//
//   swell     WAVE_COUNT Gerstner waves travelling with the wind (directions spread around
//             it, wavelengths falling geometrically from `wavelength`, deep-water dispersion
//             ω = √(g·k), equal steepness); the normal is Gerstner's (crests sharpened by
//             `WAVE_STEEPNESS`), the height their sum of sines (a height field the ray can
//             intersect; the horizontal displacement is left out).
//   ripples   two octaves of the detail noise volume drifting with the wind; their gradient
//             by central differences tilts the normal (they are too small to move the
//             intersection).
//
// Every component fades out once its wavelength drops below a few pixel footprints, so
// distant water neither aliases nor sparkles at random. Needs material.wgsl (fastNoise) and
// `water: WaterParams`.

const WAVE_COUNT: u32 = 4u;
/// 2π (primary.wgsl, the includer, has no PI constant: that comes with the sky shaders).
const WAVE_TAU: f32 = 6.28318530718;
const GRAVITY: f32 = 9.81;
/// Gerstner steepness Q·k·A summed over the waves stays below 1 (no loops in the surface).
const WAVE_STEEPNESS: f32 = 0.6;
/// Ratio between successive wavelengths.
const WAVE_LENGTH_FALLOFF: f32 = 0.57;
/// Ripple feature size of the first octave (metres).
const RIPPLE_SIZE: f32 = 0.18;

struct WaveSample {
  /// Height above the rest level (metres).
  height: f32,
  normal: vec3f,
};

/// A component is gone once its wavelength is shorter than this many pixel footprints
/// (fading from 4× that): water has a short TAAU history, and waves a few pixels long
/// still shimmered in a distant lake at 4.
const WAVE_FADE_PIXELS: f32 = 8.0;

/// Fade of a component of wavelength `lambda` for a pixel footprint `footprint` (metres).
fn waveFade(lambda: f32, footprint: f32) -> f32 {
  return 1.0 - smoothstep(0.25 * lambda, lambda, WAVE_FADE_PIXELS * footprint);
}

/// Ripples (normal detail only: the sun's glitter) fade later, at four footprints: near
/// water keeps its broken, sparkling highlight; once they are sub-pixel the glint turns
/// into the smooth lobe of waterSlopeVariance. At eight, close water lost its sparkle too.
const RIPPLE_FADE_PIXELS: f32 = 4.0;

fn rippleFade(size: f32, footprint: f32) -> f32 {
  return 1.0 - smoothstep(0.25 * size, size, RIPPLE_FADE_PIXELS * footprint);
}

/// Smallest |dir.y| the footprint stretch uses (at most √20 ≈ 4.5×).
const FOOTPRINT_MIN_SLANT: f32 = 0.05;

/// Pixel footprint (metres) on the surface for a ray `dir` meeting it at distance `t`: at a
/// grazing angle the pixel's ellipse is 1/|dir.y| longer along the view than across; the
/// footprint is the geometric mean of the two axes (the ellipse's area) — the across-view
/// width alone left sub-pixel waves in far water, the long axis smoothed near water too
/// (a low camera sees even close water at a grazing angle) and blew the glint into a blob.
fn waterFootprint(t: f32, dir: vec3f) -> f32 {
  return max(t, 1.0) * material_params.pixel_spread / sqrt(max(abs(dir.y), FOOTPRINT_MIN_SLANT));
}

/// Footprints for waterWaves / waterSlopeVariance: x for the swell (waterFootprint), y for
/// the ripples — the pixel's width across the view only: they make the sun's glitter, and
/// with the ellipse's stretch even close water (a low camera sees it at a grazing angle)
/// lost its sparkle to a smooth lobe. They still fade out between ~5 and ~25 m.
fn waterFootprints(t: f32, dir: vec3f) -> vec2f {
  return vec2f(waterFootprint(t, dir), max(t, 1.0) * material_params.pixel_spread);
}

/// Slope variance of the wave and ripple detail faded out at `footprint` (Toksvig-like):
/// the sun's highlight widens by it, so distant water keeps its glitter path as a broad
/// lobe instead of sparkling pixels. Mean squared slope of a sine of slope amplitude s: s²/2.
fn waterSlopeVariance(footprint: vec2f) -> f32 {
  var v = 0.0;
  for (var i = 0u; i < WAVE_COUNT; i++) {
    let lambda = water.wavelength * pow(WAVE_LENGTH_FALLOFF, f32(i));
    let k = WAVE_TAU / lambda;
    let s = k * water.wave_amplitude * (lambda / water.wavelength) * (1.0 - waveFade(lambda, footprint.x));
    v += 0.5 * s * s;
  }
  for (var o = 0u; o < 2u; o++) {
    let size = RIPPLE_SIZE * pow(0.5, f32(o));
    // Noise gradients are ~±0.5 per unit (see waterWaves): slope amplitude ~ half of it.
    let s = 0.5 * water.wave_amplitude * WAVE_TAU / size * water.ripple * (1.0 - rippleFade(size, footprint.y));
    v += 0.5 * s * s;
  }
  return v;
}

fn rotateByAngle(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

/// Swell height above the rest level at `xz` (the ripples do not move the surface): what
/// the ray intersection iterates on, without the normal and its noise lookups.
fn waterWaveHeight(xz: vec2f, footprint: f32) -> f32 {
  var h = 0.0;
  var spread = array<f32, 4>(0.0, 0.55, -0.4, 1.0);
  for (var i = 0u; i < WAVE_COUNT; i++) {
    let lambda = water.wavelength * pow(WAVE_LENGTH_FALLOFF, f32(i));
    let fade = waveFade(lambda, footprint);
    if (fade <= 0.0) {
      break;
    }
    let d = rotateByAngle(water.wind_dir, spread[i]);
    let k = WAVE_TAU / lambda;
    let a = water.wave_amplitude * (lambda / water.wavelength) * fade;
    h += a * sin(k * dot(d, xz) - sqrt(GRAVITY * k) * water.time + f32(i) * 1.7);
  }
  return h;
}

/// Waves at world position `xz` (metres), seen with pixel footprint `footprint`.
fn waterWaves(xz: vec2f, footprint: vec2f) -> WaveSample {
  var h = 0.0;
  // Gerstner normal: (−Σ D·k·A·cos, 1 − Σ Q·k·A·sin, …) in (x, y, z).
  var nxz = vec2f(0.0);
  var ny = 1.0;
  var spread = array<f32, 4>(0.0, 0.55, -0.4, 1.0);
  for (var i = 0u; i < WAVE_COUNT; i++) {
    let lambda = water.wavelength * pow(WAVE_LENGTH_FALLOFF, f32(i));
    let fade = waveFade(lambda, footprint.x);
    if (fade <= 0.0) {
      break; // shorter ones fade even more
    }
    let d = rotateByAngle(water.wind_dir, spread[i]);
    let k = WAVE_TAU / lambda;
    let a = water.wave_amplitude * (lambda / water.wavelength) * fade;
    let phase = k * dot(d, xz) - sqrt(GRAVITY * k) * water.time + f32(i) * 1.7;
    let s = sin(phase);
    let c = cos(phase);
    h += a * s;
    nxz -= d * (k * a * c);
    ny -= WAVE_STEEPNESS / f32(WAVE_COUNT) * k * a * s;
  }
  // Ripples: the noise field (period 16 units) scaled to RIPPLE_SIZE features, drifting
  // with the wind and evolving slowly.
  let drift = water.wind_dir * water.time * 0.3;
  for (var o = 0u; o < 2u; o++) {
    let size = RIPPLE_SIZE * pow(0.5, f32(o));
    let fade = rippleFade(size, footprint.y) * water.ripple;
    if (fade <= 0.0) {
      break;
    }
    let scale = 0.25 / size; // one noise texel (0.25 units) per feature
    let q = (xz + drift * (1.0 + f32(o))) * scale;
    let z = water.time * 0.4 + f32(o) * 5.3;
    let e = 0.5;
    let gx = fastNoise(vec3f(q.x + e, q.y, z)) - fastNoise(vec3f(q.x - e, q.y, z));
    let gz = fastNoise(vec3f(q.x, q.y + e, z)) - fastNoise(vec3f(q.x, q.y - e, z));
    // Slope amplitude ∝ wind like the swell's; noise differences are ~±0.5 per unit.
    let slope = water.wave_amplitude * WAVE_TAU / size * fade;
    nxz -= vec2f(gx, gz) / (2.0 * e) * slope;
  }
  return WaveSample(h, normalize(vec3f(nxz.x, max(ny, 0.2), nxz.y)));
}
