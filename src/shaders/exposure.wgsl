// Automatic exposure and white balance, measured on the lit, pre-exposed HDR frame.
//
// Exposure: a luminance histogram (log2) over a grid of samples, centre-weighted, with
// sky pixels nearly excluded; the darkest and brightest `trim` of the weight are dropped
// and the rest averaged. The exposure moves towards key / scene luminance in log space,
// slowly when the scene darkens (eyes open up) and faster when it brightens.
//
// White balance: the colour of the light falling on the scene (dominant light on open
// ground + sky irradiance) is partly neutralised with a von Kries transform in CAT02 LMS
// space and adapted slowly; `strength` < 1 keeps sunsets warm and nights blue. The
// resulting 3×3 matrix (linear sRGB → linear sRGB) is applied in the blit.
#include "sky.wgsl"
#include "gbuffer.wgsl"

struct ExposureParams {
  dt: f32,
  key: f32,
  compensation_ev: f32,
  trim: f32,
  min_exposure: f32,
  max_exposure: f32,
  /// Adaptation time constants (s): scene getting darker / brighter.
  tau_darker: f32,
  tau_brighter: f32,
  /// Gaussian centre weighting: sigma as a fraction of the half screen size.
  center_sigma: f32,
  /// Weight of sky pixels relative to ground (≈ excluded).
  sky_weight: f32,
  /// White balance: 0 off, 1 auto, 2 manual (illuminant from `manual_illuminant`).
  wb_mode: u32,
  wb_strength: f32,
  manual_illuminant: vec3f,
  wb_tau: f32,
};

/// Exposure state, also read by sky.wgsl (element 0) and the blit (matrix).
struct ExposureState {
  exposure: f32,
  // Scalars, not a vec3f: a vec3f would be 16-byte aligned and shift everything below.
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  /// Current (adapted) von Kries gains in LMS.
  wb_gains: vec4f,
  /// White-balance matrix columns for the blit.
  wb0: vec4f,
  wb1: vec4f,
  wb2: vec4f,
};

const THREADS: u32 = 256u;
const GRID: vec2u = vec2u(128u, 72u);
const BINS: u32 = 64u;
const LOG_MIN: f32 = -12.0;
const LOG_MAX: f32 = 8.0;
const WEIGHT_SCALE: f32 = 256.0;

// Linear sRGB ↔ CAT02 LMS (CAT02 · sRGB→XYZ, D65), columns; computed, not hand-typed.
const RGB_TO_LMS: mat3x3f = mat3x3f(
  vec3f(0.3905, 0.0709, 0.0231),
  vec3f(0.5499, 0.9631, 0.1280),
  vec3f(0.0089, 0.0014, 0.9361),
);
const LMS_TO_RGB: mat3x3f = mat3x3f(
  vec3f(2.8583, -0.2104, -0.0419),
  vec3f(-1.6287, 1.1584, -0.1182),
  vec3f(-0.0248, 0.0003, 1.0689),
);

@group(0) @binding(0) var<uniform> params: ExposureParams;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> state: ExposureState;
@group(0) @binding(3) var gbuffer0: texture_2d<u32>;

var<workgroup> bins: array<atomic<u32>, BINS>;

fn binOf(log_lum: f32) -> u32 {
  return u32(clamp((log_lum - LOG_MIN) / (LOG_MAX - LOG_MIN) * f32(BINS), 0.0, f32(BINS) - 1.0));
}

fn binCentre(b: u32) -> f32 {
  return LOG_MIN + (f32(b) + 0.5) / f32(BINS) * (LOG_MAX - LOG_MIN);
}

@compute @workgroup_size(THREADS)
fn main(@builtin(local_invocation_index) i: u32) {
  if (i < BINS) {
    atomicStore(&bins[i], 0u);
  }
  workgroupBarrier();

  let size = vec2f(textureDimensions(hdr));
  let gsize = vec2f(textureDimensions(gbuffer0));
  let count = GRID.x * GRID.y;
  for (var k = i; k < count; k += THREADS) {
    let cell = (vec2f(f32(k % GRID.x), f32(k / GRID.x)) + 0.5) / vec2f(GRID);
    let c = textureLoad(hdr, vec2i(cell * size), 0).rgb;
    let lum = max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-9);
    let d = (cell - 0.5) * 2.0;
    var w = exp(-dot(d, d) / (2.0 * params.center_sigma * params.center_sigma));
    if (gbIsSky(textureLoad(gbuffer0, vec2i(cell * gsize), 0).w)) {
      w *= params.sky_weight;
    }
    atomicAdd(&bins[binOf(log2(lum))], u32(w * WEIGHT_SCALE + 0.5));
  }
  workgroupBarrier();
  if (i != 0u) {
    return;
  }

  // Trimmed weighted mean of log luminance.
  var total = 0.0;
  for (var b = 0u; b < BINS; b++) {
    total += f32(atomicLoad(&bins[b]));
  }
  let lo = total * params.trim;
  let hi = total * (1.0 - params.trim);
  var acc = 0.0;
  var sum = 0.0;
  var weight = 0.0;
  for (var b = 0u; b < BINS; b++) {
    let n = f32(atomicLoad(&bins[b]));
    // Portion of this bin inside [lo, hi] of the cumulative weight.
    let inside = max(0.0, min(acc + n, hi) - max(acc, lo));
    sum += inside * binCentre(b);
    weight += inside;
    acc += n;
  }
  let previous = state.exposure;
  if (weight > 0.0) {
    let scene = exp2(sum / weight) / previous;
    let goal = clamp(params.key * exp2(params.compensation_ev) / scene, params.min_exposure, params.max_exposure);
    let tau = select(params.tau_brighter, params.tau_darker, goal > previous);
    let blend = 1.0 - exp(-params.dt / max(tau, 1e-3));
    state.exposure = exp2(mix(log2(previous), log2(goal), blend));
  }

  // White balance.
  var goal_gains = vec3f(1.0);
  if (params.wb_mode != 0u) {
    var illuminant = params.manual_illuminant;
    if (params.wb_mode == 1u) {
      illuminant = lighting.light_illuminance * max(sky.light_dir.y, 0.0) + lighting.sky_irradiance;
    }
    let lms = RGB_TO_LMS * max(illuminant / max(dot(illuminant, vec3f(0.2126, 0.7152, 0.0722)), 1e-9), vec3f(1e-6));
    let white = RGB_TO_LMS * vec3f(1.0);
    goal_gains = pow(white / lms, vec3f(params.wb_strength));
  }
  let wb_blend = select(1.0, 1.0 - exp(-params.dt / max(params.wb_tau, 1e-3)), params.wb_mode == 1u && state.wb_gains.w > 0.0);
  let gains = exp2(mix(log2(max(state.wb_gains.xyz, vec3f(1e-6))), log2(goal_gains), select(1.0, wb_blend, state.wb_gains.w > 0.0)));
  state.wb_gains = vec4f(gains, 1.0);
  let m = LMS_TO_RGB * mat3x3f(vec3f(gains.x, 0.0, 0.0), vec3f(0.0, gains.y, 0.0), vec3f(0.0, 0.0, gains.z)) * RGB_TO_LMS;
  state.wb0 = vec4f(m[0], 0.0);
  state.wb1 = vec4f(m[1], 0.0);
  state.wb2 = vec4f(m[2], 0.0);
}
