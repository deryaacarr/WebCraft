// Shore details of the water (Aşama 9d), for the lighting pass: sediment in shallow water,
// foam where the surface meets the shore, caustics on the bottom, wet ground beside the
// water. Needs `water: WaterParams`, `cam: Camera`, getVoxel (brickmap.wgsl) and fastNoise
// (material.wgsl). Nothing here adds light: foam is a white surface lit by the sun and sky,
// caustics redistribute the sun on the bottom (mean 1), wetness only darkens and smooths.

/// Foam albedo (linear): fresh foam is nearly white.
const FOAM_ALBEDO: f32 = 0.75;
/// Caustic pattern: sharpness of the bright web and its mean (normalised to 1).
const CAUSTIC_POWER: f32 = 6.0;
/// Mean of the web for a triangular distribution of a + b − 1: 2 / (power + 2).
const CAUSTIC_MEAN: f32 = 0.25;
/// Caustics: pattern size (metres) and the depth (metres) over which they blur out.
const CAUSTIC_SIZE: f32 = 0.35;
const CAUSTIC_FADE_DEPTH: f32 = 4.0;

struct WaterMedium {
  /// σ_t per metre (absorption + scattering), linear rgb.
  extinction: vec3f,
  /// σ_s · single-scattering albedo per metre: light scattered towards the viewer.
  scatter: vec3f,
};

/// The water where the column above the bottom is `depth` metres deep: clear water plus
/// sediment stirred up in the shallows (its scattering falls off over turbidity_depth),
/// which scatters in its own brownish colour and absorbs the rest.
fn waterMedium(depth: f32) -> WaterMedium {
  let tau = water.turbidity * exp(-max(depth, 0.0) / max(water.turbidity_depth, 1e-3));
  return WaterMedium(
    water.absorption + water.scattering + vec3f(tau),
    water.scattering * water.scatter_albedo + tau * water.sediment,
  );
}

fn isShoreSolid(id: u32) -> bool {
  return id != 0u && id != WATER_ID;
}

/// Foam coverage (0..1) at a surface point `s` (camera-cell frame): a thin strip where the
/// surface meets a solid block of the shore, its width varying along the shore and its
/// edge broken up by noise drifting with the wind and slowly changing.
fn waterFoam(s: vec3f, time: f32) -> f32 {
  if (water.foam <= 0.0) {
    return 0.0;
  }
  let c = vec3i(floor(s));
  let f = s - vec3f(c);
  let abs_cell = cam.origin_cell + c;
  // Distance to the nearest shore face on the four sides of the surface cell.
  var d = 2.0;
  if (isShoreSolid(getVoxel(abs_cell + vec3i(-1, 0, 0)))) { d = min(d, f.x); }
  if (isShoreSolid(getVoxel(abs_cell + vec3i(1, 0, 0)))) { d = min(d, 1.0 - f.x); }
  if (isShoreSolid(getVoxel(abs_cell + vec3i(0, 0, -1)))) { d = min(d, f.z); }
  if (isShoreSolid(getVoxel(abs_cell + vec3i(0, 0, 1)))) { d = min(d, 1.0 - f.z); }
  if (d >= 1.0) {
    return 0.0;
  }
  let world = vec2f(abs_cell.xz) + f.xz;
  let drift = water.wind_dir * time * 0.15;
  // Width wanders along the shore (~1 m); the strip breaks into patches (~0.2 m).
  let width = water.foam_width * (0.5 + fastNoise(vec3f(world * 1.0, time * 0.1)));
  let strip = 1.0 - smoothstep(0.0, width, d);
  let patches = smoothstep(0.42, 0.72, fastNoise(vec3f((world + drift) * 4.0, time * 0.5 + 9.0)));
  return water.foam * strip * patches;
}

/// Caustics on the bottom at world position `xz`, `depth` metres below the surface: a
/// multiplier of the direct sun with mean 1 — a bright web of two noise layers drifting
/// against each other, sharp in the shallows, blurring out with depth, gone on still water.
fn waterCaustics(xz: vec2f, depth: f32, time: f32) -> f32 {
  let calm = clamp(water.wave_amplitude / 0.001, 0.0, 1.0);
  let amount = water.caustics * calm * exp(-max(depth, 0.0) / CAUSTIC_FADE_DEPTH);
  if (amount <= 0.0) {
    return 1.0;
  }
  let p = xz / CAUSTIC_SIZE * 0.25; // noise units: one texel (0.25) per pattern cell
  let a = fastNoise(vec3f(p + water.wind_dir * time * 0.12, time * 0.35));
  let b = fastNoise(vec3f(p * 1.37 - water.wind_dir * time * 0.09, time * 0.3 + 7.0));
  let web = pow(1.0 - abs(a + b - 1.0), CAUSTIC_POWER);
  return max(1.0 + amount * (web / CAUSTIC_MEAN - 1.0), 0.0);
}

/// Wetness (0..1) of ground in cell `c` (absolute) beside the water: water one block away
/// on the same level or one below (the shore block's own level, or a step above the
/// water), or two blocks away on its level (a beach at the waterline). Within
/// wet_max_distance only (8–12 voxel reads).
fn shoreWetness(c: vec3i, dist: f32) -> f32 {
  if (water.wet <= 0.0 || dist > water.wet_max_distance) {
    return 0.0;
  }
  var w = 0.0;
  let dirs = array<vec3i, 4>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
  for (var k = 0; k < 4; k++) {
    let o = dirs[k];
    if (getVoxel(c + o) == WATER_ID || getVoxel(c + o - vec3i(0, 1, 0)) == WATER_ID) {
      w = 1.0;
      break;
    }
    if (getVoxel(c + 2 * o) == WATER_ID) {
      w = max(w, 0.55);
    }
  }
  return w * water.wet * (1.0 - smoothstep(0.75 * water.wet_max_distance, water.wet_max_distance, dist));
}

/// Wet surface: darker (water fills the pores) and smoother. `wet` 0..1.
fn applyWetness(s: ptr<function, Surface>, wet: f32) {
  (*s).albedo *= mix(1.0, WET_DARKEN, wet);
  (*s).roughness *= mix(1.0, WET_SMOOTH, wet);
}
