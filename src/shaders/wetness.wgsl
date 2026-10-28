// Wetness (Aşama 9e): how wet a surface is, from a global value (rain, later: up-facing
// ground open to the sky; sheltered spots stay dry) and local zones (position + radius),
// plus puddles on flat ground (noise mask). The GI prepare pass (restir.wgsl) and the
// lighting pass evaluate the same function: GI traces specular rays off puddles (smooth,
// flat) and the lighting pass shades them. Nothing when everything is dry (one uniform
// test). Needs `water: WaterParams`, getVoxel (brickmap.wgsl), giValueNoise
// (gi-common.wgsl) and `cam: Camera`.

/// Cells above a surface checked for a roof (rain does not reach under it).
const WET_SHELTER_CELLS: i32 = 4;
/// Puddles: roughness of the water film, albedo under it (darker), and how sharply its
/// edge follows the noise.
const PUDDLE_ROUGHNESS: f32 = 0.03;
const PUDDLE_DARKEN: f32 = 0.8;
const PUDDLE_EDGE: f32 = 0.08;
/// Wet surface (applyWetness): albedo and roughness multipliers when fully wet.
const WET_DARKEN: f32 = 0.6;
const WET_SMOOTH: f32 = 0.8;

fn wetnessActive() -> bool {
  return water.wet_global > 0.0 || water.zone_count > 0u;
}

/// (wetness, puddle coverage), both 0..1, of a surface in absolute cell `cell` at world
/// position `world` with geometric normal `n_geo`.
fn wetnessAt(world: vec3f, cell: vec3i, n_geo: vec3f) -> vec2f {
  var wet = 0.0;
  if (water.wet_global > 0.0) {
    var sheltered = false;
    for (var k = 1; k <= WET_SHELTER_CELLS; k++) {
      let id = getVoxel(cell + vec3i(0, k, 0));
      if (id != 0u && id != WATER_ID) {
        sheltered = true;
        break;
      }
    }
    // Faces turned up catch the rain; walls get wet from runoff and splashes, less.
    let open = mix(0.5, 1.0, clamp(n_geo.y, 0.0, 1.0));
    wet = select(water.wet_global * open, 0.0, sheltered);
  }
  for (var i = 0u; i < min(water.zone_count, 4u); i++) {
    let z = water.zones[i];
    let d = distance(world, z.xyz);
    wet = max(wet, water.zone_amount[i] * (1.0 - smoothstep(0.7 * z.w, z.w, d)));
  }
  var puddle = 0.0;
  if (wet > 0.0 && water.puddles > 0.0 && n_geo.y > 0.9) {
    // Two octaves of value noise over puddle_scale metres; the wetter, the larger.
    let q = world.xz / max(water.puddle_scale, 0.1);
    let noise = 0.65 * giValueNoise(vec3f(q, 0.0)) + 0.35 * giValueNoise(vec3f(q * 2.3, 7.0));
    let level = 1.0 - wet * water.puddles * 0.6;
    puddle = smoothstep(level, level + PUDDLE_EDGE, noise);
  }
  return vec2f(clamp(wet, 0.0, 1.0), puddle);
}
