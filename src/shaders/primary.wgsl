// Primary rays: one per pixel from the camera into the brickmap; materials are sampled at
// the hit and the result goes to the G-buffer. Shaped blocks (trace.wgsl MICRO_WALK) and
// water: the main pipeline flags pixels whose ray stopped at one and lists their workgroup
// tiles; the fix-up pipeline (same shader, MICRO_WALK on) runs as an indirect dispatch over
// just those tiles and redoes the flagged pixels — shaped blocks on the micro grid, water
// by refracting at the surface and tracing on to the bottom (water.wgsl). (A full-screen
// fix-up dispatch that returned early cost ~0.35 ms on frames without any flagged pixel.)
#include "trace.wgsl"
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "material.wgsl"
#include "water.wgsl"

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var gbuffer0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var gdepth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var gmotion: texture_storage_2d<rg32float, write>;
/// Depth prepass: per tile, the distance before which no ray of the tile hits anything.
@group(0) @binding(4) var coarse: texture_2d<f32>;
/// Pixels whose ray stopped at a shaped block: the distance to it (f32 bits; 0 = none),
/// written by the main pipeline, read by the fix-up — which starts its ray just before
/// that block instead of at the camera. Each pipeline binds a 1×1 dummy for the other.
@group(0) @binding(5) var micro_flag_out: texture_storage_2d<r32uint, write>;
@group(0) @binding(6) var micro_flag_in: texture_2d<u32>;
/// Tile list (primary-pass.ts): words 0–2 = indirect dispatch args (x = tiles listed), then
/// one mark per tile, then the listed tile indices. Main pipeline: read_write; fix-up:
/// read-only (the buffer is also its indirect args). Each binds a dummy for the other.
@group(0) @binding(7) var<storage, read_write> micro_tiles: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> micro_tile_list: array<u32>;
/// Water samples (water.wgsl), written by the fix-up only (cleared every frame).
@group(0) @binding(9) var gwater: texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var<uniform> water: WaterParams;

/// Fix-up: set while tracing the refracted ray under a water surface (water lets it through).
var<private> in_water: bool;

const TILE_HEADER: u32 = 4u;

fn tilesX() -> u32 {
  return (cam.size.x + WORKGROUP_X - 1u) / WORKGROUP_X;
}

fn tileCount() -> u32 {
  return tilesX() * ((cam.size.y + WORKGROUP_Y - 1u) / WORKGROUP_Y);
}

/// Alpha-tested materials (leaves) let the ray through transparent texels.
fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
  if (MICRO_WALK && in_water && id == WATER_ID) {
    return false;
  }
  return alphaOpaque(id, cell, normal, local, dir, t);
}

fn clipToUv(clip: vec4f) -> vec2f {
  let ndc = clip.xy / clip.w;
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(workgroup_id) wid: vec3u,
        @builtin(local_invocation_id) lid: vec3u) {
  var gid = global_id.xy;
  if (MICRO_WALK) {
    // Workgroup i handles the i-th listed tile.
    let tile = micro_tile_list[TILE_HEADER + tileCount() + wid.x];
    gid = vec2u(tile % tilesX(), tile / tilesX()) * vec2u(WORKGROUP_X, WORKGROUP_Y) + lid.xy;
  }
  if (any(gid >= cam.size)) {
    return;
  }
  let px = vec2i(gid);
  var flag = 0u;
  if (MICRO_WALK) {
    flag = textureLoad(micro_flag_in, px, 0).x;
    if (flag == 0u) {
      return;
    }
  }
  let uv = (vec2f(gid) + 0.5) / vec2f(cam.size);
  let dir = rayDir(cam.inv_view_proj, uv);
  var t_min = 0.0;
  if (cam.prepass_tile > 0u) {
    t_min = textureLoad(coarse, vec2i(gid / cam.prepass_tile), 0).x;
  }
  if (MICRO_WALK) {
    // Nothing is hit before the shaped block or water the main pipeline stopped at.
    t_min = max(t_min, bitcast<f32>(flag) * (1.0 - 1e-5) - 1e-4);
  }

  var r = traceRay(cam.origin_cell, cam.origin_frac, dir, t_min, cam.far, cam.max_steps);
  let water_on = water.enabled != 0u;
  if (!MICRO_WALK && r.hit && (isShapedId(r.id) || (water_on && r.id == WATER_ID))) {
    // The flags are cleared every frame (primary-pass.ts): only flagged pixels write. An
    // unconditional store per pixel cost the main pipeline ~0.4 ms.
    textureStore(micro_flag_out, px, vec4u(bitcast<u32>(max(r.t, 1e-30)), 0u, 0u, 0u));
    let tile = (gid.y / WORKGROUP_Y) * tilesX() + gid.x / WORKGROUP_X;
    if (atomicExchange(&micro_tiles[TILE_HEADER + tile], 1u) == 0u) {
      let i = atomicAdd(&micro_tiles[0], 1u);
      atomicStore(&micro_tiles[TILE_HEADER + tileCount() + i], tile);
    }
    if (r.id == WATER_ID) {
      // The fix-up writes this pixel (the bottom under the water): no shading here.
      return;
    }
  }

  // Hit position relative to the camera, direction and length of the ray that reached it.
  var p = dir * r.t;
  var hit_dir = dir;
  var hit_dist = r.t;
  var water_sample = vec4f(0.0);
  if (MICRO_WALK && water_on && r.hit && r.id == WATER_ID) {
    // Into the water: refract at the surface (a ray starting inside goes straight on) and
    // trace through it to the bottom.
    let inside = all(r.normal == vec3i(0));
    let n_s = select(vec3f(r.normal), -dir, inside);
    let d = select(refract(dir, n_s, 1.0 / water.ior), dir, inside);
    let s_rel = cam.origin_frac + dir * r.t;
    let cell = vec3i(floor(s_rel));
    in_water = true;
    trace_skip_id = WATER_ID;
    let b = traceRay(cam.origin_cell + cell, s_rel - vec3f(cell), d, 0.0, water.max_distance, cam.max_steps);
    in_water = false;
    trace_skip_id = 0u;
    water_sample = packWaterSample(max(r.t, 1e-6), select(water.max_distance, b.t, b.hit), n_s);
    textureStore(gwater, px, water_sample);
    p = dir * r.t + d * b.t;
    hit_dir = d;
    // Forward scattering blurs the bottom with the path: a wider texture footprint.
    let blur = water.detail_blur * waterMeanScattering(water) * b.t * b.t;
    hit_dist = r.t + b.t + blur / max(material_params.pixel_spread, 1e-6);
    r.hit = b.hit;
    r.cell = b.cell;
    r.normal = b.normal;
    r.id = b.id;
    r.steps += b.steps;
  }

  let water_flag = select(0u, GB_WATER_FLAG, throughWater(water_sample));
  if (!r.hit) {
    // Sky (or deep water: lighting.wgsl composites it from the water sample alone).
    // Reproject the direction only (infinitely far, camera translation irrelevant).
    let motion = clipToUv(cam.view_proj * vec4f(dir, 0.0)) - clipToUv(cam.prev_view_proj * vec4f(dir, 0.0));
    textureStore(gbuffer0, px, vec4u(0u, 0u, 0u, packMaterialWord(0u, 0u, 0u, r.steps, true) | water_flag));
    textureStore(gdepth, px, vec4f(cam.far, 0.0, 0.0, 0.0));
    textureStore(gmotion, px, vec4f(motion, 0.0, 0.0));
    return;
  }

  var normal = r.normal;
  if (all(normal == vec3i(0))) {
    // Started inside a block: face the camera along the dominant axis.
    let a = abs(hit_dir);
    let axis = select(select(2u, 1u, a.y > a.z), 0u, a.x > a.y && a.x > a.z);
    normal[axis] = select(1, -1, hit_dir[axis] > 0.0);
  }
  let face = faceIndex(normal);
  let material = faceMaterial(r.id, face);

  // Position inside the hit voxel in [0, 1]³ (camera-relative, so precise).
  let local = clamp(cam.origin_frac + p - vec3f(r.cell - cam.origin_cell), vec3f(0.0), vec3f(1.0));
  let s = shadeSurface(material, r.cell, face, local, hit_dir, hit_dist);

  if (throughWater(water_sample)) {
    // Seen through water: depth and motion of the point where the camera ray meets the hit
    // face's plane (see water.wgsl); along the ray at the bottom's depth if it misses it.
    let n_f = faceNormal(face);
    let denom = dot(n_f, dir);
    let along = dir * (dot(p, cam.forward) / dot(dir, cam.forward));
    p = select(along, dir * (dot(n_f, p) / denom), denom < -1e-3 && dot(n_f, p) / denom > 0.0);
  }
  let motion = clipToUv(cam.view_proj * vec4f(p, 1.0)) - clipToUv(cam.prev_view_proj * vec4f(p + cam.prev_delta, 1.0));

  textureStore(gbuffer0, px, vec4u(
    pack4x8unorm(vec4f(s.albedo, s.ao)),
    pack2x16snorm(octEncode(s.normal)),
    pack4x8unorm(vec4f(s.roughness, s.metalness, s.emission, s.subsurface)),
    packMaterialWord(r.id, material, face, r.steps, false) | water_flag,
  ));
  textureStore(gdepth, px, vec4f(dot(p, cam.forward), 0.0, 0.0, 0.0));
  textureStore(gmotion, px, vec4f(motion, 0.0, 0.0));
}
