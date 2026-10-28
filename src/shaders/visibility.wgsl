// Per-pixel light visibility, one sample per frame (accumulated by temporal.wgsl):
//   r = dominant light: a shadow ray towards a random point of the sun / moon disk
//   g = sky: a cosine-weighted ray over the hemisphere, short range (local occlusion)
// Rays start on the hit face, offset along its geometric normal.
//
// Foliage, in one traversal per ray (no restart per leaf voxel):
//   shadow rays  leaf texels are alpha tested (dappled light through the holes); an opaque
//                texel does not stop the ray but multiplies its light by
//                `leaf_transmission` (light passing through leaves) — the same result as
//                restarting the ray behind each leaf voxel, without the restarts. Once the
//                light left is negligible (after `maxLeafLayers` opaque texels) the ray stops.
//   sky rays     stop at the first leaf texel and return `leaf_transmission` (short-range
//                occlusion only, fewer steps).
// (A coarser alpha-test LOD from the primary pixel footprint was tried: no faster, and it
// closed leaf holes — coverage-preserving mips — so the fine LOD is kept.)
//
// Shaped blocks (trace.wgsl MICRO_WALK): the main pipeline flags pixels where either ray
// stopped at one (cubes there); the fix-up pipeline recomputes just those pixels, with the
// same random numbers, on the micro grid.
//
// Water lets the rays through (it is a medium, not a surface; its absorption is applied
// when shading). Pixels seen through water (water.wgsl) start at the true bottom position;
// their shadow ray leaves the water along the refracted light direction, then goes on
// towards the light from the surface, and its light cone widens with depth and scattering.
#include "trace.wgsl"
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "material.wgsl"
#include "sky.wgsl"
#include "water.wgsl"

struct VisibilityParams {
  shadow_distance: f32,
  sky_distance: f32,
  max_steps: u32,
  /// 1 = every pixel every frame; 2 = one pixel per 2×2 block per frame (rotating).
  block: u32,
  sky_max_steps: u32,
  /// Fraction of light a leaf voxel lets through (on top of its alpha holes).
  leaf_transmission: f32,
  /// 0 while GI is on: GI brings the sky light (with real occlusion), no sky ray needed.
  sky_enabled: u32,
  /// Shadow rays stop once the light left behind leaves drops below this
  /// (leaf_transmission ^ maxLeafLayers, config).
  min_leaf_light: f32,
};

/// Per-invocation state shared with traceOpaque (it has no extra parameters).
/// Light left after the leaf texels crossed so far (shadow rays).
var<private> leaf_light: f32;
/// 1 while tracing a sky ray: the first leaf texel ends it (see transmittance).
var<private> sky_ray: u32;
/// Set by traceOpaque when a sky ray was stopped by a leaf.
var<private> stopped_by_leaf: bool;
/// Set when a ray stopped at a shaped block (main pipeline: the pixel needs the fix-up).
var<private> hit_shaped: bool;

/// Pixel of each 2×2 block traced in a frame (rotating so all four get samples).
fn tracedOffset(frame: u32) -> vec2u {
  return array<vec2u, 4>(vec2u(0u, 0u), vec2u(1u, 1u), vec2u(1u, 0u), vec2u(0u, 1u))[frame & 3u];
}

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;
/// Offset of ray origins from the surface (blocks): leaves the hit voxel cleanly.
const SURFACE_OFFSET: f32 = 1e-3;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var gbuffer0: texture_2d<u32>;
@group(0) @binding(2) var gdepth: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: VisibilityParams;
/// Fix-up flags (see primary.wgsl): main pipeline writes, fix-up reads.
@group(0) @binding(5) var micro_flag_out: texture_storage_2d<r32uint, write>;
@group(0) @binding(6) var micro_flag_in: texture_2d<u32>;
@group(0) @binding(7) var gwater: texture_2d<f32>;
@group(0) @binding(8) var<uniform> water: WaterParams;

fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
  if (id == WATER_ID && water.enabled != 0u) {
    return false;
  }
  let inside = all(normal == vec3i(0));
  let face = select(faceIndex(normal), 2u, inside);
  let material = faceMaterial(id, face);
  if ((materials[material].flags & MATERIAL_ALPHA_TEST) == 0u) {
    return true;
  }
  if (inside) {
    // The ray starts inside a leaf voxel: an inner leaf seen through the holes of the one
    // in front. No face to alpha test; count the voxel as one leaf layer (treating it as
    // solid left those inner leaves black).
    if (sky_ray != 0u) {
      stopped_by_leaf = true;
      return true;
    }
    leaf_light *= params.leaf_transmission;
    return leaf_light < params.min_leaf_light;
  }
  let m = faceMapping(material, cell, face, local);
  let lod = coneLod(dir, m.n, t * material_params.pixel_spread);
  if (textureSampleLevel(tex_albedo, tex_sampler, m.uv, m.layer, lod).a < material_params.alpha_cutoff) {
    return false; // a hole: light passes
  }
  if (sky_ray != 0u) {
    stopped_by_leaf = true;
    return true;
  }
  // An opaque leaf texel attenuates the light and the ray goes on, unless little is left.
  leaf_light *= params.leaf_transmission;
  return leaf_light < params.min_leaf_light;
}

fn hash(v: vec3u) -> u32 {
  var h = (v.x * 0x8da6b343u) ^ (v.y * 0xd8163841u) ^ (v.z * 0xcb1ab31fu);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

fn random2(px: vec2u, frame: u32, stream: u32) -> vec2f {
  let h = hash(vec3u(px, frame * 2u + stream));
  return vec2f(f32(h & 0xffffu), f32(h >> 16u)) / 65536.0;
}

/// Orthonormal basis around `n` (Frisvad / Duff et al.).
fn basis(n: vec3f) -> mat3x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat3x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}

/// Light reaching `start` along `dir` within `distance`, in one traversal: shadow rays
/// return the product of the leaf transmissions crossed (0 behind solid blocks); sky rays
/// stopped by a leaf return `leaf_transmission`.
fn transmittance(start: vec3f, dir: vec3f, distance: f32, max_steps: u32) -> f32 {
  stopped_by_leaf = false;
  leaf_light = 1.0;
  let cell = vec3i(floor(start));
  let r = traceRay(cam.origin_cell + cell, start - vec3f(cell), dir, 0.0, distance, max_steps);
  if (!r.hit) {
    return leaf_light;
  }
  hit_shaped = hit_shaped || isShapedId(r.id);
  return select(0.0, params.leaf_transmission, stopped_by_leaf);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  // With block = 2 the dispatch covers half the resolution: one pixel per 2×2 block.
  let pixel = id.xy * params.block + select(vec2u(0u), tracedOffset(cam.frame_index), params.block == 2u);
  if (any(pixel >= cam.size)) {
    return;
  }
  let gid = vec3u(pixel, 0u);
  let px = vec2i(gid.xy);
  if (MICRO_WALK && textureLoad(micro_flag_in, px, 0).x == 0u) {
    return;
  }
  hit_shaped = false;
  let g = textureLoad(gbuffer0, px, 0);
  if (gbIsSky(g.w)) {
    textureStore(output, px, vec4f(1.0, 1.0, 0.0, 0.0));
    if (!MICRO_WALK) {
      textureStore(micro_flag_out, px, vec4u(0u));
    }
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  let view = rayDir(cam.inv_view_proj, uv);
  let depth = textureLoad(gdepth, px, 0).x;
  let n = faceNormal(gbFace(g.w));
  var under = gbThroughWater(g.w);
  var pos = viewRelativePosition(view, cam.forward, depth);
  var n_s = vec3f(0.0, 1.0, 0.0);
  /// Height of the water surface (camera-cell frame).
  var surface_y = 0.0;
  var w = vec4f(0.0);
  if (under) {
    w = textureLoad(gwater, px, 0);
    pos = waterBottom(view, w, water.ior);
    // Camera under water, seen out through the surface: the hit is in the air.
    under = w.w != WATER_EXIT;
    n_s = waterSurfaceNormal(w);
    surface_y = cam.origin_frac.y + view.y * w.x;
  }
  let start = cam.origin_frac + pos + n * SURFACE_OFFSET;

  // Shadow ray: uniform point on the light's disk (a cone of half-angle light_radius).
  var sun = 0.0;
  let r = random2(gid.xy, cam.frame_index, 0u);
  let cos_max = cos(sky.light_radius);
  let cos_t = mix(1.0, cos_max, r.x);
  let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
  let phi = 2.0 * PI * r.y;
  var l = basis(sky.light_dir) * vec3f(sin_t * cos(phi), sin_t * sin(phi), cos_t);
  var dir = l;
  var dist = params.shadow_distance;
  /// Whether the face looks towards the light (the unjittered direction it arrives from).
  var facing = dot(n, l) > 0.0;
  if (under) {
    // Under water: up along the refracted light direction to the surface, then (second
    // segment) on towards the light in air. Scattering in the water spreads where the light
    // came through the surface over a disk growing with depth (soft shadows on the bottom
    // of turbid water — the blur depends on the water depth only, not on how far the
    // occluder is).
    let l_w = waterLightDir(l, n_s, water.ior);
    let up = max(surface_y - start.y, 0.0);
    let exit = start + l_w * (up / max(dot(l_w, n_s), 1e-3));
    let q = random2(gid.xy, cam.frame_index, 2u);
    let spread = water.shadow_blur * waterMeanScattering(water) * up * up;
    let b = basis(n_s);
    let jitter = (b[0] * cos(2.0 * PI * q.y) + b[1] * sin(2.0 * PI * q.y)) * sqrt(q.x) * spread;
    let to_exit = exit + jitter - start;
    dist = length(to_exit);
    dir = select(l_w, to_exit / max(dist, 1e-6), dist > 1e-6);
    // The jitter only spreads occlusion: whether the face is lit at all is decided by the
    // refracted light direction (testing the jittered one let faces turned away from the
    // light catch ~half the samples: grey noise on the sides of underwater steps). A
    // jittered ray dipping below the face takes the light direction instead.
    facing = dot(l_w, n_s) > 0.0 && dot(n, l_w) > 0.0;
    if (dot(n, dir) <= 0.0) {
      dir = l_w;
      dist = up / max(dot(l_w, n_s), 1e-3);
    }
  }
  if (facing) {
    sky_ray = 0u;
    var origin = start;
    // A dynamic segment count, so the loop (and the traversal in it) is not unrolled.
    let segments = select(1u, 2u, under);
    sun = 1.0;
    for (var k = 0u; k < segments; k++) {
      sun *= transmittance(origin, dir, dist, params.max_steps);
      if (sun <= 0.0 || k + 1u == segments) {
        break;
      }
      origin += dir * dist;
      dir = l;
      dist = params.shadow_distance;
    }
  }

  // Sky ray: cosine-weighted around the face normal.
  var skyv = 1.0;
  if (params.sky_enabled != 0u) {
    let q = random2(gid.xy, cam.frame_index, 1u);
    let sin_t = sqrt(q.x);
    let phi = 2.0 * PI * q.y;
    let d = basis(n) * vec3f(sin_t * cos(phi), sin_t * sin(phi), sqrt(1.0 - q.x));
    sky_ray = 1u;
    skyv = transmittance(start, d, params.sky_distance, params.sky_max_steps);
  }

  textureStore(output, px, vec4f(sun, skyv, 0.0, 0.0));
  if (!MICRO_WALK) {
    textureStore(micro_flag_out, px, vec4u(select(0u, 1u, hit_shaped), 0u, 0u, 0u));
  }
}
