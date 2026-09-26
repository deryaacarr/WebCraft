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
#include "trace.wgsl"
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "material.wgsl"
#include "sky.wgsl"

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

fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
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
  let g = textureLoad(gbuffer0, px, 0);
  if (gbIsSky(g.w)) {
    textureStore(output, px, vec4f(1.0, 1.0, 0.0, 0.0));
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  let view = rayDir(cam.inv_view_proj, uv);
  let depth = textureLoad(gdepth, px, 0).x;
  let n = faceNormal(gbFace(g.w));
  let start = cam.origin_frac + viewRelativePosition(view, cam.forward, depth) + n * SURFACE_OFFSET;

  // Shadow ray: uniform point on the light's disk (a cone of half-angle light_radius).
  var sun = 0.0;
  if (dot(n, sky.light_dir) > 0.0) {
    let r = random2(gid.xy, cam.frame_index, 0u);
    let cos_max = cos(sky.light_radius);
    let cos_t = mix(1.0, cos_max, r.x);
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let phi = 2.0 * PI * r.y;
    let l = basis(sky.light_dir) * vec3f(sin_t * cos(phi), sin_t * sin(phi), cos_t);
    sky_ray = 0u;
    sun = transmittance(start, l, params.shadow_distance, params.max_steps);
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
}
