// Ray-traced reflections on water, at reduced resolution (one ray per `scale`² block of
// full-res pixels; scale 2 by default). The ray starts on the water surface of the first
// pixel in the block seen through water (water.wgsl) and follows the view ray mirrored
// about the surface normal.
//
//   hit on screen   last frame's lit image there (shadows, GI, everything), as GI does
//   hit off screen  the material's mean albedo (coarsest mip) lit by the dominant light
//                   (one shadow ray to the disk centre) and the sky light from above
//                   (unoccluded) plus its emission
//   miss / range    the sky without the sun disk (moon and stars included)
//
// The pixel of the block that is traced rotates every frame and the result is
// accumulated over frames (`temporal` below: exponential average, history clipped to the
// 3×3 neighbourhood of this frame's samples): alpha-tested leaves give each texel a
// binary hit / miss, which showed as grain along reflected leaf edges.
//
// The sun's own highlight is not traced (a mirror image of the disk aliases at this
// resolution): the lighting pass adds it analytically with the sun visibility at the
// surface (a), from one shadow ray to the disk centre — only within the highlight's lobe
// (elsewhere the lighting pass does not use it). The lighting pass upsamples the
// accumulated result (bilinear over texels on water).
// Output (rgba16float): rgb = reflected radiance (pre-exposed), a = sun visibility at the
// surface; a = −1 where the block has no water.
#include "trace.wgsl"
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "material.wgsl"
#include "sky.wgsl"
#include "water.wgsl"

struct ReflectionParams {
  max_distance: f32,
  max_steps: u32,
  shadow_distance: f32,
  shadow_max_steps: u32,
  /// Albedo mip level at off-screen hits (coarse: a few texels per face).
  hit_lod: f32,
  /// 1: hits visible on screen take last frame's lit image.
  screen_reuse: u32,
  scale: u32,
  /// Weight of this frame's sample in the accumulated result (1 = no history).
  blend: f32,
};

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;
const SURFACE_OFFSET: f32 = 1e-3;
/// Screen reuse: relative depth difference still counted as the same surface.
const SCREEN_DEPTH_TOLERANCE: f32 = 0.02;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: ReflectionParams;
@group(0) @binding(2) var gbuffer0: texture_2d<u32>;
@group(0) @binding(3) var gwater: texture_2d<f32>;
@group(0) @binding(4) var gdepth: texture_2d<f32>;
@group(0) @binding(5) var gmotion: texture_2d<f32>;
/// Last frame's lit HDR image (pre-exposed).
@group(0) @binding(6) var prev_lit: texture_2d<f32>;
/// This frame's samples (main writes, temporal reads).
@group(0) @binding(7) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var<storage, read> block_radiance: array<vec4f>;
@group(0) @binding(10) var<uniform> water: WaterParams;
/// The primary pass's fix-up flags (non-zero where the ray stopped at water or a shaped
/// block): 4 bytes per pixel, so blocks without water are skipped cheaply.
@group(0) @binding(11) var primary_flags: texture_2d<u32>;

/// Lowest upward component of a reflection ray (see main).
const MIN_REFLECTION_Y: f32 = 0.01;
/// GGX lobe of the sun's highlight: beyond this fraction of its peak it is not drawn (the
/// sun is so bright that the tail stays visible far out: 1e-4 clipped it to a small oval).
const GLINT_CUTOFF: f32 = 1e-8;

/// Distance from the camera to where the current ray started, for the texture footprint
/// of alpha tests (the reflected ray's cone continues the camera's).
var<private> cone_start: f32;

/// Water never stops these rays (they leave the surface upwards; light passes through it);
/// leaves are alpha tested at the footprint of a reduced-resolution pixel seen via the
/// water (an alpha test at the ray's own distance picked mips far too fine: the reflected
/// leaf silhouettes broke up into stippled grain).
fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
  if (id == WATER_ID) {
    return false;
  }
  return alphaOpaque(id, cell, normal, local, dir, (cone_start + t) * f32(params.scale));
}

fn traceFrom(p: vec3f, dir: vec3f, distance: f32, max_steps: u32) -> TraceResult {
  let cell = vec3i(floor(p));
  return traceRay(cam.origin_cell + cell, p - vec3f(cell), dir, 0.0, distance, max_steps);
}

/// 1 if the dominant light's disk centre is visible from `p`, else 0.
fn sunVisible(p: vec3f) -> f32 {
  return select(1.0, 0.0, traceFrom(p, sky.light_dir, params.shadow_distance, params.shadow_max_steps).hit);
}

/// Last frame's lit radiance of a camera-relative point on face `face` of a block `id` if it
/// is visible on screen now (a = 1), else a = 0. Pre-exposed. The pixel must show the same
/// face of the same block type and not be seen through water: depth alone matched the lit
/// front of a tree trunk to the dark back the ray hit (a bright streak in the reflection).
fn screenRadiance(rel: vec3f, face: u32, id: u32) -> vec4f {
  let clip = cam.view_proj * vec4f(rel, 1.0);
  if (clip.w <= 0.0) {
    return vec4f(0.0);
  }
  let uv = vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) {
    return vec4f(0.0);
  }
  let x = vec2i(uv * vec2f(cam.size));
  let depth = dot(rel, cam.forward);
  if (abs(textureLoad(gdepth, x, 0).x - depth) > SCREEN_DEPTH_TOLERANCE * depth) {
    return vec4f(0.0);
  }
  let word = textureLoad(gbuffer0, x, 0).w;
  if (gbIsSky(word) || gbThroughWater(word) || gbFace(word) != face || gbBlockId(word) != id) {
    return vec4f(0.0);
  }
  let prev_uv = uv - textureLoad(gmotion, x, 0).xy;
  if (any(prev_uv < vec2f(0.0)) || any(prev_uv >= vec2f(1.0))) {
    return vec4f(0.0);
  }
  return vec4f(textureLoad(prev_lit, vec2i(prev_uv * vec2f(cam.size)), 0).rgb, 1.0);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(output);
  if (any(gid.xy >= size)) {
    return;
  }
  let p = vec2i(gid.xy);
  // A pixel of the block seen through water, starting at one that rotates every frame.
  var q = vec2i(-1);
  let count = params.scale * params.scale;
  for (var i = 0u; i < count; i++) {
    let k = (i + cam.frame_index * 3u) % count;
    let c = p * i32(params.scale) + vec2i(i32(k % params.scale), i32(k / params.scale));
    if (all(c < vec2i(cam.size)) && textureLoad(primary_flags, c, 0).x != 0u && gbThroughWater(textureLoad(gbuffer0, c, 0).w)) {
      q = c;
      break;
    }
  }
  // Camera under water: no surface seen from above (its view of the surface from below is
  // traced by the primary UNDERWATER pipeline).
  if (q.x < 0 || water.camera_under != 0u) {
    textureStore(output, p, vec4f(0.0, 0.0, 0.0, -1.0));
    return;
  }
  let w = textureLoad(gwater, q, 0);
  let uv = (vec2f(q) + 0.5) / vec2f(cam.size);
  let view = rayDir(cam.inv_view_proj, uv);
  let n_s = waterSurfaceNormal(w);
  let surface = cam.origin_frac + view * w.x + n_s * SURFACE_OFFSET;
  // Mirrored about a wavy normal, a grazing ray can point down into the water: it would
  // meet the next wave; it is kept just above the horizon instead (walking under water to
  // the ray's end cost up to ~1 ms in peaks, with the wave phase).
  let mirrored = reflect(view, n_s);
  let dir = normalize(vec3f(mirrored.x, max(mirrored.y, MIN_REFLECTION_Y), mirrored.z));
  cone_start = w.x;
  // GGX D(n·h) / D(1) = (α² / ((n·h)²(α² − 1) + 1))².
  let h = normalize(sky.light_dir - view);
  let a2 = pow(water.sun_roughness, 4.0);
  let n_h = max(dot(n_s, h), 0.0);
  let lobe = a2 / (n_h * n_h * (a2 - 1.0) + 1.0);
  var sun_vis = 0.0;
  if (dot(n_s, sky.light_dir) > 0.0 && lobe * lobe > GLINT_CUTOFF) {
    sun_vis = sunVisible(surface);
  }

  let r = traceFrom(surface, dir, params.max_distance, params.max_steps);
  var radiance: vec3f;
  if (!r.hit) {
    radiance = (skyScattering(dir) + moonDisk(dir) + stars(dir)) * preExposure();
  } else {
    let hit = surface + dir * r.t;
    let normal = select(r.normal, vec3i(0, 1, 0), all(r.normal == vec3i(0)));
    let face = faceIndex(normal);
    var lit = vec4f(0.0);
    if (params.screen_reuse != 0u) {
      lit = screenRadiance(hit - cam.origin_frac, face, r.id);
    }
    if (lit.a > 0.0) {
      radiance = lit.rgb;
    } else {
      let info = materials[faceMaterial(r.id, face)];
      let layer = info.first_layer + hashCellFace(r.cell, face) % max(info.variants, 1u);
      let albedo = srgbDecode(textureSampleLevel(tex_albedo, tex_sampler, vec2f(0.5), layer, params.hit_lod).rgb);
      let n = vec3f(normal);
      let cos_l = dot(n, sky.light_dir);
      var e = lighting.sky_irradiance * (0.5 + 0.5 * n.y);
      if (cos_l > 0.0) {
        e += lighting.light_illuminance * cos_l * sunVisible(hit + n * SURFACE_OFFSET);
      }
      radiance = (albedo / PI * e + block_radiance[r.id].rgb) * preExposure();
    }
  }
  textureStore(output, p, vec4f(radiance, sun_vis));
}

// ---------------------------------------------------------------------------------------
// Temporal accumulation: this frame's sample blended into last frame's result at the
// reprojected texel (motion of the block's first pixel), the history clipped to the min /
// max of this frame's valid samples around (no trails behind a moving camera).

@group(0) @binding(12) var current: texture_2d<f32>;
@group(0) @binding(13) var history: texture_2d<f32>;
@group(0) @binding(14) var accumulated: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn temporal(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(accumulated));
  let p = vec2i(gid.xy);
  if (any(p >= size)) {
    return;
  }
  let c = textureLoad(current, p, 0);
  if (c.a < 0.0) {
    textureStore(accumulated, p, c);
    return;
  }
  var lo = c;
  var hi = c;
  for (var k = 0; k < 9; k++) {
    let q = clamp(p + vec2i(k % 3 - 1, k / 3 - 1), vec2i(0), size - 1);
    let v = textureLoad(current, q, 0);
    if (v.a >= 0.0) {
      lo = min(lo, v);
      hi = max(hi, v);
    }
  }
  let full = min(p * i32(params.scale), vec2i(cam.size) - 1);
  let prev_uv = (vec2f(full) + 0.5) / vec2f(cam.size) - textureLoad(gmotion, full, 0).xy;
  var result = c;
  if (all(prev_uv >= vec2f(0.0)) && all(prev_uv < vec2f(1.0))) {
    let h = textureLoad(history, clamp(vec2i(prev_uv * vec2f(size)), vec2i(0), size - 1), 0);
    if (h.a >= 0.0) {
      result = mix(clamp(h, lo, hi), c, params.blend);
    }
  }
  textureStore(accumulated, p, result);
}
