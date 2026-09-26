// Global illumination samples at half resolution, one path per traced pixel and frame
// (every half-res pixel, or an alternating checkerboard half of them):
//
//   diffuse  a cosine-weighted bounce ray (blue noise). Within `range` blocks it may hit a
//            surface. If that point is visible on screen, its radiance is last frame's lit
//            result there (reprojected; this also brings further bounces). Otherwise it is
//            lit by the dominant light (shadow ray), the sky (one short cosine-weighted
//            ray) and, if ReSTIR DI is off, its own emission. Missing, or leaving the
//            range, the ray returns the sky-view LUT. Irradiance estimate π·L.
//   specular for smooth surfaces only (probability from roughness): a GGX VNDF ray; the
//            estimate is the incoming radiance (Fresnel is applied when shading).
//   emissive the light chosen by ReSTIR DI (restir.wgsl): one shadow ray to a random point
//            of the emitting block; its own signal (raw_emissive), accumulated and denoised
//            separately and without flicker — the lighting pass applies the flicker, which
//            temporal accumulation would otherwise average away.
//
// Outputs are pre-exposed: raw_diffuse = irradiance, raw_specular = radiance. Only the
// pixels traced this frame are written (the dispatch covers just those, see tracedPixel);
// the others keep older samples, which the temporal pass ignores (isTraced).
#include "trace.wgsl"
#include "gi-common.wgsl"
#include "material.wgsl"
#include "sky.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: GiParams;
@group(0) @binding(2) var guide: texture_2d<u32>;
@group(0) @binding(3) var reservoirs: texture_2d<u32>;
@group(0) @binding(4) var raw_diffuse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var raw_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var blue_noise: texture_2d_array<f32>;
@group(0) @binding(7) var<storage, read> block_radiance: array<vec4f>;
@group(0) @binding(8) var gdepth: texture_2d<f32>;
@group(0) @binding(9) var gmotion: texture_2d<f32>;
/// Last frame's lit HDR image (the lighting pass writes it after this pass).
@group(0) @binding(10) var prev_lit: texture_2d<f32>;
/// Exposure last frame's image was stored with.
@group(0) @binding(11) var<storage, read> history_exposure: array<f32>;
@group(0) @binding(12) var raw_emissive: texture_storage_2d<rgba16float, write>;

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;
const SURFACE_OFFSET: f32 = 1e-3;
/// Screen reuse: relative depth difference still counted as the same surface.
const SCREEN_DEPTH_TOLERANCE: f32 = 0.02;

/// GI rays treat foliage stochastically instead of alpha testing texels: a leaf voxel lets
/// a ray through with probability (1 − coverage) + coverage · leaf_transmission (coverage =
/// the texture's mean opacity, see MaterialInfo). Same expected light as the alpha-tested
/// shadow rays, without texture lookups per voxel or loops over leaf crossings.
fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
  let face = faceIndex(select(normal, vec3i(0, 1, 0), all(normal == vec3i(0))));
  let info = materials[faceMaterial(id, face)];
  if ((info.flags & MATERIAL_ALPHA_TEST) == 0u) {
    return true;
  }
  let pass_through = (1.0 - info.coverage) + info.coverage * params.leaf_transmission;
  let d = bitcast<vec3u>(dir);
  let u = f32(giHash(bitcast<vec3u>(cell) ^ d ^ vec3u(cam.frame_index * 0x9e3779b9u)) >> 8u) / 16777216.0;
  return u >= pass_through;
}

fn traceFrom(p: vec3f, dir: vec3f, distance: f32, max_steps: u32) -> TraceResult {
  let cell = vec3i(floor(p));
  return traceRay(cam.origin_cell + cell, p - vec3f(cell), dir, 0.0, distance, max_steps);
}

/// 1 if nothing blocks `dir` from `start` within `distance`, else 0 (leaves pass
/// stochastically, see traceOpaque).
fn transmittance(start: vec3f, dir: vec3f, distance: f32, max_steps: u32) -> f32 {
  return select(1.0, 0.0, traceFrom(start, dir, distance, max_steps).hit);
}

/// Radiance arriving at `start` from direction `dir` (not pre-exposed).
fn incomingRadiance(start: vec3f, dir: vec3f, noise: vec4f) -> vec3f {
  let r = traceFrom(start, dir, params.range, params.max_steps);
  if (!r.hit) {
    // Missed, or beyond the GI range: the sky (without the sun / moon disks, which are
    // direct light).
    return skyScattering(dir);
  }
  var normal = r.normal;
  if (all(normal == vec3i(0))) {
    normal = vec3i(0, 1, 0);
  }
  let face = faceIndex(normal);
  let material = faceMaterial(r.id, face);
  let hit = start + dir * r.t;
  let emissive = any(block_radiance[r.id].rgb > vec3f(0.0));
  if (params.screen_reuse != 0u && !emissive) {
    let lit = screenRadiance(hit - cam.origin_frac);
    if (lit.a > 0.0) {
      return lit.rgb;
    }
  }
  // Bounce light is low-frequency: the material's mean albedo (last mip of a variant chosen
  // per block) is enough, without the full face mapping.
  let info = materials[material];
  let layer = info.first_layer + hashCellFace(r.cell, face) % max(info.variants, 1u);
  let albedo = srgbDecode(textureSampleLevel(tex_albedo, tex_sampler, vec2f(0.5), layer, params.hit_lod).rgb);
  let n = vec3f(normal);
  let origin = hit + n * SURFACE_OFFSET;

  // Dominant light at the hit: a shadow ray towards a random point of its disk.
  var e = vec3f(0.0);
  let cos_l = dot(n, sky.light_dir);
  if (cos_l > 0.0) {
    let cos_t = mix(1.0, cos(sky.light_radius), noise.x);
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let phi = 2.0 * GI_PI * noise.y;
    let l = giBasis(sky.light_dir) * vec3f(sin_t * cos(phi), sin_t * sin(phi), cos_t);
    e += lighting.light_illuminance * cos_l * transmittance(origin, l, params.shadow_distance, params.shadow_max_steps);
  }
  // Sky at the hit: one short cosine-weighted ray (enclosed spots such as caves get none).
  let sky_dir = cosineSample(n, noise.zw);
  let sky_t = transmittance(origin, sky_dir, params.sky_distance, params.sky_max_steps);
  e += GI_PI * skyScattering(sky_dir) * sky_t;

  var radiance = albedo / GI_PI * e;
  if (params.restir_enabled == 0u) {
    // Emitters seen by the bounce ray; with ReSTIR DI on they are direct light instead.
    radiance += block_radiance[r.id].rgb;
  }
  return radiance;
}

/// Last frame's lit radiance of a camera-relative point if it is visible on screen now
/// (a = 1), else a = 0. Not pre-exposed.
fn screenRadiance(rel: vec3f) -> vec4f {
  let clip = cam.view_proj * vec4f(rel, 1.0);
  if (clip.w <= 0.0) {
    return vec4f(0.0);
  }
  let uv = vec2f(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) {
    return vec4f(0.0);
  }
  let x = vec2i(uv * vec2f(params.full_size));
  let depth = dot(rel, cam.forward);
  if (abs(textureLoad(gdepth, x, 0).x - depth) > SCREEN_DEPTH_TOLERANCE * depth) {
    return vec4f(0.0);
  }
  let prev_uv = uv - textureLoad(gmotion, x, 0).xy;
  if (any(prev_uv < vec2f(0.0)) || any(prev_uv >= vec2f(1.0))) {
    return vec4f(0.0);
  }
  let lit = textureLoad(prev_lit, vec2i(prev_uv * vec2f(params.full_size)), 0).rgb;
  return vec4f(lit / max(history_exposure[0], 1e-20), 1.0);
}

/// Irradiance from the light selected by ReSTIR DI, with its shadow ray (not pre-exposed).
fn emissiveIrradiance(p: vec2u, start: vec3f, n: vec3f, jitter: vec3f) -> vec3f {
  let e = textureLoad(reservoirs, vec2i(p), 0);
  let wm = unpack2x16float(e.w);
  if (wm.y <= 0.0 || wm.x <= 0.0) {
    return vec3f(0.0);
  }
  let cell = vec3i(bitcast<i32>(e.x), i32(e.y & 0xffffu) - 32768, bitcast<i32>(e.z));
  let block = e.y >> 16u;
  // A random point inside the emitting block: soft shadows over frames.
  let target_pos = vec3f(cell - cam.origin_cell) + mix(vec3f(0.1), vec3f(0.9), jitter);
  let to = target_pos - start;
  let d = length(to);
  let l = to / d;
  let cos_n = dot(n, l);
  if (cos_n <= 0.0) {
    return vec3f(0.0);
  }
  let r = traceFrom(start, l, d, params.shadow_max_steps);
  if (r.hit && any(r.cell != cell)) {
    return vec3f(0.0);
  }
  let area = abs(l.x) + abs(l.y) + abs(l.z);
  return block_radiance[block].rgb * area * cos_n / max(d * d, 0.25) * wm.x;
}

fn clampLuminance(c: vec3f, limit: f32) -> vec3f {
  let l = luminance(c);
  return select(c, c * (limit / l), l > limit);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = tracedPixel(gid.xy, cam.frame_index, params.checkerboard);
  if (any(p >= params.half_size)) {
    return;
  }
  let g = textureLoad(guide, vec2i(p), 0);
  if (gbIsSky(g.z)) {
    textureStore(raw_diffuse, vec2i(p), vec4f(0.0));
    textureStore(raw_specular, vec2i(p), vec4f(0.0));
    textureStore(raw_emissive, vec2i(p), vec4f(0.0));
    return;
  }
  let q = fullPixel(p, params.full_size, params.scale);
  let uv = (vec2f(q) + 0.5) / vec2f(params.full_size);
  let view = rayDir(cam.inv_view_proj, uv);
  let n_geo = faceNormal(gbFace(g.z));
  let n = guideNormal(g);
  let start = cam.origin_frac + viewRelativePosition(view, cam.forward, guideDepth(g)) + n_geo * SURFACE_OFFSET;

  let noise0 = blueNoise(blue_noise, p, cam.frame_index, 0u);
  let noise1 = blueNoise(blue_noise, p, cam.frame_index, 1u);
  let noise2 = blueNoise(blue_noise, p, cam.frame_index, 2u);
  let roughness = guideRoughness(g);
  let p_spec = specularProbability(roughness, params.spec_threshold);

  var diffuse = vec3f(0.0);
  var specular = vec3f(0.0);
  if (noise0.z < p_spec) {
    // Specular lobe: GGX visible normals around the shading normal.
    let basis = giBasis(n);
    let v_local = transpose(basis) * -view;
    let alpha = max(roughness * roughness, 1e-3);
    let h = basis * sampleGgxVndf(v_local, alpha, noise0.xy);
    let d = reflect(view, h);
    if (dot(d, n_geo) > 0.0) {
      specular = incomingRadiance(start, d, noise1) / p_spec;
    }
  } else {
    var d = cosineSample(n, noise0.xy);
    // Normal maps can tilt the lobe below the face: mirror such rays back above it.
    let below = dot(d, n_geo);
    if (below <= 0.0) {
      d = normalize(d - 2.0 * below * n_geo);
    }
    diffuse = GI_PI * incomingRadiance(start, d, noise1) / (1.0 - p_spec);
  }
  var emissive = vec3f(0.0);
  if (params.restir_enabled != 0u && params.light_count > 0u) {
    emissive = emissiveIrradiance(p, start, n, noise2.xyz);
  }
  let exposure = preExposure();
  textureStore(raw_emissive, vec2i(p), vec4f(emissive * exposure, 1.0));
  textureStore(raw_diffuse, vec2i(p), vec4f(clampLuminance(diffuse * exposure, params.firefly_clamp), 1.0));
  textureStore(raw_specular, vec2i(p), vec4f(clampLuminance(specular * exposure, params.firefly_clamp), 1.0));
}
