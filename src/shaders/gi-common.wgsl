// Shared by the global illumination passes (restir.wgsl, gi-trace.wgsl, gi-denoise.wgsl)
// and the lighting pass (upsampling). Every includer declares `cam: Camera` and, except
// lighting, `params: GiParams`.
//
// GI runs at reduced resolution (divisor `scale`, config gi.resolutionDivisor): GI pixel p
// stands for full-res pixel p · scale ("half-res" below means this grid). Its "guide"
// (rgba32uint) caches what the half-res passes need from the G-buffer:
//   x  linear view depth (f32 bits)
//   y  shading normal, octahedral pack2x16snorm
//   z  gbuffer0.w (block, material, geometric face, sky flag) with the DDA step bits
//      replaced by the roughness (12 bits)
//   w  plane coordinate: dot(face normal, camera-relative position) (f32 bits); two
//      pixels on the same face plane have equal values (cheap edge-stopping)
#include "gbuffer.wgsl"
#include "camera.wgsl"

struct GiParams {
  half_size: vec2u,
  full_size: vec2u,
  /// Traced pixels per frame: 0 all half-res pixels, 1 half (alternating checkerboard),
  /// 2 a quarter (one per 2×2 block, rotating).
  checkerboard: u32,
  /// Bounce ray length (blocks) and step budget; beyond it the sky LUT is used.
  range: f32,
  max_steps: u32,
  /// Shadow and sky rays from bounce hits.
  shadow_distance: f32,
  shadow_max_steps: u32,
  sky_distance: f32,
  sky_max_steps: u32,
  leaf_transmission: f32,
  /// Surfaces with (1 − roughness)² below this get no specular rays.
  spec_threshold: f32,
  /// Texture LOD for albedo at bounce hits (coarse: a few texels per face).
  hit_lod: f32,
  /// Per-sample luminance ceiling (pre-exposed) against fireflies.
  firefly_clamp: f32,
  light_count: u32,
  restir_enabled: u32,
  restir_candidates: u32,
  /// Temporal reuse: history reservoirs are capped at this many samples.
  restir_max_m: f32,
  restir_spatial_count: u32,
  restir_spatial_radius: f32,
  history_still: f32,
  history_moving: f32,
  /// Camera motion this frame, 0 (still) … 1 (fast).
  camera_motion: f32,
  motion_pixels: f32,
  depth_tolerance: f32,
  sigma_luminance: f32,
  /// "Same surface" test (blocks): distance between face planes. Voxel faces are exactly
  /// planar and parallel faces lie whole blocks apart, so this is absolute, not relative to
  /// depth (a relative tolerance mixed neighbouring staircase faces far away).
  plane_tolerance: f32,
  sigma_luminance_spec: f32,
  spec_roughness_sigma: f32,
  /// Bounce hits visible on screen take last frame's lit radiance (0 = always shade).
  screen_reuse: u32,
  /// GI grid divisor: GI pixel p stands for full-res pixel p · scale.
  scale: u32,
};

const GI_PI: f32 = 3.14159265358979;
const GOLDEN_RATIO_FRACT: f32 = 0.61803398875;

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn giHash(v: vec3u) -> u32 {
  var h = (v.x * 0x8da6b343u) ^ (v.y * 0xd8163841u) ^ (v.z * 0xcb1ab31fu);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

/// Uniform random number in [0, 1) from a pixel, frame and a running counter.
fn giRandom(p: vec2u, frame: u32, counter: ptr<function, u32>) -> f32 {
  *counter += 1u;
  return f32(giHash(vec3u(p, frame * 64u + *counter)) >> 8u) / 16777216.0;
}

// ------------------------------------------------------------------------ guide

const GUIDE_ROUGHNESS_MASK: u32 = 0xfffu << GB_STEPS_SHIFT;

fn packGuide(depth: f32, normal: vec3f, word: u32, roughness: f32, plane: f32) -> vec4u {
  let r = u32(round(clamp(roughness, 0.0, 1.0) * 4095.0)) << GB_STEPS_SHIFT;
  return vec4u(bitcast<u32>(depth), pack2x16snorm(octEncode(normal)), (word & ~GUIDE_ROUGHNESS_MASK) | r, bitcast<u32>(plane));
}

fn guidePlane(g: vec4u) -> f32 {
  return bitcast<f32>(g.w);
}

/// Whether guide `g` lies on the face plane (`face`, plane coordinate `plane`).
fn onPlane(g: vec4u, face: u32, plane: f32, tolerance: f32) -> bool {
  return !gbIsSky(g.z) && gbFace(g.z) == face && abs(guidePlane(g) - plane) <= tolerance;
}

/// The half-res pixel traced by compact thread `id` this frame (the dispatch covers only
/// traced pixels, so no SIMD lanes idle next to working ones). Inverse of isTraced.
fn tracedPixel(id: vec2u, frame: u32, pattern: u32) -> vec2u {
  if (pattern == 0u) {
    return id;
  }
  if (pattern == 1u) {
    return vec2u(id.x * 2u + ((id.y + frame) & 1u), id.y);
  }
  return id * 2u + array<vec2u, 4>(vec2u(0u, 0u), vec2u(1u, 1u), vec2u(1u, 0u), vec2u(0u, 1u))[frame & 3u];
}

/// Whether half-res pixel `p` is traced this frame.
fn isTraced(p: vec2u, frame: u32, pattern: u32) -> bool {
  if (pattern == 0u) {
    return true;
  }
  if (pattern == 1u) {
    return ((p.x + p.y + frame) & 1u) == 0u;
  }
  let o = array<vec2u, 4>(vec2u(0u, 0u), vec2u(1u, 1u), vec2u(1u, 0u), vec2u(0u, 1u))[frame & 3u];
  return all((p & vec2u(1u)) == o);
}

fn guideDepth(g: vec4u) -> f32 {
  return bitcast<f32>(g.x);
}

fn guideNormal(g: vec4u) -> vec3f {
  return octDecode(unpack2x16snorm(g.y));
}

fn guideRoughness(g: vec4u) -> f32 {
  return f32((g.z & GUIDE_ROUGHNESS_MASK) >> GB_STEPS_SHIFT) / 4095.0;
}

/// Full-res pixel a GI pixel stands for (GI grid divisor `scale`).
fn fullPixel(p: vec2u, full: vec2u, scale: u32) -> vec2u {
  return min(p * scale, full - 1u);
}

/// Continuous GI-grid coordinate of a full-res uv (GI pixel p sits at full-res pixel
/// p · scale).
fn giGridCoord(uv: vec2f, full: vec2u, scale: u32) -> vec2f {
  return (uv * vec2f(full) - 0.5) / f32(scale);
}

/// Camera-relative position of a full-res pixel at linear view depth `depth`.
fn pixelPosition(q: vec2u, full: vec2u, depth: f32) -> vec3f {
  let uv = (vec2f(q) + 0.5) / vec2f(full);
  return viewRelativePosition(rayDir(cam.inv_view_proj, uv), cam.forward, depth);
}

/// Previous frame's expected linear depth of a camera-relative point (for reprojection).
fn previousDepth(p: vec3f) -> f32 {
  return dot(p + cam.prev_delta, cam.prev_forward);
}

fn giLatticeValue(c: vec3i) -> f32 {
  return f32(giHash(bitcast<vec3u>(c)) >> 8u) / 16777216.0;
}

/// Smooth 3D value noise in [0, 1] (flicker field).
fn giValueNoise(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let x00 = mix(giLatticeValue(i), giLatticeValue(i + vec3i(1, 0, 0)), u.x);
  let x10 = mix(giLatticeValue(i + vec3i(0, 1, 0)), giLatticeValue(i + vec3i(1, 1, 0)), u.x);
  let x01 = mix(giLatticeValue(i + vec3i(0, 0, 1)), giLatticeValue(i + vec3i(1, 0, 1)), u.x);
  let x11 = mix(giLatticeValue(i + vec3i(0, 1, 1)), giLatticeValue(i + vec3i(1, 1, 1)), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

/// Flicker factor of emitted light around world position `p` at `time` (mean 1): two
/// octaves of noise drifting through a smooth spatial field, so neighbouring torches
/// flicker independently while each lit surface follows the torch next to it.
fn flicker(p: vec3f, time: f32, amount: f32, speed: f32, scale: f32) -> f32 {
  let q = p / scale;
  let slow = giValueNoise(q + vec3f(0.0, time * speed, 0.0));
  let fast = giValueNoise(q * 1.7 + vec3f(31.0, time * speed * 2.9, 17.0));
  return 1.0 + amount * (2.0 * (0.65 * slow + 0.35 * fast) - 1.0) * 2.0;
}

// ------------------------------------------------------------------------ sampling

/// Spatiotemporal blue noise (64 × 64 × 16 slices, four independent channels). Each
/// `stream` reads the tile at a different offset (decorrelated); after every 16-frame cycle
/// the values are shifted by the golden ratio so the sequence keeps going.
fn blueNoise(tex: texture_2d_array<f32>, p: vec2u, frame: u32, stream: u32) -> vec4f {
  let size = textureDimensions(tex);
  let slices = textureNumLayers(tex);
  let cycle = frame / slices;
  let s = f32(stream + cycle * 7u);
  let offset = vec2u(fract(vec2f(s * 0.7548776662, s * 0.5698402910)) * vec2f(size));
  let v = textureLoad(tex, (p + offset) % size, frame % slices, 0);
  return fract(v + f32(cycle) * GOLDEN_RATIO_FRACT);
}

/// Orthonormal basis around `n` (Frisvad / Duff et al.): columns t, b, n.
fn giBasis(n: vec3f) -> mat3x3f {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return mat3x3f(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}

/// Cosine-weighted direction around `n` (pdf = cos / π).
fn cosineSample(n: vec3f, u: vec2f) -> vec3f {
  let r = sqrt(u.x);
  let phi = 2.0 * GI_PI * u.y;
  return giBasis(n) * vec3f(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

/// GGX visible-normal sample (Heitz 2018) in the local frame (z = normal) for view `v`.
fn sampleGgxVndf(v: vec3f, alpha: f32, u: vec2f) -> vec3f {
  let vh = normalize(vec3f(alpha * v.x, alpha * v.y, v.z));
  let len2 = vh.x * vh.x + vh.y * vh.y;
  let t1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(max(len2, 1e-12)), len2 > 0.0);
  let t2 = cross(vh, t1);
  let r = sqrt(u.x);
  let phi = 2.0 * GI_PI * u.y;
  let p1 = r * cos(phi);
  let s = 0.5 * (1.0 + vh.z);
  let p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * r * sin(phi);
  let nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
  return normalize(vec3f(alpha * nh.x, alpha * nh.y, max(nh.z, 0.0)));
}

/// Probability of tracing the specular lobe; 0 for rough surfaces (their specular is
/// taken from the diffuse irradiance when shading).
fn specularProbability(roughness: f32, threshold: f32) -> f32 {
  let s = (1.0 - roughness) * (1.0 - roughness);
  return select(0.0, s, s >= threshold && threshold < 1.0);
}
