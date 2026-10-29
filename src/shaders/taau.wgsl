// Temporal upsampling (TAAU): the lit image is rendered at a reduced internal resolution
// with a sub-pixel Halton jitter per frame; this pass accumulates the frames into a
// history at the output (canvas) resolution.
//
//   current   this frame's jittered samples around the output pixel, Gaussian-weighted by
//             their distance from it (3×3 internal pixels; the jitter is known)
//   history   last frame's result where this surface was (motion vector of the nearest
//             sample in the 3×3 — edges keep the foreground's motion), Catmull-Rom
//             filtered, rescaled to this frame's exposure
//   reject    disocclusion: the surface's expected depth last frame differs from the depth
//             found there
//   clip      the history is clipped (towards the mean) to mean ± γσ of the 3×3 samples in
//             YCoCg: no trails behind moving objects, no stale lighting
//   blend     1 / frames (capped; lower while moving), luminance-weighted against
//             fireflies; water — waves, reflections, foam move on their own, and the
//             motion vectors follow the bottom — gets a narrower box and a shorter history
//
// Output (rgba16float): pre-exposed colour, a = history length (frames). Sharpening
// (CAS-like) happens in the blit, after tone mapping.
#include "camera.wgsl"
#include "gbuffer.wgsl"

struct TaauParams {
  /// This frame's jitter (internal pixels). The jittered projection moves the image by
  /// +jitter (math.ts jitterProjection), so sample (i, j) sees the scene at
  /// (i, j) + 0.5 − jitter of the unjittered image.
  jitter: vec2f,
  /// History cap (frames) when still / when the camera moves fast.
  frames_still: f32,
  frames_moving: f32,
  /// Clip box half-size in standard deviations; on water.
  gamma: f32,
  gamma_water: f32,
  /// History cap on water.
  frames_water: f32,
  /// Relative depth difference still counted as the same surface.
  depth_tolerance: f32,
  /// Camera motion this frame, 0 (still) … 1 (fast).
  camera_motion: f32,
  /// 1: the history is invalid (resize, first frame).
  reset: u32,
  _pad0: f32,
  _pad1: f32,
};

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: TaauParams;
@group(0) @binding(2) var color_in: texture_2d<f32>;
@group(0) @binding(3) var gdepth: texture_2d<f32>;
@group(0) @binding(4) var gmotion: texture_2d<f32>;
@group(0) @binding(5) var gbuffer0: texture_2d<u32>;
@group(0) @binding(6) var prev_depth: texture_2d<f32>;
@group(0) @binding(7) var history_in: texture_2d<f32>;
@group(0) @binding(8) var history_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var linear_sampler: sampler;
/// Exposure state (exposure.wgsl): [0] = this frame's pre-exposure.
@group(0) @binding(10) var<storage, read> exposure: array<f32>;
/// Pre-exposure the history was stored with ([0]); this pass updates it.
@group(0) @binding(11) var<storage, read_write> history_exposure: array<f32>;

fn rgbToYcocg(c: vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

fn ycocgToRgb(c: vec3f) -> vec3f {
  return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

/// Catmull-Rom history sample from 5 bilinear taps (the corners' weights are negligible).
fn historyCatmullRom(uv: vec2f) -> vec4f {
  let size = vec2f(textureDimensions(history_in));
  let p = uv * size;
  let c = floor(p - 0.5) + 0.5;
  let f = p - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (c - 1.0) / size;
  let t3 = (c + 2.0) / size;
  let t12 = (c + w2 / w12) / size;
  var sum = vec4f(0.0);
  var wsum = 0.0;
  let taps = array<vec3f, 5>(
    vec3f(t12.x, t0.y, w12.x * w0.y),
    vec3f(t0.x, t12.y, w0.x * w12.y),
    vec3f(t12.x, t12.y, w12.x * w12.y),
    vec3f(t3.x, t12.y, w3.x * w12.y),
    vec3f(t12.x, t3.y, w12.x * w3.y),
  );
  for (var k = 0; k < 5; k++) {
    sum += textureSampleLevel(history_in, linear_sampler, taps[k].xy, 0.0) * taps[k].z;
    wsum += taps[k].z;
  }
  return max(sum / max(wsum, 1e-4), vec4f(0.0));
}

/// Clips `h` towards the box centre until it is inside [lo, hi].
fn clipToBox(h: vec3f, lo: vec3f, hi: vec3f) -> vec3f {
  let centre = 0.5 * (hi + lo);
  let extent = max(0.5 * (hi - lo), vec3f(1e-5));
  let d = h - centre;
  let t = max(max(abs(d.x) / extent.x, abs(d.y) / extent.y), abs(d.z) / extent.z);
  return select(h, centre + d / t, t > 1.0);
}

/// Colour (rgb) and depth (a) of internal pixel `q` (clamped). Sharing the 3×3
/// neighbourhoods through workgroup memory was tried: no faster (the texture cache already
/// serves the neighbours; the barrier and the shared array cost as much).
fn sampleAt(q: vec2i, in_size: vec2i) -> vec4f {
  let c = clamp(q, vec2i(0), in_size - 1);
  return vec4f(textureLoad(color_in, c, 0).rgb, textureLoad(gdepth, c, 0).x);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let out_size = textureDimensions(history_out);
  // Active internal region (dynamic resolution); every internal read stays inside it.
  let in_size = vec2i(cam.size);
  resolve(gid.xy, out_size, in_size);
}

fn resolve(gid: vec2u, out_size: vec2u, in_size: vec2i) {
  if (any(gid >= out_size)) {
    return;
  }
  let uv = (vec2f(gid) + 0.5) / vec2f(out_size);
  // Output pixel in sample-index space: sample q sits at q + 0.5 − jitter.
  let p = uv * vec2f(in_size) - 0.5 + params.jitter;
  let centre = vec2i(round(p));
  // Gaussian falloff wide enough to cover the gaps between samples when upscaling.
  let scale = max(f32(out_size.x) / f32(in_size.x), 1.0);
  let inv_two_sigma2 = 1.0 / (2.0 * 0.47 * 0.47 * min(scale, 1.6));

  var sum = vec3f(0.0);
  var wsum = 0.0;
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var nearest = centre;
  var nearest_depth = 1e30;
  for (var k = 0; k < 9; k++) {
    let o = vec2i(k % 3 - 1, k / 3 - 1);
    let q = clamp(centre + o, vec2i(0), in_size - 1);
    let sample = sampleAt(centre + o, in_size);
    let c = sample.rgb;
    let d = vec2f(q) - p;
    let w = exp(-dot(d, d) * inv_two_sigma2);
    sum += c * w;
    wsum += w;
    let y = rgbToYcocg(c);
    m1 += y;
    m2 += y * y;
    // Nearest surface among the centre and its four edge neighbours (the corners add
    // little: edges are found by the cross already).
    if (o.x * o.y == 0 && sample.a < nearest_depth) {
      nearest_depth = sample.a;
      nearest = q;
    }
  }
  let current = sum / max(wsum, 1e-5);
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
  let water = gbThroughWater(textureLoad(gbuffer0, clamp(centre, vec2i(0), in_size - 1), 0).w);
  let gamma = select(params.gamma, params.gamma_water, water);

  let motion = textureLoad(gmotion, nearest, 0).xy;
  let prev_uv = uv - motion;
  var result = current;
  var frames = 1.0;
  let inside = all(prev_uv >= vec2f(0.0)) && all(prev_uv < vec2f(1.0));
  if (params.reset == 0u && inside) {
    // Disocclusion: the nearest sample's surface, reprojected, must match last frame's depth.
    // Pixel centre through the jittered inverse: exactly the ray this sample traced.
    let q_uv = (vec2f(nearest) + 0.5) / vec2f(in_size);
    let rel = viewRelativePosition(rayDir(cam.inv_view_proj, q_uv), cam.forward, nearest_depth);
    let expected = dot(rel + cam.prev_delta, cam.prev_forward);
    // Last frame's depth is at last frame's active size.
    let prev_in = vec2i(cam.prev_size);
    let prev_px = clamp(vec2i(prev_uv * vec2f(prev_in)), vec2i(0), prev_in - 1);
    let tolerance = params.depth_tolerance * max(expected, 1.0);
    let sky = nearest_depth >= cam.far * 0.999;
    var same = sky || abs(textureLoad(prev_depth, prev_px, 0).x - expected) <= tolerance;
    // Last frame's depth was sampled with last frame's jitter: on a silhouette the pixel
    // there may hold the other side of the edge. The surface counts as seen if any pixel
    // of the 3×3 held it (rejecting there reset the history every frame: edges flickered).
    for (var k = 0; k < 9 && !same; k++) {
      let t = clamp(prev_px + vec2i(k % 3 - 1, k / 3 - 1), vec2i(0), prev_in - 1);
      same = abs(textureLoad(prev_depth, t, 0).x - expected) <= tolerance;
    }
    if (same) {
      let h = historyCatmullRom(prev_uv);
      let rescale = exposure[0] / max(history_exposure[0], 1e-20);
      let clipped = ycocgToRgb(clipToBox(rgbToYcocg(h.rgb * rescale), mean - gamma * sigma, mean + gamma * sigma));
      let moving = clamp(params.camera_motion + length(motion * vec2f(in_size)) / 8.0, 0.0, 1.0);
      let cap = select(mix(params.frames_still, params.frames_moving, moving), params.frames_water, water);
      frames = min(h.a + 1.0, cap);
      // Luminance-weighted blend: bright outliers do not dominate (no fireflies).
      let wc = (1.0 / frames) / (1.0 + luma(current));
      let wh = (1.0 - 1.0 / frames) / (1.0 + luma(clipped));
      result = (current * wc + clipped * wh) / (wc + wh);
    }
  }
  textureStore(history_out, vec2i(gid), vec4f(result, frames));
}

/// One invocation: remembers this frame's pre-exposure for the next frame's history.
@compute @workgroup_size(1)
fn storeExposure() {
  history_exposure[0] = exposure[0];
}
