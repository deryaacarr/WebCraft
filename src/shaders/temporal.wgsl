// Temporal accumulation of the visibility samples (visibility.wgsl).
// Output / history: r, g = sun / sky visibility, b = unused, a = history length (frames).
//
//   1. Reprojection: history is fetched at the motion-vector position and discarded
//      unless depth, geometric face and material all match the previous frame there
//      (disocclusion, e.g. a new surface sliding in behind a moving edge).
//   2. Neighbourhood clipping: history is clipped to mean ± k·σ of this frame's samples
//      around the pixel, so stale shadows cannot trail behind moving geometry.
//   3. History length per pixel, capped at max_static (still) … max_moving (fast camera
//      or fast screen motion); the blend weight of new samples grows with motion.
//   4. Checkerboard: a pixel not traced this frame keeps its (clipped) history; if that
//      history is invalid it is rebuilt from this frame's traced neighbours, weighted by
//      depth and face similarity.
#include "camera.wgsl"
#include "gbuffer.wgsl"

struct TemporalParams {
  max_static: f32,
  max_moving: f32,
  /// Camera motion this frame, 0 (still) … 1 (fast); per-pixel motion is added on top.
  camera_motion: f32,
  /// Screen motion (pixels per frame) at which a pixel counts as fully moving.
  motion_pixels: f32,
  depth_tolerance: f32,
  clip_k: f32,
  /// Visibility tracing block size (see visibility.wgsl): 1 or 2.
  block: u32,
  _pad: f32,
};

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: TemporalParams;
@group(0) @binding(2) var current: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var gdepth: texture_2d<f32>;
@group(0) @binding(5) var gmotion: texture_2d<f32>;
@group(0) @binding(6) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var gbuffer0: texture_2d<u32>;
@group(0) @binding(8) var prev_gbuffer0: texture_2d<u32>;
@group(0) @binding(9) var prev_depth: texture_2d<f32>;

fn tracedOffset(frame: u32) -> vec2u {
  return array<vec2u, 4>(vec2u(0u, 0u), vec2u(1u, 1u), vec2u(1u, 0u), vec2u(0u, 1u))[frame & 3u];
}

/// Pixel traced this frame in block `b` (block coordinates), or −1 if off screen.
fn tracedPixel(b: vec2i) -> vec2i {
  let p = b * i32(params.block) + vec2i(select(vec2u(0u), tracedOffset(cam.frame_index), params.block == 2u));
  if (any(p < vec2i(0)) || any(p >= vec2i(cam.size))) {
    return vec2i(-1);
  }
  return p;
}

/// Material identity for history validation: block id, material and geometric face.
fn surfaceKey(word: u32) -> u32 {
  return word & 0x7ffffu;
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= cam.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  let word = textureLoad(gbuffer0, px, 0).w;
  if (gbIsSky(word)) {
    textureStore(output, px, vec4f(1.0, 1.0, 0.0, 0.0));
    return;
  }
  let depth = textureLoad(gdepth, px, 0).x;
  let key = surfaceKey(word);
  let blk = vec2i(gid.xy / params.block);
  let own = tracedPixel(blk);
  let traced = all(own == px);

  // This frame's samples around the pixel (3×3 traced pixels), for clipping and for
  // rebuilding pixels without history. Weighted by depth and surface similarity.
  var m1 = vec2f(0.0);
  var m2 = vec2f(0.0);
  var n = 0.0;
  var rebuilt = vec2f(0.0);
  var wsum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = tracedPixel(blk + vec2i(x, y));
      if (q.x < 0) {
        continue;
      }
      let s = textureLoad(current, q, 0).rg;
      let qd = textureLoad(gdepth, q, 0).x;
      let same = select(0.05, 1.0, surfaceKey(textureLoad(gbuffer0, q, 0).w) == key);
      let w = exp(-abs(qd - depth) / (params.depth_tolerance * depth + 0.1)) * same;
      m1 += s * w;
      m2 += s * s * w;
      n += w;
      rebuilt += s * w;
      wsum += w;
    }
  }
  let mean = m1 / max(n, 1e-4);
  let sigma = sqrt(max(m2 / max(n, 1e-4) - mean * mean, vec2f(0.0)));
  let sample = select(rebuilt / max(wsum, 1e-4), textureLoad(current, px, 0).rg, traced);

  // Motion: camera (from the CPU) plus this pixel's own screen motion.
  let motion = textureLoad(gmotion, px, 0).xy;
  let moving = clamp(params.camera_motion + length(motion * vec2f(cam.size)) / params.motion_pixels, 0.0, 1.0);
  let max_frames = mix(params.max_static, params.max_moving, moving);

  var result = sample;
  var frames = 1.0;
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  let prev_uv = uv - motion;
  if (all(prev_uv >= vec2f(0.0)) && all(prev_uv < vec2f(1.0))) {
    let prev_px = vec2i(prev_uv * vec2f(cam.size));
    let h = textureLoad(history, prev_px, 0);
    let rel = viewRelativePosition(rayDir(cam.inv_view_proj, uv), cam.forward, depth);
    let expected = dot(rel + cam.prev_delta, cam.prev_forward);
    let depth_ok = abs(textureLoad(prev_depth, prev_px, 0).x - expected) <= params.depth_tolerance * max(expected, 1.0);
    let surface_ok = surfaceKey(textureLoad(prev_gbuffer0, prev_px, 0).w) == key;
    if (h.a > 0.0 && depth_ok && surface_ok) {
      let clipped = clamp(h.rg, mean - params.clip_k * sigma, mean + params.clip_k * sigma);
      frames = min(select(h.a, h.a + 1.0, traced), max_frames);
      result = select(clipped, mix(clipped, sample, 1.0 / frames), traced);
    }
  }
  textureStore(output, px, vec4f(result, 0.0, frames));
}

// ---------------------------------------------------------------------------------------
// Spatial filter of the accumulated visibility for display (history stays unfiltered):
// cross-bilateral over a 5×5 footprint sampled sparsely (3×3 taps, 2 px apart: 9 instead
// of 25 loads), depth- and surface-aware. Pixels with little history (just
// disoccluded, or a fast camera) are smoothed more, converged ones hardly at all.

@group(0) @binding(10) var accumulated: texture_2d<f32>;
@group(0) @binding(11) var filtered: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn spatial(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= cam.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  let centre = textureLoad(accumulated, px, 0);
  let word = textureLoad(gbuffer0, px, 0).w;
  if (gbIsSky(word)) {
    textureStore(filtered, px, centre);
    return;
  }
  let depth = textureLoad(gdepth, px, 0).x;
  let key = surfaceKey(word);
  // Spatial sigma (pixels): wide for fresh history, narrow once converged.
  let confidence = clamp(centre.a / params.max_static, 0.0, 1.0);
  let sigma = mix(2.5, 0.75, confidence);
  var sum = vec2f(0.0);
  var wsum = 0.0;
  for (var y = -2; y <= 2; y += 2) {
    for (var x = -2; x <= 2; x += 2) {
      let q = px + vec2i(x, y);
      if (any(q < vec2i(0)) || any(q >= vec2i(cam.size))) {
        continue;
      }
      let qw = textureLoad(gbuffer0, q, 0).w;
      if (gbIsSky(qw)) {
        continue;
      }
      let qd = textureLoad(gdepth, q, 0).x;
      let same = select(0.0, 1.0, surfaceKey(qw) == key);
      let w = exp(-f32(x * x + y * y) / (2.0 * sigma * sigma)) *
        exp(-abs(qd - depth) / (params.depth_tolerance * depth + 0.05)) * same;
      sum += textureLoad(accumulated, q, 0).rg * w;
      wsum += w;
    }
  }
  textureStore(filtered, px, vec4f(sum / max(wsum, 1e-4), 0.0, centre.a));
}
