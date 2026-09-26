// Temporal accumulation of the visibility samples (visibility.wgsl). History layout:
//   r, g  accumulated sun / sky visibility
//   b     linear depth of the pixel when it was written
//   a     number of accumulated frames
// History is fetched at the reprojected position (motion vectors) and rejected when the
// depth it recorded does not match the depth this surface had in the previous frame.
#include "camera.wgsl"

struct TemporalParams {
  max_frames: f32,
  depth_tolerance: f32,
  /// Visibility tracing block size (see visibility.wgsl): 1 or 2.
  block: u32,
  _pad: f32,
};

fn tracedOffset(frame: u32) -> vec2u {
  return array<vec2u, 4>(vec2u(0u, 0u), vec2u(1u, 1u), vec2u(1u, 0u), vec2u(0u, 1u))[frame & 3u];
}

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: TemporalParams;
@group(0) @binding(2) var current: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var gdepth: texture_2d<f32>;
@group(0) @binding(5) var gmotion: texture_2d<f32>;
@group(0) @binding(6) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= cam.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  // Pixels not traced this frame borrow the sample of their block's traced pixel when
  // they have no usable history; otherwise they keep their history unchanged.
  var traced = true;
  var sample_px = px;
  if (params.block == 2u) {
    let offset = tracedOffset(cam.frame_index);
    let block_px = (gid.xy / 2u) * 2u + offset;
    traced = all(gid.xy == block_px);
    sample_px = vec2i(min(block_px, cam.size - 1u));
  }
  let sample = textureLoad(current, sample_px, 0).rg;
  let depth = textureLoad(gdepth, px, 0).x;
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  let prev_uv = uv - textureLoad(gmotion, px, 0).xy;

  var accumulated = sample;
  var frames = 1.0;
  if (all(prev_uv >= vec2f(0.0)) && all(prev_uv < vec2f(1.0))) {
    let h = textureLoad(history, vec2i(prev_uv * vec2f(cam.size)), 0);
    // Where this surface was, seen from the previous camera.
    let rel = viewRelativePosition(rayDir(cam.inv_view_proj, uv), cam.forward, depth);
    let expected = dot(rel + cam.prev_delta, cam.prev_forward);
    if (h.a > 0.0 && abs(h.b - expected) <= params.depth_tolerance * max(expected, 1.0)) {
      if (traced) {
        frames = min(h.a + 1.0, params.max_frames);
        accumulated = mix(h.rg, sample, 1.0 / frames);
      } else {
        frames = h.a;
        accumulated = h.rg;
      }
    }
  }
  textureStore(output, px, vec4f(accumulated, depth, frames));
}
