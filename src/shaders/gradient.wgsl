#include "common.wgsl"

struct Params {
  size: vec2u,
  time: f32,
  speed: f32,
  wave_strength: f32,
};

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= params.size)) {
    return;
  }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(params.size);
  let t = params.time * params.speed;
  let wave = sin(uv.x * TAU * 2.0 + t * TAU) * cos(uv.y * TAU + t * PI);
  let k = uv.x * 0.6 + uv.y * 0.4 + wave * params.wave_strength + t;
  let color = palette(
    k,
    vec3f(0.5, 0.5, 0.5),
    vec3f(0.5, 0.5, 0.5),
    vec3f(1.0, 1.0, 1.0),
    vec3f(0.0, 0.33, 0.67),
  );
  textureStore(output, vec2i(id.xy), vec4f(color, 1.0));
}
