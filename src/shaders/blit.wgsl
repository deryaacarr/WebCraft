struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

// Single oversized triangle covering the screen; no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
  let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var out: VertexOut;
  out.position = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  return out;
}

#include "color.wgsl"

struct BlitParams {
  /// 0 = show linear values as they are, 1 = ACES tone mapping (pre-exposed HDR input).
  tonemap: u32,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BlitParams;

// ACES filmic fit by Stephen Hill (RRT + ODT), sRGB primaries in and out.
const ACES_IN: mat3x3f = mat3x3f(
  vec3f(0.59719, 0.07600, 0.02840),
  vec3f(0.35458, 0.90834, 0.13383),
  vec3f(0.04823, 0.01566, 0.83777),
);
const ACES_OUT: mat3x3f = mat3x3f(
  vec3f(1.60475, -0.10208, -0.00327),
  vec3f(-0.53108, 1.10813, -0.07276),
  vec3f(-0.07367, -0.00605, 1.07602),
);

fn aces(color: vec3f) -> vec3f {
  let v = ACES_IN * color;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(ACES_OUT * (a / b), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
  var c = textureSample(source, source_sampler, in.uv).rgb;
  if (params.tonemap == 1u) {
    c = aces(c);
  }
  // Scene textures are linear; the swap chain is a plain (non-sRGB) format.
  return vec4f(linearToSrgb(c), 1.0);
}
