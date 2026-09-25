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

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
  return vec4f(textureSample(source, source_sampler, in.uv).rgb, 1.0);
}
