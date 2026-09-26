// Camera uniform shared by the primary ray passes (layout filled in primary-pass.ts).

struct Camera {
  /// Rotation-only (camera at the origin), jittered: ray generation.
  inv_view_proj: mat4x4f,
  /// Rotation-only, unjittered, this and the previous frame: motion vectors.
  view_proj: mat4x4f,
  prev_view_proj: mat4x4f,
  origin_cell: vec3i,
  max_steps: u32,
  origin_frac: vec3f,
  far: f32,
  /// Camera position now − previous frame (world units).
  prev_delta: vec3f,
  /// Depth prepass: spread of a tile's rays around its centre ray per unit distance.
  cone_k: f32,
  forward: vec3f,
  /// Depth prepass: distance subtracted from the coarse result before use.
  prepass_safety: f32,
  size: vec2u,
  /// Depth prepass tile edge in pixels; 0 = prepass disabled.
  prepass_tile: u32,
  prepass_max_steps: u32,
  /// Previous frame's forward axis (reprojected depth for temporal accumulation).
  prev_forward: vec3f,
  /// Increments every frame (noise seeds).
  frame_index: u32,
};

/// World position of a G-buffer pixel relative to the camera, from its linear depth.
fn viewRelativePosition(dir: vec3f, forward: vec3f, depth: f32) -> vec3f {
  return dir * (depth / dot(dir, forward));
}

fn rayDir(inv_view_proj: mat4x4f, uv: vec2f) -> vec3f {
  // Any point on the ray through uv; the camera sits at the origin of this space.
  let h = inv_view_proj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
  return normalize(h.xyz / h.w);
}
