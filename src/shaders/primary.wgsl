// Primary rays: one per pixel from the camera into the brickmap, results to the G-buffer.
#include "trace.wgsl"
#include "gbuffer.wgsl"

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
  _pad0: f32,
  forward: vec3f,
  _pad1: f32,
  size: vec2u,
  _pad2: vec2u,
};

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var gbuffer0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var gdepth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var gmotion: texture_storage_2d<rg32float, write>;
/// Per block id: face material indices (top, side, bottom, unused).
@group(0) @binding(4) var<storage, read> block_faces: array<vec4u>;
/// Per material: linear albedo in xyz.
@group(0) @binding(5) var<storage, read> material_albedo: array<vec4f>;

fn clipToUv(clip: vec4f) -> vec2f {
  let ndc = clip.xy / clip.w;
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= cam.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  // Any point on the ray through the pixel; the camera sits at the origin of this space.
  let h = cam.inv_view_proj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
  let dir = normalize(h.xyz / h.w);

  let r = traceRay(cam.origin_cell, cam.origin_frac, dir, cam.far, cam.max_steps);

  if (!r.hit) {
    // Sky: reproject the direction only (infinitely far, camera translation irrelevant).
    let motion = clipToUv(cam.view_proj * vec4f(dir, 0.0)) - clipToUv(cam.prev_view_proj * vec4f(dir, 0.0));
    textureStore(gbuffer0, px, vec4u(0u, 0u, 0u, packMaterialWord(0u, 0u, r.steps, true)));
    textureStore(gdepth, px, vec4f(cam.far, 0.0, 0.0, 0.0));
    textureStore(gmotion, px, vec4f(motion, 0.0, 0.0));
    return;
  }

  var n = vec3f(r.normal);
  if (all(r.normal == vec3i(0))) {
    // Started inside a block: face the camera along the dominant axis.
    let a = abs(dir);
    n = select(select(vec3f(0.0, 0.0, -sign(dir.z)), vec3f(0.0, -sign(dir.y), 0.0), a.y > a.z), vec3f(-sign(dir.x), 0.0, 0.0), a.x > a.y && a.x > a.z);
  }

  let faces = block_faces[r.id];
  let material = select(select(faces.y, faces.z, n.y < -0.5), faces.x, n.y > 0.5);
  let albedo = material_albedo[material].xyz;

  // Position inside the hit voxel in [0, 1]³ (camera-relative, so precise).
  let rel = cam.origin_frac + dir * r.t;
  let local = clamp(rel - vec3f(r.cell - cam.origin_cell), vec3f(0.0), vec3f(1.0));
  var face_uv: vec2f;
  if (abs(n.x) > 0.5) {
    face_uv = vec2f(local.z, 1.0 - local.y);
  } else if (abs(n.y) > 0.5) {
    face_uv = local.xz;
  } else {
    face_uv = vec2f(local.x, 1.0 - local.y);
  }

  let p = dir * r.t;
  let motion = clipToUv(cam.view_proj * vec4f(p, 1.0)) - clipToUv(cam.prev_view_proj * vec4f(p + cam.prev_delta, 1.0));

  textureStore(gbuffer0, px, vec4u(
    pack4x8unorm(vec4f(albedo, 1.0)),
    pack2x16snorm(octEncode(n)),
    pack2x16unorm(face_uv),
    packMaterialWord(r.id, material, r.steps, false),
  ));
  textureStore(gdepth, px, vec4f(r.t * dot(dir, cam.forward), 0.0, 0.0, 0.0));
  textureStore(gmotion, px, vec4f(motion, 0.0, 0.0));
}
