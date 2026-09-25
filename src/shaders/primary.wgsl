// Primary rays: one per pixel from the camera into the brickmap; materials are sampled at
// the hit and the result goes to the G-buffer.
#include "trace.wgsl"
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "material.wgsl"

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var gbuffer0: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(2) var gdepth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var gmotion: texture_storage_2d<rg32float, write>;
/// Depth prepass: per tile, the distance before which no ray of the tile hits anything.
@group(0) @binding(4) var coarse: texture_2d<f32>;

/// Alpha-tested materials (leaves) let the ray through transparent texels.
fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
  return alphaOpaque(id, cell, normal, local, dir, t);
}

fn clipToUv(clip: vec4f) -> vec2f {
  let ndc = clip.xy / clip.w;
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= cam.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  let dir = rayDir(cam.inv_view_proj, uv);
  var t_min = 0.0;
  if (cam.prepass_tile > 0u) {
    t_min = textureLoad(coarse, vec2i(gid.xy / cam.prepass_tile), 0).x;
  }

  let r = traceRay(cam.origin_cell, cam.origin_frac, dir, t_min, cam.far, cam.max_steps);

  if (!r.hit) {
    // Sky: reproject the direction only (infinitely far, camera translation irrelevant).
    let motion = clipToUv(cam.view_proj * vec4f(dir, 0.0)) - clipToUv(cam.prev_view_proj * vec4f(dir, 0.0));
    textureStore(gbuffer0, px, vec4u(0u, 0u, 0u, packMaterialWord(0u, 0u, 0u, r.steps, true)));
    textureStore(gdepth, px, vec4f(cam.far, 0.0, 0.0, 0.0));
    textureStore(gmotion, px, vec4f(motion, 0.0, 0.0));
    return;
  }

  var normal = r.normal;
  if (all(normal == vec3i(0))) {
    // Started inside a block: face the camera along the dominant axis.
    let a = abs(dir);
    let axis = select(select(2u, 1u, a.y > a.z), 0u, a.x > a.y && a.x > a.z);
    normal[axis] = select(1, -1, dir[axis] > 0.0);
  }
  let face = faceIndex(normal);
  let material = faceMaterial(r.id, face);

  // Position inside the hit voxel in [0, 1]³ (camera-relative, so precise).
  let rel = cam.origin_frac + dir * r.t;
  let local = clamp(rel - vec3f(r.cell - cam.origin_cell), vec3f(0.0), vec3f(1.0));
  let s = shadeSurface(material, r.cell, face, local, dir, r.t);

  let p = dir * r.t;
  let motion = clipToUv(cam.view_proj * vec4f(p, 1.0)) - clipToUv(cam.prev_view_proj * vec4f(p + cam.prev_delta, 1.0));

  textureStore(gbuffer0, px, vec4u(
    pack4x8unorm(vec4f(s.albedo, s.ao)),
    pack2x16snorm(octEncode(s.normal)),
    pack4x8unorm(vec4f(s.roughness, s.metalness, s.emission, s.subsurface)),
    packMaterialWord(r.id, material, face, r.steps, false),
  ));
  textureStore(gdepth, px, vec4f(r.t * dot(dir, cam.forward), 0.0, 0.0, 0.0));
  textureStore(gmotion, px, vec4f(motion, 0.0, 0.0));
}
