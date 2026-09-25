// Depth prepass: one conservative "cone" ray per tile of prepass_tile² pixels. It records
// a distance before which none of the tile's rays can hit anything; the primary pass
// starts its rays there.
//
// Every ray of the tile stays within r(t) = cone_k · t of the tile's centre ray at
// distance t. The centre ray therefore only advances while the cube of half-size r(t)
// around it lies in space known to be empty: outside the occupied box (bounds) or inside
// a distance-field box. It stops at the first occupied brick or when that margin runs out.
#include "trace.wgsl"

/// No materials here: every non-air voxel is opaque (see trace.wgsl).
fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool {
  return true;
}
#include "camera.wgsl"

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var coarse: texture_storage_2d<r32float, write>;

fn coarseDistance(dir: vec3f) -> f32 {
  let step = select(vec3i(-1), vec3i(1), dir >= vec3f(0.0));
  let step_pos = select(vec3i(0), vec3i(1), dir >= vec3f(0.0));
  let safe = select(dir, vec3f(step) * 1e-30, abs(dir) < vec3f(1e-30));
  let inv = 1.0 / safe;
  let bs = brickSize();
  let o = cam.origin_cell;
  let f = cam.origin_frac;
  if (any(brickmap.bounds_min >= brickmap.bounds_max)) {
    return cam.far; // nothing loaded
  }

  // Outside the occupied box everything is empty: skip to where the cone could first
  // touch it (box grown by the cone radius at the plain entry point, plus a brick).
  let bmin = vec3f(brickmap.bounds_min * bs - o) - f;
  let bmax = vec3f(brickmap.bounds_max * bs - o) - f;
  let t_plain = max(max(max(min(bmin.x * inv.x, bmax.x * inv.x), min(bmin.y * inv.y, bmax.y * inv.y)), min(bmin.z * inv.z, bmax.z * inv.z)), 0.0);
  let grow = cam.cone_k * t_plain + f32(bs);
  let gmin = (bmin - grow) * inv;
  let gmax = (bmax + grow) * inv;
  let t_lo = min(gmin, gmax);
  let t_hi = max(gmin, gmax);
  var t = max(max(t_lo.x, t_lo.y), max(t_lo.z, 0.0));
  let t_exit = min(min(t_hi.x, t_hi.y), min(t_hi.z, cam.far));
  if (t >= t_exit) {
    return cam.far; // the whole cone misses the occupied box
  }

  for (var i = 0u; i < cam.prepass_max_steps; i++) {
    if (t >= t_exit) {
      return t;
    }
    // Position relative to the ray origin (the same frame as the boxes below).
    let p = dir * t;
    let b = brickCoord(o + vec3i(floor(f + p)));
    let ptr = brickPointer(b);
    if (!brickIsEmpty(ptr)) {
      return t;
    }
    let d = max(i32(ptr & BRICK_DISTANCE_MASK), 1);
    let lo = b - vec3i(d - 1);
    let n = 2 * d - 1;
    // Known-empty box, shrunk by the cone radius where the ray leaves it (the largest
    // radius along this step, so conservative for every point before).
    let t_box = boundaryT(lo * bs, n * bs, step_pos, o, f, inv);
    let r = cam.cone_k * min(min(t_box.x, t_box.y), t_box.z);
    let box_lo = vec3f(lo * bs - o) - f + r;
    let box_hi = vec3f((lo + vec3i(n)) * bs - o) - f - r;
    if (any(p < box_lo) || any(p > box_hi)) {
      return t; // the cone already pokes out of the known-empty box
    }
    // Where the ray reaches the shrunk box's far faces.
    let t_s = select(box_lo, box_hi, step > vec3i(0)) * inv;
    let t_next = min(min(t_s.x, t_s.y), t_s.z);
    if (t_next <= t + 1e-3) {
      return t; // no progress left
    }
    t = t_next;
  }
  return t;
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(coarse);
  if (any(gid.xy >= dims)) {
    return;
  }
  // Centre of the tile, in full-resolution UV.
  let center = (vec2f(gid.xy) + 0.5) * f32(cam.prepass_tile) / vec2f(cam.size);
  let t = coarseDistance(rayDir(cam.inv_view_proj, center));
  textureStore(coarse, vec2i(gid.xy), vec4f(max(t - cam.prepass_safety, 0.0), 0.0, 0.0, 0.0));
}
