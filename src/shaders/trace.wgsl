// Hierarchical voxel ray traversal over the brickmap (Amanatides & Woo at two levels):
// empty bricks are crossed in a single step, occupied bricks are walked voxel by voxel.
#include "brickmap.wgsl"

struct TraceResult {
  hit: bool,
  /// Absolute block coordinates of the hit voxel.
  cell: vec3i,
  /// Outward face normal of the hit face (zero if the ray started inside a block).
  normal: vec3i,
  /// Distance along `dir` to the hit (or where the trace stopped).
  t: f32,
  id: u32,
  /// Brick steps + voxel steps taken.
  steps: u32,
};

/// Axis with the smallest value; ties resolve x, then y, then z (same as raycast.ts).
fn minAxis(v: vec3f) -> u32 {
  if (v.x < v.y) {
    if (v.x < v.z) {
      return 0u;
    }
    return 2u;
  }
  if (v.y < v.z) {
    return 1u;
  }
  return 2u;
}

/// Distance along the ray to the far boundary plane of `cell` (edge length `size`) on
/// each axis. Used for bricks and voxels alike so both levels agree exactly on exits.
fn boundaryT(cell_min: vec3i, size: i32, step_pos: vec3i, origin_cell: vec3i, origin_frac: vec3f, inv: vec3f) -> vec3f {
  return (vec3f(cell_min + step_pos * size - origin_cell) - origin_frac) * inv;
}

/// Traces from origin_cell + origin_frac (frac in [0, 1)) along the normalised `dir`.
/// Positions are kept relative to origin_cell, so precision does not degrade far from 0.
fn traceRay(origin_cell: vec3i, origin_frac: vec3f, dir: vec3f, max_t: f32, max_steps: u32) -> TraceResult {
  var r: TraceResult;
  r.hit = false;
  r.t = max_t;
  r.steps = 0u;

  let step = select(vec3i(-1), vec3i(1), dir >= vec3f(0.0));
  let step_pos = select(vec3i(0), vec3i(1), dir >= vec3f(0.0));
  // Avoid division by zero (indeterminate in WGSL): a tiny component gives a huge t.
  let safe = select(dir, vec3f(step) * 1e-30, abs(dir) < vec3f(1e-30));
  let inv = 1.0 / safe;
  let bs = brickSize();
  let size = vec3i(brickmap.size);

  var b = brickCoord(origin_cell);
  var t_brick = boundaryT(b * bs, bs, step_pos, origin_cell, origin_frac, inv);
  var t = 0.0;
  var normal = vec3i(0);

  loop {
    if (r.steps >= max_steps || t > max_t) {
      break;
    }
    // Outside the resident window and moving further away: nothing more to hit.
    let rel = b - brickmap.origin;
    let below = (rel < vec3i(0)) & (step < vec3i(0));
    let above = (rel >= size) & (step > vec3i(0));
    if (any(below | above)) {
      break;
    }
    r.steps += 1u;

    let ptr = brickPointer(b);
    if (ptr != BRICK_EMPTY) {
      let bmin = b * bs;
      // Entry voxel: the ray position at t, clamped into this brick against rounding.
      var c = clamp(origin_cell + vec3i(floor(origin_frac + dir * t)), bmin, bmin + vec3i(bs - 1));
      if ((ptr & BRICK_UNIFORM_FLAG) != 0u) {
        r.hit = true;
        r.cell = c;
        r.normal = normal;
        r.t = t;
        r.id = ptr & BRICK_UNIFORM_ID_MASK;
        return r;
      }
      var vt = t;
      var vn = normal;
      loop {
        let id = brickVoxel(ptr, vec3u(c - bmin));
        if (id != 0u) {
          r.hit = true;
          r.cell = c;
          r.normal = vn;
          r.t = vt;
          r.id = id;
          return r;
        }
        let t_voxel = boundaryT(c, 1, step_pos, origin_cell, origin_frac, inv);
        let a = minAxis(t_voxel);
        vt = t_voxel[a];
        c[a] += step[a];
        vn = vec3i(0);
        vn[a] = -step[a];
        r.steps += 1u;
        if (c[a] < bmin[a] || c[a] >= bmin[a] + bs || r.steps >= max_steps || vt > max_t) {
          break;
        }
      }
    }

    let a = minAxis(t_brick);
    t = t_brick[a];
    b[a] += step[a];
    t_brick = boundaryT(b * bs, bs, step_pos, origin_cell, origin_frac, inv);
    normal = vec3i(0);
    normal[a] = -step[a];
  }
  return r;
}
