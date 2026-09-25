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

/// Axis with the smallest value. Tie order is that of raycast.ts:
/// x only if strictly below y and z; otherwise y if strictly below z; otherwise z.
fn minAxis(v: vec3f) -> u32 {
  return select(select(2u, 1u, v.y < v.z), select(2u, 0u, v.x < v.z), v.x < v.y);
}

/// Distance along the ray to the far boundary plane of `cell` (edge length `size`) on
/// each axis. Used for bricks and voxels alike so both levels agree exactly on exits.
fn boundaryT(cell_min: vec3i, size: i32, step_pos: vec3i, origin_cell: vec3i, origin_frac: vec3f, inv: vec3f) -> vec3f {
  return (vec3f(cell_min + step_pos * size - origin_cell) - origin_frac) * inv;
}

// Every shader that includes this file defines
//   fn traceOpaque(id: u32, cell: vec3i, normal: vec3i, local: vec3f, t: f32, dir: vec3f) -> bool
// which decides whether a non-air voxel stops the ray at this point (alpha-tested leaves
// let it through where transparent). Passes without materials simply return true.

/// Traces from origin_cell + origin_frac (frac in [0, 1)) along the normalised `dir`.
/// Positions are kept relative to origin_cell, so precision does not degrade far from 0.
/// `t_min`: the ray is known to hit nothing before this distance (depth prepass); 0 if unknown.
fn traceRay(origin_cell: vec3i, origin_frac: vec3f, dir: vec3f, t_min: f32, max_t: f32, max_steps: u32) -> TraceResult {
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

  // Clip to the box around all non-empty bricks: nothing can be hit outside it.
  let box_min = vec3f(brickmap.bounds_min * bs - origin_cell) - origin_frac;
  let box_max = vec3f(brickmap.bounds_max * bs - origin_cell) - origin_frac;
  let t_lo = min(box_min * inv, box_max * inv);
  let t_hi = max(box_min * inv, box_max * inv);
  let t_enter = max(max(t_lo.x, t_lo.y), max(t_lo.z, 0.0));
  let t_exit = min(min(t_hi.x, t_hi.y), min(t_hi.z, max_t));
  if (t_enter >= t_exit || any(brickmap.bounds_min >= brickmap.bounds_max)) {
    return r;
  }

  var t = max(t_enter, t_min);
  if (t >= t_exit) {
    return r;
  }
  var normal = vec3i(0);
  if (t_enter > 0.0 && t_enter >= t_min) {
    // Entered through a box face: that face's axis is the last one crossed.
    let a = select(select(2u, 1u, t_lo.y >= t_lo.z), select(2u, 0u, t_lo.x >= t_lo.z), t_lo.x >= t_lo.y);
    normal[a] = -step[a];
  }

  // One flat loop instead of nested brick/voxel loops: every iteration is either a
  // brick-level step or a voxel step, which keeps SIMD lanes in the same loop.
  let box_lo_cell = brickmap.bounds_min * bs;
  let box_hi_cell = brickmap.bounds_max * bs - vec3i(1);
  var c = clamp(origin_cell + vec3i(floor(origin_frac + dir * t)), box_lo_cell, box_hi_cell);
  var b = brickCoord(c);
  var in_brick = false;
  var bmin = vec3i(0);
  var base = 0u;
  var occ = vec2u(0u);
  /// Non-zero while walking a uniform brick voxel by voxel (a see-through uniform brick).
  var uniform_id = 0u;
  // Sub-cells: 4 per brick axis, each sub_size³ voxels.
  let sub_bits = BRICK_BITS - 2u;
  let sub_size = 1 << sub_bits;

  loop {
    if (r.steps >= max_steps) {
      break;
    }
    r.steps += 1u;

    if (!in_brick) {
      let ptr = brickPointerInBox(b);
      if (brickIsEmpty(ptr)) {
        // Distance field: every brick within Chebyshev distance d − 1 is empty, so the box
        // [lo, lo + n) of bricks can be left in one step (d = 0, not yet computed, acts as 1).
        let d = max(i32(ptr & BRICK_DISTANCE_MASK), 1);
        let lo = b - vec3i(d - 1);
        let n = 2 * d - 1;
        let t_box = boundaryT(lo * bs, n * bs, step_pos, origin_cell, origin_frac, inv);
        let a = minAxis(t_box);
        t = t_box[a];
        if (t >= t_exit) {
          break;
        }
        // Entry voxel of the next brick: exact on the exit axis, from the position on the others.
        let hi = (lo + vec3i(n)) * bs - vec3i(1);
        c = clamp(origin_cell + vec3i(floor(origin_frac + dir * t)), lo * bs, hi);
        c[a] = select(lo[a] * bs - 1, hi[a] + 1, step[a] > 0);
        b = brickCoord(c);
        normal = vec3i(0);
        normal[a] = -step[a];
        continue;
      }
      in_brick = true;
      bmin = b * bs;
      if ((ptr & BRICK_UNIFORM_FLAG) != 0u) {
        // Uniform brick: the entry voxel is the hit, unless it is see-through there; then
        // walk it voxel by voxel like a fully occupied mixed brick.
        let id = ptr & BRICK_UNIFORM_ID_MASK;
        if (traceOpaque(id, c, normal, origin_frac + dir * t - vec3f(c - origin_cell), t, dir)) {
          r.hit = true;
          r.cell = c;
          r.normal = normal;
          r.t = t;
          r.id = id;
          return r;
        }
        uniform_id = id;
        base = 0u;
        occ = vec2u(0xffffffffu);
      } else {
        // Mixed brick: enter it and take the first voxel / sub-cell step right away.
        uniform_id = 0u;
        base = ptr * brickStride();
        occ = vec2u(brick_pool[base + brickVoxelWords()], brick_pool[base + brickVoxelWords() + 1u]);
      }
    }

    let i = vec3u(c - bmin);
    let sub = i >> vec3u(sub_bits);
    let s = (sub.y << 4u) | (sub.z << 2u) | sub.x;
    var a = 0u;
    if (((select(occ.x, occ.y, s >= 32u) >> (s & 31u)) & 1u) == 0u) {
      // Empty sub-cell (the brick's occupancy mask): leave it in one step.
      let sub_min = bmin + vec3i(sub << vec3u(sub_bits));
      let sub_hi = sub_min + vec3i(sub_size - 1);
      let t_sub = boundaryT(sub_min, sub_size, step_pos, origin_cell, origin_frac, inv);
      a = minAxis(t_sub);
      t = t_sub[a];
      c = clamp(origin_cell + vec3i(floor(origin_frac + dir * t)), sub_min, sub_hi);
      c[a] = select(sub_min[a] - 1, sub_hi[a] + 1, step[a] > 0);
    } else {
      // Voxel step: one load per voxel.
      let v = (i.y << (2u * BRICK_BITS)) | (i.z << BRICK_BITS) | i.x;
      let id = select((brick_pool[base + (v >> 2u)] >> ((v & 3u) * 8u)) & 0xffu, uniform_id, uniform_id != 0u);
      if (id != 0u && traceOpaque(id, c, normal, origin_frac + dir * t - vec3f(c - origin_cell), t, dir)) {
        r.hit = true;
        r.cell = c;
        r.normal = normal;
        r.t = t;
        r.id = id;
        return r;
      }
      let t_voxel = boundaryT(c, 1, step_pos, origin_cell, origin_frac, inv);
      a = minAxis(t_voxel);
      t = t_voxel[a];
      c[a] += step[a];
    }
    normal = vec3i(0);
    normal[a] = -step[a];
    if (c[a] < bmin[a] || c[a] >= bmin[a] + bs) {
      // Left through the brick face on axis a: the neighbour brick, entered at voxel c.
      in_brick = false;
      b[a] += step[a];
    }
    if (t >= t_exit) {
      break;
    }
  }
  return r;
}
