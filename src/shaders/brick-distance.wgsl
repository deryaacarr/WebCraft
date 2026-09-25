// Chebyshev distance field over the brick grid, computed in three separable passes
// (1-D min filters along X, then Y, then Z) over a region of cells. The result is stored
// in the grid pointers of empty bricks: BRICK_EMPTY_FLAG | distance (1 … max_distance + 1,
// where max_distance + 1 means "at least that far"). Only regions around changed bricks
// are recomputed; the intermediate X and XY distances persist between updates.

const EMPTY_FLAG: u32 = 0x40000000u;
const UNIFORM_FLAG: u32 = 0x80000000u;

struct Region {
  /// First cell (grid cell coords; X/Z may be negative or ≥ size and wrap).
  origin: vec3i,
  max_distance: u32,
  extent: vec3u,
  _pad0: u32,
  /// Grid size in cells (X and Z powers of two).
  size: vec3u,
  _pad1: u32,
};

override WORKGROUP_SIZE: u32 = 4u;

@group(0) @binding(0) var<uniform> region: Region;
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> dist_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> dist_xy: array<u32>;

/// Linear index of a cell (X/Z wrap toroidally); -1 when Y is outside the grid.
fn cellIndex(c: vec3i) -> i32 {
  if (c.y < 0 || c.y >= i32(region.size.y)) {
    return -1;
  }
  let x = u32(c.x) & (region.size.x - 1u);
  let z = u32(c.z) & (region.size.z - 1u);
  return i32((u32(c.y) * region.size.z + z) * region.size.x + x);
}

fn occupied(ptr: u32) -> bool {
  return ptr != 0u && (ptr & EMPTY_FLAG) == 0u;
}

fn regionCell(gid: vec3u) -> vec3i {
  return region.origin + vec3i(gid);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn pass_x(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= region.extent)) {
    return;
  }
  let c = regionCell(gid);
  let i = cellIndex(c);
  if (i < 0) {
    return;
  }
  let far = i32(region.max_distance) + 1;
  var best = far;
  for (var k = -far + 1; k < far; k++) {
    if (occupied(grid[cellIndex(c + vec3i(k, 0, 0))])) {
      best = min(best, abs(k));
    }
  }
  dist_x[i] = u32(best);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn pass_y(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= region.extent)) {
    return;
  }
  let c = regionCell(gid);
  let i = cellIndex(c);
  if (i < 0) {
    return;
  }
  let far = i32(region.max_distance) + 1;
  var best = far;
  for (var k = -far + 1; k < far; k++) {
    let n = cellIndex(c + vec3i(0, k, 0));
    if (n >= 0) {
      best = min(best, max(abs(k), i32(dist_x[n])));
    }
  }
  dist_xy[i] = u32(best);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn pass_z(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= region.extent)) {
    return;
  }
  let c = regionCell(gid);
  let i = cellIndex(c);
  if (i < 0) {
    return;
  }
  let ptr = grid[i];
  if (occupied(ptr)) {
    return;
  }
  let far = i32(region.max_distance) + 1;
  var best = far;
  for (var k = -far + 1; k < far; k++) {
    best = min(best, max(abs(k), i32(dist_xy[cellIndex(c + vec3i(0, 0, k))])));
  }
  grid[i] = EMPTY_FLAG | u32(best);
}
