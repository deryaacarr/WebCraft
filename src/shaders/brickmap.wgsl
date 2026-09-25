// Brickmap access, shared by every pass that reads the voxel world.
//
// Include this file and bind the brickmap (GpuBrickmap.bindGroup) at @group(1):
//   binding 0: BrickmapParams (uniform)
//   binding 1: grid of brick pointers (toroidal in X/Z)
//   binding 2: brick pool (8-bit voxels, 4 per u32)
// Pipelines must set the BRICK_BITS override (config.world.brickBits).
//
// Pointer encoding (see src/gpu/brick-pack.ts):
//   0                          empty brick (all air), no skip information
//   BRICK_EMPTY_FLAG | d       empty brick; d (bits 0-7) = Chebyshev distance in bricks to
//                              the nearest non-empty brick (brick-distance.wgsl)
//   BRICK_UNIFORM_FLAG | id    every voxel is block `id`, no pool storage
//   otherwise                  pool slot of a mixed brick

override BRICK_BITS: u32 = 3u;

const BRICK_EMPTY: u32 = 0u;
const BRICK_UNIFORM_FLAG: u32 = 0x80000000u;
const BRICK_EMPTY_FLAG: u32 = 0x40000000u;
const BRICK_UNIFORM_ID_MASK: u32 = 0xffffu;
const BRICK_DISTANCE_MASK: u32 = 0xffu;

struct BrickmapParams {
  /// Minimum corner of the resident window, in brick coordinates.
  origin: vec3i,
  _pad0: i32,
  /// Window / grid size in bricks (X and Z are powers of two).
  size: vec3u,
  /// Brick Y of the grid's bottom layer.
  min_brick_y: i32,
  /// Box around every non-empty brick (brick coords, max exclusive); min == max if empty.
  /// Its top is the highest occupied layer, so rays above the terrain stop at once.
  bounds_min: vec3i,
  _pad1: i32,
  bounds_max: vec3i,
  _pad2: i32,
};

@group(1) @binding(0) var<uniform> brickmap: BrickmapParams;
@group(1) @binding(1) var<storage, read> brick_grid: array<u32>;
@group(1) @binding(2) var<storage, read> brick_pool: array<u32>;

fn brickSize() -> i32 {
  return 1 << BRICK_BITS;
}

/// u32 words of voxel data per pool slot (8-bit voxels, 4 per word).
fn brickVoxelWords() -> u32 {
  return (1u << (3u * BRICK_BITS)) / 4u;
}

/// Words per pool slot: voxels, then a 64-bit occupancy mask of the 4×4×4 sub-cells
/// (bit ((y·4 + z)·4 + x) set = sub-cell holds a solid voxel).
fn brickStride() -> u32 {
  return brickVoxelWords() + 2u;
}

/// Brick containing voxel `p` (arithmetic shift floors negative coordinates).
fn brickCoord(p: vec3i) -> vec3i {
  return p >> vec3u(BRICK_BITS);
}

fn brickInWindow(b: vec3i) -> bool {
  let rel = b - brickmap.origin;
  return all(rel >= vec3i(0)) && all(rel < vec3i(brickmap.size));
}

/// Pointer of brick `b`; bricks outside the resident window read as empty.
fn brickPointer(b: vec3i) -> u32 {
  if (!brickInWindow(b)) {
    return BRICK_EMPTY;
  }
  let s = brickmap.size;
  // u32(i32) reinterprets the bits, so masking gives a correct modulo for negatives.
  let cell = (u32(b.y - brickmap.min_brick_y) * s.z + (u32(b.z) & (s.z - 1u))) * s.x + (u32(b.x) & (s.x - 1u));
  return brick_grid[cell];
}

fn brickIsEmpty(ptr: u32) -> bool {
  return ptr == BRICK_EMPTY || (ptr & BRICK_EMPTY_FLAG) != 0u;
}

/// Pointer of brick `b` known to lie inside the resident window (the tracer only visits
/// bricks inside the occupied box, which the window contains). Skips the X/Z window test.
fn brickPointerInBox(b: vec3i) -> u32 {
  let s = brickmap.size;
  let y = u32(clamp(b.y - brickmap.min_brick_y, 0, i32(s.y) - 1));
  return brick_grid[(y * s.z + (u32(b.z) & (s.z - 1u))) * s.x + (u32(b.x) & (s.x - 1u))];
}

/// Occupancy query: false means the whole brick is air and a ray can skip it.
fn brickOccupied(b: vec3i) -> bool {
  return !brickIsEmpty(brickPointer(b));
}

/// Block id at `local` (0..brickSize-1 per axis) inside the brick behind `ptr`.
fn brickVoxel(ptr: u32, local: vec3u) -> u32 {
  if (brickIsEmpty(ptr)) {
    return 0u;
  }
  if ((ptr & BRICK_UNIFORM_FLAG) != 0u) {
    return ptr & BRICK_UNIFORM_ID_MASK;
  }
  let i = (local.y << (2u * BRICK_BITS)) | (local.z << BRICK_BITS) | local.x;
  let word = brick_pool[ptr * brickStride() + (i >> 2u)];
  return (word >> ((i & 3u) * 8u)) & 0xffu;
}

/// Block id at world voxel `p` (0 = air, also outside the resident window).
fn getVoxel(p: vec3i) -> u32 {
  let local = vec3u(p & vec3i(brickSize() - 1));
  return brickVoxel(brickPointer(brickCoord(p)), local);
}
