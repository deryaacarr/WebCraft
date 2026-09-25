// Brickmap access, shared by every pass that reads the voxel world.
//
// Include this file and bind the brickmap (GpuBrickmap.bindGroup) at @group(1):
//   binding 0: BrickmapParams (uniform)
//   binding 1: grid of brick pointers (toroidal in X/Z)
//   binding 2: brick pool (8-bit voxels, 4 per u32)
// Pipelines must set the BRICK_BITS override (config.world.brickBits).
//
// Pointer encoding (see src/gpu/brick-pack.ts):
//   0                          empty brick (all air) — skip it in one step
//   BRICK_UNIFORM_FLAG | id    every voxel is block `id`, no pool storage
//   otherwise                  pool slot of a mixed brick

override BRICK_BITS: u32 = 3u;

const BRICK_EMPTY: u32 = 0u;
const BRICK_UNIFORM_FLAG: u32 = 0x80000000u;
const BRICK_UNIFORM_ID_MASK: u32 = 0xffffu;

struct BrickmapParams {
  /// Minimum corner of the resident window, in brick coordinates.
  origin: vec3i,
  _pad0: i32,
  /// Window / grid size in bricks (X and Z are powers of two).
  size: vec3u,
  /// Brick Y of the grid's bottom layer.
  min_brick_y: i32,
};

@group(1) @binding(0) var<uniform> brickmap: BrickmapParams;
@group(1) @binding(1) var<storage, read> brick_grid: array<u32>;
@group(1) @binding(2) var<storage, read> brick_pool: array<u32>;

fn brickSize() -> i32 {
  return 1 << BRICK_BITS;
}

fn brickWords() -> u32 {
  return (1u << (3u * BRICK_BITS)) / 4u;
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

/// Occupancy query: false means the whole brick is air and a ray can skip it.
fn brickOccupied(b: vec3i) -> bool {
  return brickPointer(b) != BRICK_EMPTY;
}

/// Block id at `local` (0..brickSize-1 per axis) inside the brick behind `ptr`.
fn brickVoxel(ptr: u32, local: vec3u) -> u32 {
  if (ptr == BRICK_EMPTY) {
    return 0u;
  }
  if ((ptr & BRICK_UNIFORM_FLAG) != 0u) {
    return ptr & BRICK_UNIFORM_ID_MASK;
  }
  let i = (local.y << (2u * BRICK_BITS)) | (local.z << BRICK_BITS) | local.x;
  let word = brick_pool[ptr * brickWords() + (i >> 2u)];
  return (word >> ((i & 3u) * 8u)) & 0xffu;
}

/// Block id at world voxel `p` (0 = air, also outside the resident window).
fn getVoxel(p: vec3i) -> u32 {
  let local = vec3u(p & vec3i(brickSize() - 1));
  return brickVoxel(brickPointer(brickCoord(p)), local);
}
