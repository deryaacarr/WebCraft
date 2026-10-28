// Brickmap access, shared by every pass that reads the voxel world.
//
// Include this file and bind the brickmap (GpuBrickmap.bindGroup) at @group(1):
//   binding 0: BrickmapParams (uniform)
//   binding 1: grid of brick pointers (toroidal in X/Z)
//   binding 2: brick pool (8-bit voxels, 4 per u32)
//   binding 3: sub-voxel shape table (src/world/shapes.ts packShapeTable) — tracers only;
//              a uniform buffer (GI and visibility already use 8 storage buffers)
// Pipelines must set the BRICK_BITS override (config.world.brickBits).
//
// Pointer encoding (see src/gpu/brick-pack.ts):
//   0                          empty brick (all air), no skip information
//   BRICK_EMPTY_FLAG | d       empty brick; d (bits 0-7) = Chebyshev distance in bricks to
//                              the nearest non-empty brick (brick-distance.wgsl)
//   BRICK_UNIFORM_FLAG | id    every voxel is block `id`, no pool storage
//   otherwise                  pool slot of a mixed brick

override BRICK_BITS: u32 = 3u;
/// Bit i = block id i (< 32) has a sub-voxel shape (shapes.ts shapedIdMask).
override SHAPED_IDS: u32 = 0u;

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
/// Words 0..255: per block id 0 = full cube, else (first shape << 8) | variant count.
/// Then 16 words per shape: 8³ occupancy bits, bit (y·64 + z·8 + x).
@group(1) @binding(3) var<uniform> shape_table: array<vec4u, SHAPE_TABLE_VEC4>;

const MICRO: i32 = 8;
const SHAPE_HEADER: u32 = 256u;
const SHAPE_WORDS: u32 = 16u;
/// shapes.ts SHAPE_TABLE_WORDS / 4 (header + MAX_SHAPES shapes).
const SHAPE_TABLE_VEC4: u32 = 320u;

fn shapeWord(i: u32) -> u32 {
  return shape_table[i >> 2u][i & 3u];
}

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

// ------------------------------------------------------------------ sub-voxel shapes

/// Shape header of block `id` (0 = full cube); the table is read only for shaped ids.
fn shapeHeader(id: u32) -> u32 {
  if (id >= 32u || ((SHAPED_IDS >> id) & 1u) == 0u) {
    return 0u;
  }
  return shapeWord(id);
}

/// Same as shapes.ts `shapeHash`: picks a cell's shape variant and quarter turn.
fn shapeHash(c: vec3i) -> u32 {
  var h = (u32(c.x) * 0x9e3779b1u) ^ (u32(c.y) * 0x85ebca77u) ^ (u32(c.z) * 0xc2b2ae3du);
  h ^= h >> 15u;
  h *= 0x2c1b3c6du;
  h ^= h >> 12u;
  h *= 0x297a2d39u;
  h ^= h >> 15u;
  return h;
}

/// Shape of the block at `cell`: x = first table word of its variant, y = quarter turns.
fn cellShape(header: u32, cell: vec3i) -> vec2u {
  let h = shapeHash(cell);
  let shape = (header >> 8u) + (h & 0xffffu) % (header & 0xffu);
  return vec2u(SHAPE_HEADER + shape * SHAPE_WORDS, h >> 30u);
}

/// Micro voxel m (0..7 per axis, cell space) of a shape turned `rot` quarter turns about Y.
fn microSolid(shape: vec2u, m: vec3i) -> bool {
  let k = MICRO - 1;
  var x = m.x;
  var z = m.z;
  if (shape.y == 1u) {
    x = m.z;
    z = k - m.x;
  } else if (shape.y == 2u) {
    x = k - m.x;
    z = k - m.z;
  } else if (shape.y == 3u) {
    x = k - m.z;
    z = m.x;
  }
  let i = u32((m.y * MICRO + z) * MICRO + x);
  return ((shapeWord(shape.x + (i >> 5u)) >> (i & 31u)) & 1u) != 0u;
}
