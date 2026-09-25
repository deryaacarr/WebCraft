import { BlockId } from '../world/blocks';
import type { Chunk } from '../world/chunk';
import {
  BRICK_BITS,
  BRICK_SIZE,
  BRICK_VOLUME,
  BRICKS_PER_AXIS,
  CHUNK_BITS,
} from '../world/coords';

/**
 * Brick pointer encoding (one u32 per grid cell):
 *   0                        → empty brick (all air), no skip information
 *   EMPTY_FLAG | d           → empty brick; d (bits 0-7) is the Chebyshev distance in bricks
 *                                to the nearest non-empty brick, written by the GPU distance
 *                                pass (brick-distance.wgsl). The CPU only ever writes 0.
 *   UNIFORM_FLAG | id        → every voxel is block `id`; no pool slot (deep stone, lake water)
 *   slot (1 … 2^30 − 1)      → mixed brick stored in pool slot `slot`
 * Slot 0 is reserved so that 0 can mean "empty".
 */
export const EMPTY_BRICK = 0;
export const UNIFORM_FLAG = 0x80000000;
export const EMPTY_FLAG = 0x40000000;
export const UNIFORM_ID_MASK = 0xffff;
export const EMPTY_DISTANCE_MASK = 0xff;

/** Voxels are 8-bit block ids packed 4 per u32 (little-endian byte order). */
export const BRICK_VOXEL_WORDS = BRICK_VOLUME / 4;
export const BRICK_BYTES = BRICK_VOLUME;
/**
 * After its voxels every pool brick stores a 64-bit occupancy mask of its 4×4×4
 * sub-cells (each (BRICK_SIZE/4)³ voxels): bit ((y·4 + z)·4 + x) set = sub-cell not empty.
 * Rays skip empty sub-cells in one step.
 */
export const BRICK_OCCUPANCY_WORDS = 2;
export const SUBCELL_BITS = BRICK_BITS - 2;
/** Words per pool slot (voxels + occupancy mask). */
export const BRICK_STRIDE_WORDS = BRICK_VOXEL_WORDS + BRICK_OCCUPANCY_WORDS;
export const BRICK_STRIDE_BYTES = BRICK_STRIDE_WORDS * 4;
if (SUBCELL_BITS < 0) throw new Error('world.brickBits must be at least 2 (4 sub-cells per axis)');
export const MAX_GPU_BLOCK_ID = 0xff;

/** Result of `packBrick` for a brick that needs no pool slot. */
export const MIXED = -1;

export function isSlot(pointer: number): boolean {
  return pointer !== EMPTY_BRICK && (pointer & (UNIFORM_FLAG | EMPTY_FLAG)) === 0;
}

export function isEmptyPointer(pointer: number): boolean {
  return pointer === EMPTY_BRICK || (pointer & EMPTY_FLAG) !== 0;
}

export function uniformPointer(id: number): number {
  return id === BlockId.air ? EMPTY_BRICK : (UNIFORM_FLAG | id) >>> 0;
}

/**
 * Analyses brick `brick` of `chunk`. Returns the block id if the brick is uniform
 * (0 = empty), otherwise writes its voxels as bytes to `out` at `byteOffset`, its sub-cell
 * occupancy mask to `mask[maskOffset..+1]`, and returns MIXED.
 */
export function packBrick(
  chunk: Chunk,
  brick: number,
  out: Uint8Array,
  byteOffset: number,
  mask: Uint32Array,
  maskOffset: number,
): number {
  const data = chunk.denseData;
  if (!data) return chunk.uniformBlock ?? BlockId.air;

  const bx = (brick % BRICKS_PER_AXIS) << BRICK_BITS;
  const bz = (Math.floor(brick / BRICKS_PER_AXIS) % BRICKS_PER_AXIS) << BRICK_BITS;
  const by = Math.floor(brick / (BRICKS_PER_AXIS * BRICKS_PER_AXIS)) << BRICK_BITS;
  const first = data[(by << (2 * CHUNK_BITS)) | (bz << CHUNK_BITS) | bx] ?? BlockId.air;
  let uniform = true;
  let o = byteOffset;
  let lo = 0;
  let hi = 0;
  for (let y = 0; y < BRICK_SIZE; y++) {
    for (let z = 0; z < BRICK_SIZE; z++) {
      // One brick row is contiguous in the chunk (x fastest in both layouts).
      const row = ((by + y) << (2 * CHUNK_BITS)) | ((bz + z) << CHUNK_BITS) | bx;
      const subRow = ((y >> SUBCELL_BITS) * 4 + (z >> SUBCELL_BITS)) * 4;
      for (let x = 0; x < BRICK_SIZE; x++) {
        const id = data[row + x]!;
        if (id !== first) uniform = false;
        if (id > MAX_GPU_BLOCK_ID) throw new Error(`Block id ${id} does not fit the 8-bit GPU brick format`);
        out[o++] = id;
        if (id !== BlockId.air) {
          const bit = subRow + (x >> SUBCELL_BITS);
          if (bit < 32) lo |= 1 << bit;
          else hi |= 1 << (bit - 32);
        }
      }
    }
  }
  mask[maskOffset] = lo >>> 0;
  mask[maskOffset + 1] = hi >>> 0;
  return uniform ? first : MIXED;
}
