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
 *   0                     → empty brick (all air); rays skip it in one step
 *   UNIFORM_FLAG | id     → every voxel is block `id`; no pool slot (deep stone, lake water)
 *   slot (1 … 2^31 − 1)   → mixed brick stored in pool slot `slot`
 * Slot 0 is reserved so that 0 can mean "empty".
 */
export const EMPTY_BRICK = 0;
export const UNIFORM_FLAG = 0x80000000;
export const UNIFORM_ID_MASK = 0xffff;

/** Voxels are 8-bit block ids packed 4 per u32 (little-endian byte order). */
export const BRICK_WORDS = BRICK_VOLUME / 4;
export const BRICK_BYTES = BRICK_VOLUME;
export const MAX_GPU_BLOCK_ID = 0xff;

/** Result of `packBrick` for a brick that needs no pool slot. */
export const MIXED = -1;

export function isSlot(pointer: number): boolean {
  return pointer !== EMPTY_BRICK && (pointer & UNIFORM_FLAG) === 0;
}

export function uniformPointer(id: number): number {
  return id === BlockId.air ? EMPTY_BRICK : (UNIFORM_FLAG | id) >>> 0;
}

/**
 * Analyses brick `brick` of `chunk`. Returns the block id if the brick is uniform
 * (0 = empty), otherwise writes its voxels as bytes to `out` at `byteOffset` and
 * returns MIXED.
 */
export function packBrick(chunk: Chunk, brick: number, out: Uint8Array, byteOffset: number): number {
  const data = chunk.denseData;
  if (!data) return chunk.uniformBlock ?? BlockId.air;

  const bx = (brick % BRICKS_PER_AXIS) << BRICK_BITS;
  const bz = (Math.floor(brick / BRICKS_PER_AXIS) % BRICKS_PER_AXIS) << BRICK_BITS;
  const by = Math.floor(brick / (BRICKS_PER_AXIS * BRICKS_PER_AXIS)) << BRICK_BITS;
  const first = data[(by << (2 * CHUNK_BITS)) | (bz << CHUNK_BITS) | bx] ?? BlockId.air;
  let uniform = true;
  let o = byteOffset;
  for (let y = 0; y < BRICK_SIZE; y++) {
    for (let z = 0; z < BRICK_SIZE; z++) {
      // One brick row is contiguous in the chunk (x fastest in both layouts).
      const row = ((by + y) << (2 * CHUNK_BITS)) | ((bz + z) << CHUNK_BITS) | bx;
      for (let x = 0; x < BRICK_SIZE; x++) {
        const id = data[row + x]!;
        if (id !== first) uniform = false;
        if (id > MAX_GPU_BLOCK_ID) throw new Error(`Block id ${id} does not fit the 8-bit GPU brick format`);
        out[o++] = id;
      }
    }
  }
  return uniform ? first : MIXED;
}
