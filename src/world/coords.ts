import { config } from '../config';

export const CHUNK_BITS = config.world.chunkBits;
export const CHUNK_SIZE = 1 << CHUNK_BITS;
export const CHUNK_MASK = CHUNK_SIZE - 1;
export const CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

// GPU bricks: the chunk is split into BRICKS_PER_AXIS³ bricks of BRICK_SIZE³ voxels.
export const BRICK_BITS = config.world.brickBits;
export const BRICK_SIZE = 1 << BRICK_BITS;
export const BRICK_MASK = BRICK_SIZE - 1;
export const BRICK_VOLUME = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
export const BRICKS_PER_AXIS = CHUNK_SIZE >> BRICK_BITS;
export const BRICKS_PER_CHUNK = BRICKS_PER_AXIS * BRICKS_PER_AXIS * BRICKS_PER_AXIS;
if (BRICK_BITS > CHUNK_BITS) throw new Error('world.brickBits must not exceed world.chunkBits');

/** Brick index inside a chunk for local block coordinates. Layout: x fastest, then z, then y. */
export function brickIndexInChunk(lx: number, ly: number, lz: number): number {
  return ((ly >> BRICK_BITS) * BRICKS_PER_AXIS + (lz >> BRICK_BITS)) * BRICKS_PER_AXIS + (lx >> BRICK_BITS);
}

/** Voxel index inside a brick for coordinates local to the brick. Layout: x fastest, then z, then y. */
export function voxelIndexInBrick(x: number, y: number, z: number): number {
  return (y << (2 * BRICK_BITS)) | (z << BRICK_BITS) | x;
}

// Block coordinates are int32: bit ops floor correctly for negatives
// (-1 >> 5 === -1, -1 & 31 === 31), so no branch is needed.

/** Chunk coordinate containing world coordinate `v` (block or fractional position). */
export function toChunkCoord(v: number): number {
  return Math.floor(v) >> CHUNK_BITS;
}

/** Position of world coordinate `v` inside its chunk, in [0, CHUNK_SIZE). */
export function toLocalCoord(v: number): number {
  return Math.floor(v) & CHUNK_MASK;
}

/** World coordinate of a chunk's minimum corner. */
export function chunkOrigin(c: number): number {
  return c << CHUNK_BITS;
}

/** Index into chunk data. Layout: x fastest, then z, then y. */
export function localIndex(lx: number, ly: number, lz: number): number {
  return (ly << (2 * CHUNK_BITS)) | (lz << CHUNK_BITS) | lx;
}

/** Chunk Y range [min, max] (inclusive) overlapping the world's vertical bounds. */
export function worldChunkYRange(): [number, number] {
  const { minY, maxY } = config.world;
  if (maxY <= minY) throw new Error(`world.maxY (${maxY}) must be above world.minY (${minY})`);
  return [toChunkCoord(minY), toChunkCoord(maxY - 1)];
}

/** True if block Y lies inside the world's vertical bounds. */
export function inWorldY(y: number): boolean {
  return y >= config.world.minY && y < config.world.maxY;
}

export function chunkKey(cx: number, cy: number, cz: number): string {
  return `${cx},${cy},${cz}`;
}

export function parseChunkKey(key: string): [number, number, number] {
  const [x, y, z] = key.split(',').map(Number);
  if (x === undefined || y === undefined || z === undefined || [x, y, z].some(Number.isNaN)) {
    throw new Error(`Invalid chunk key "${key}"`);
  }
  return [x, y, z];
}
