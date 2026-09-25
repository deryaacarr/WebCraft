import { BlockId } from './blocks';
import type { Chunk } from './chunk';
import { CHUNK_MASK, chunkKey, inWorldY, toChunkCoord, toLocalCoord } from './coords';

const NEIGHBOR_OFFSETS = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
] as const;

/**
 * Loaded chunks keyed by "x,y,z", plus tracking of chunks whose contents changed
 * since the consumer (GPU upload, meshing) last called `takeDirty()`.
 */
export class World {
  private readonly chunks = new Map<string, Chunk>();
  private readonly dirty = new Set<string>();
  // Consecutive block lookups usually hit the same chunk; skip the key string then.
  private cached: Chunk | null = null;

  get chunkCount(): number {
    return this.chunks.size;
  }

  getChunk(cx: number, cy: number, cz: number): Chunk | undefined {
    const c = this.cached;
    if (c && c.cx === cx && c.cy === cy && c.cz === cz) return c;
    const chunk = this.chunks.get(chunkKey(cx, cy, cz));
    if (chunk) this.cached = chunk;
    return chunk;
  }

  hasChunk(cx: number, cy: number, cz: number): boolean {
    return this.getChunk(cx, cy, cz) !== undefined;
  }

  chunkValues(): IterableIterator<Chunk> {
    return this.chunks.values();
  }

  /** Inserts (or replaces) a chunk. It and its loaded neighbours become dirty. */
  addChunk(chunk: Chunk): void {
    const prev = this.chunks.get(chunk.key);
    if (prev && this.cached === prev) this.cached = null;
    this.chunks.set(chunk.key, chunk);
    this.dirty.add(chunk.key);
    this.markNeighborsDirty(chunk.cx, chunk.cy, chunk.cz);
  }

  removeChunk(cx: number, cy: number, cz: number): Chunk | undefined {
    const key = chunkKey(cx, cy, cz);
    const chunk = this.chunks.get(key);
    if (!chunk) return undefined;
    this.chunks.delete(key);
    this.dirty.delete(key);
    if (this.cached === chunk) this.cached = null;
    this.markNeighborsDirty(cx, cy, cz);
    return chunk;
  }

  /** Removes every chunk (e.g. before regenerating). */
  clear(): void {
    this.chunks.clear();
    this.dirty.clear();
    this.cached = null;
  }

  /** Block at world coordinates; air when the chunk is not loaded or Y is out of bounds. */
  getBlock(x: number, y: number, z: number): number {
    if (!inWorldY(Math.floor(y))) return BlockId.air;
    const chunk = this.getChunk(toChunkCoord(x), toChunkCoord(y), toChunkCoord(z));
    return chunk ? chunk.get(toLocalCoord(x), toLocalCoord(y), toLocalCoord(z)) : BlockId.air;
  }

  /**
   * Sets a block at world coordinates. Returns false if the chunk is not loaded, Y is
   * outside the world bounds, or the block already had that id. Border edits also
   * dirty the touching neighbour.
   */
  setBlock(x: number, y: number, z: number, id: number): boolean {
    if (!inWorldY(Math.floor(y))) return false;
    const cx = toChunkCoord(x);
    const cy = toChunkCoord(y);
    const cz = toChunkCoord(z);
    const chunk = this.getChunk(cx, cy, cz);
    if (!chunk) return false;
    const lx = toLocalCoord(x);
    const ly = toLocalCoord(y);
    const lz = toLocalCoord(z);
    if (!chunk.set(lx, ly, lz, id)) return false;

    this.dirty.add(chunk.key);
    if (lx === 0) this.markDirty(cx - 1, cy, cz);
    if (lx === CHUNK_MASK) this.markDirty(cx + 1, cy, cz);
    if (ly === 0) this.markDirty(cx, cy - 1, cz);
    if (ly === CHUNK_MASK) this.markDirty(cx, cy + 1, cz);
    if (lz === 0) this.markDirty(cx, cy, cz - 1);
    if (lz === CHUNK_MASK) this.markDirty(cx, cy, cz + 1);
    return true;
  }

  isDirty(cx: number, cy: number, cz: number): boolean {
    return this.dirty.has(chunkKey(cx, cy, cz));
  }

  /** Returns the dirty chunks and clears the dirty set. */
  takeDirty(): Chunk[] {
    const out: Chunk[] = [];
    for (const key of this.dirty) {
      const chunk = this.chunks.get(key);
      if (chunk) out.push(chunk);
    }
    this.dirty.clear();
    return out;
  }

  /** Marks a loaded chunk dirty; no-op if it is not loaded. */
  private markDirty(cx: number, cy: number, cz: number): void {
    const key = chunkKey(cx, cy, cz);
    if (this.chunks.has(key)) this.dirty.add(key);
  }

  private markNeighborsDirty(cx: number, cy: number, cz: number): void {
    for (const [dx, dy, dz] of NEIGHBOR_OFFSETS) this.markDirty(cx + dx, cy + dy, cz + dz);
  }
}
