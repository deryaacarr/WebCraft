import { BlockId } from './blocks';
import type { Chunk } from './chunk';
import { BRICKS_PER_CHUNK, brickIndexInChunk, chunkKey, inWorldY, toChunkCoord, toLocalCoord } from './coords';

/** A loaded chunk whose contents changed; `bricks[i] === 1` marks brick i as changed. */
export interface ChunkChange {
  chunk: Chunk;
  bricks: Uint8Array;
}

export interface WorldChanges {
  /** Keys of chunks removed since the last take (process these first). */
  removed: string[];
  changed: ChunkChange[];
}

/**
 * Loaded chunks keyed by "x,y,z", plus change tracking for the consumer that mirrors
 * the world (the GPU brickmap). Changes are tracked per brick, so a single block edit
 * re-uploads one brick instead of a whole chunk. A ray tracer reads neighbours directly,
 * so an edit never dirties adjacent chunks.
 */
export class World {
  private readonly chunks = new Map<string, Chunk>();
  private readonly changed = new Map<string, Uint8Array>();
  private readonly removed = new Set<string>();
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

  /** Inserts (or replaces) a chunk; all of its bricks count as changed. */
  addChunk(chunk: Chunk): void {
    const prev = this.chunks.get(chunk.key);
    if (prev && this.cached === prev) this.cached = null;
    this.chunks.set(chunk.key, chunk);
    this.changed.set(chunk.key, new Uint8Array(BRICKS_PER_CHUNK).fill(1));
  }

  removeChunk(cx: number, cy: number, cz: number): Chunk | undefined {
    const key = chunkKey(cx, cy, cz);
    const chunk = this.chunks.get(key);
    if (!chunk) return undefined;
    this.chunks.delete(key);
    this.changed.delete(key);
    this.removed.add(key);
    if (this.cached === chunk) this.cached = null;
    return chunk;
  }

  /** Removes every chunk (e.g. before regenerating). */
  clear(): void {
    for (const key of this.chunks.keys()) this.removed.add(key);
    this.chunks.clear();
    this.changed.clear();
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
   * outside the world bounds, or the block already had that id. Only the brick holding
   * the block is marked as changed.
   */
  setBlock(x: number, y: number, z: number, id: number): boolean {
    if (!inWorldY(Math.floor(y))) return false;
    const chunk = this.getChunk(toChunkCoord(x), toChunkCoord(y), toChunkCoord(z));
    if (!chunk) return false;
    const lx = toLocalCoord(x);
    const ly = toLocalCoord(y);
    const lz = toLocalCoord(z);
    if (!chunk.set(lx, ly, lz, id)) return false;

    let bricks = this.changed.get(chunk.key);
    if (!bricks) {
      bricks = new Uint8Array(BRICKS_PER_CHUNK);
      this.changed.set(chunk.key, bricks);
    }
    bricks[brickIndexInChunk(lx, ly, lz)] = 1;
    return true;
  }

  isDirty(cx: number, cy: number, cz: number): boolean {
    return this.changed.has(chunkKey(cx, cy, cz));
  }

  /** Returns removals and per-brick changes since the last call, then clears them. */
  takeChanges(): WorldChanges {
    const removed = [...this.removed];
    const changed: ChunkChange[] = [];
    for (const [key, bricks] of this.changed) {
      const chunk = this.chunks.get(key);
      if (chunk) changed.push({ chunk, bricks });
    }
    this.removed.clear();
    this.changed.clear();
    return { removed, changed };
  }
}
