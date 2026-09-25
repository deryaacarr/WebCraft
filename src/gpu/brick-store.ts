import type { Chunk } from '../world/chunk';
import { BRICK_SIZE, BRICKS_PER_AXIS, BRICKS_PER_CHUNK, parseChunkKey } from '../world/coords';
import type { World, WorldChanges } from '../world/world';
import {
  BRICK_BYTES,
  BRICK_OCCUPANCY_WORDS,
  BRICK_STRIDE_WORDS,
  BRICK_VOXEL_WORDS,
  EMPTY_BRICK,
  isSlot,
  MIXED,
  packBrick,
  uniformPointer,
} from './brick-pack';

/** Where the store's writes go: GPU buffers in the app, plain arrays in tests. */
export interface BrickSink {
  /** Writes `pointers` into consecutive grid cells starting at `cell`. */
  writeGrid(cell: number, pointers: Uint32Array): void;
  /** Writes whole bricks (BRICK_STRIDE_WORDS each) into consecutive pool slots from `slot`. */
  writePool(slot: number, words: Uint32Array): void;
  /** Grows the pool to `capacity` slots, preserving contents. */
  growPool(capacity: number): void;
  /** The brick window moved; `origin` is its minimum corner in brick coordinates. */
  setOrigin(origin: readonly [number, number, number]): void;
  /**
   * Box (brick coordinates, max exclusive) that contains every non-empty brick; rays are
   * clipped to it. An empty world is reported as min = max.
   */
  setBounds(min: readonly [number, number, number], max: readonly [number, number, number]): void;
}

interface ResidentChunk {
  cx: number;
  cy: number;
  cz: number;
  /** Pointers of its bricks (mirror of its grid cells). */
  pointers: Uint32Array;
  /** Number of non-empty bricks. */
  occupied: number;
}

export interface BrickGridSize {
  /** Grid size in bricks (X and Z are powers of two for toroidal masking). */
  x: number;
  y: number;
  z: number;
  /** Brick Y of the grid's bottom layer (the world's minimum). */
  minBrickY: number;
}

export interface BrickStoreOptions {
  grid: BrickGridSize;
  initialPoolBricks: number;
  poolGrowth: number;
  /** Hard cap on pool slots (from the device's buffer size limits). */
  maxPoolBricks: number;
}

export interface BrickStoreStats {
  residentChunks: number;
  /** Chunks with changes waiting for upload. */
  pendingChunks: number;
  mixedBricks: number;
  uniformBricks: number;
  poolCapacity: number;
  /** Mixed bricks that could not be stored because the pool hit its limit. */
  droppedBricks: number;
  uploadedBytes: number;
}

/**
 * CPU side of the GPU brickmap.
 *
 * Grid: a toroidal window of bricks around the player. A brick at world brick coords
 * (bx, by, bz) always lives in cell (bx mod X, by − minBrickY, bz mod Z), so moving the
 * window only needs a new origin — bricks that stay inside keep their cells, and only
 * chunks leaving the window are evicted. Pool: fixed-size slots managed by a free list.
 */
export class BrickStore {
  private readonly grid: Uint32Array;
  private readonly size: BrickGridSize;
  private readonly resident = new Map<string, ResidentChunk>();
  /** Non-empty bricks per grid layer (Y): gives the lowest / highest occupied layer. */
  private readonly layerCounts: Int32Array;
  private boundsDirty = true;
  private lastBounds = '';
  /** Chunks waiting for upload, with the bricks to (re)build. Insertion order = priority. */
  private readonly pending = new Map<string, Uint8Array>();
  private readonly free: number[] = [];
  private nextSlot = 1;
  private capacity: number;
  private origin: [number, number, number] | null = null;
  private readonly staging: Uint8Array;
  private readonly stagingWords: Uint32Array;
  private readonly stagingMasks = new Uint32Array(BRICKS_PER_CHUNK * BRICK_OCCUPANCY_WORDS);
  private readonly rowPointers = new Uint32Array(BRICKS_PER_AXIS);
  private mixed = 0;
  private uniform = 0;
  private dropped = 0;
  private uploaded = 0;

  constructor(
    private readonly sink: BrickSink,
    private readonly opts: BrickStoreOptions,
  ) {
    const { grid } = opts;
    for (const d of [grid.x, grid.z]) {
      if (d < BRICKS_PER_AXIS || (d & (d - 1)) !== 0) {
        throw new Error(`Brick grid X/Z (${d}) must be a power of two ≥ ${BRICKS_PER_AXIS}`);
      }
    }
    this.size = grid;
    this.grid = new Uint32Array(grid.x * grid.y * grid.z);
    this.layerCounts = new Int32Array(grid.y);
    this.capacity = Math.max(2, opts.initialPoolBricks + 1);
    this.staging = new Uint8Array(BRICKS_PER_CHUNK * BRICK_BYTES);
    this.stagingWords = new Uint32Array(this.staging.buffer);
    sink.growPool(this.capacity);
  }

  get stats(): BrickStoreStats {
    return {
      residentChunks: this.resident.size,
      pendingChunks: this.pending.size,
      mixedBricks: this.mixed,
      uniformBricks: this.uniform,
      poolCapacity: this.capacity - 1,
      droppedBricks: this.dropped,
      uploadedBytes: this.uploaded,
    };
  }

  /** Grid cells in bytes (for memory reporting). */
  get gridBytes(): number {
    return this.grid.byteLength;
  }

  /**
   * Centres the window on a chunk column. Evicts resident chunks that fall outside and
   * queues loaded world chunks that come into view. Returns true if the origin moved.
   */
  setCenter(cx: number, cz: number, world: World): boolean {
    const chunksX = this.size.x / BRICKS_PER_AXIS;
    const chunksZ = this.size.z / BRICKS_PER_AXIS;
    const origin: [number, number, number] = [
      (cx - (chunksX >> 1)) * BRICKS_PER_AXIS,
      this.size.minBrickY,
      (cz - (chunksZ >> 1)) * BRICKS_PER_AXIS,
    ];
    const o = this.origin;
    if (o && o[0] === origin[0] && o[2] === origin[2]) return false;
    this.origin = origin;

    for (const key of [...this.resident.keys()]) {
      const [x, y, z] = parseChunkKey(key);
      if (!this.chunkInWindow(x, y, z)) this.evict(key);
    }
    for (const key of this.pending.keys()) {
      const [x, y, z] = parseChunkKey(key);
      if (!this.chunkInWindow(x, y, z)) this.pending.delete(key);
    }
    for (const chunk of world.chunkValues()) {
      if (!this.resident.has(chunk.key) && !this.pending.has(chunk.key) && this.chunkInWindow(chunk.cx, chunk.cy, chunk.cz)) {
        this.pending.set(chunk.key, new Uint8Array(BRICKS_PER_CHUNK).fill(1));
      }
    }
    this.sink.setOrigin(origin);
    this.publishBounds();
    return true;
  }

  /** Records world changes. Removals are applied immediately (cheap); uploads are queued. */
  applyChanges(changes: WorldChanges): void {
    for (const key of changes.removed) {
      this.pending.delete(key);
      this.evict(key);
    }
    for (const { chunk, bricks } of changes.changed) {
      if (!this.chunkInWindow(chunk.cx, chunk.cy, chunk.cz)) continue;
      const prev = this.pending.get(chunk.key);
      if (prev) {
        for (let i = 0; i < BRICKS_PER_CHUNK; i++) prev[i]! |= bricks[i]!;
      } else {
        this.pending.set(chunk.key, bricks.slice());
      }
    }
    this.publishBounds();
  }

  /** Uploads pending chunks until `budgetMs` is spent (at least one chunk per call). */
  process(world: World, budgetMs: number, now: () => number = () => performance.now()): void {
    const start = now();
    for (const [key, bricks] of this.pending) {
      this.pending.delete(key);
      const [cx, cy, cz] = parseChunkKey(key);
      const chunk = world.getChunk(cx, cy, cz);
      if (chunk) this.upload(chunk, bricks);
      if (now() - start >= budgetMs) break;
    }
    this.publishBounds();
  }

  /** Current ray-clipping box in brick coordinates (max exclusive); null when empty. */
  get bounds(): { min: [number, number, number]; max: [number, number, number] } | null {
    let minX = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxZ = -Infinity;
    for (const c of this.resident.values()) {
      if (c.occupied === 0) continue;
      minX = Math.min(minX, c.cx);
      maxX = Math.max(maxX, c.cx);
      minZ = Math.min(minZ, c.cz);
      maxZ = Math.max(maxZ, c.cz);
    }
    const layers = this.layerCounts;
    let lo = 0;
    while (lo < layers.length && layers[lo] === 0) lo++;
    let hi = layers.length - 1;
    while (hi >= 0 && layers[hi] === 0) hi--;
    if (minX === Infinity || lo > hi) return null;
    const P = BRICKS_PER_AXIS;
    return {
      min: [minX * P, this.size.minBrickY + lo, minZ * P],
      max: [(maxX + 1) * P, this.size.minBrickY + hi + 1, (maxZ + 1) * P],
    };
  }

  /** Uploads everything pending, regardless of time (tests, tools). */
  flush(world: World): void {
    this.process(world, Infinity);
  }

  // ------------------------------------------------------------------ internals

  private chunkInWindow(cx: number, cy: number, cz: number): boolean {
    const o = this.origin;
    if (!o) return false;
    const bx = cx * BRICKS_PER_AXIS;
    const by = cy * BRICKS_PER_AXIS;
    const bz = cz * BRICKS_PER_AXIS;
    return (
      bx >= o[0] && bx + BRICKS_PER_AXIS <= o[0] + this.size.x &&
      bz >= o[2] && bz + BRICKS_PER_AXIS <= o[2] + this.size.z &&
      by + BRICKS_PER_AXIS > o[1] && by < o[1] + this.size.y
    );
  }

  /** Grid cell of a world brick coordinate, or -1 if its Y is outside the grid. */
  private cell(bx: number, by: number, bz: number): number {
    const y = by - this.size.minBrickY;
    if (y < 0 || y >= this.size.y) return -1;
    return (y * this.size.z + (bz & (this.size.z - 1))) * this.size.x + (bx & (this.size.x - 1));
  }

  private publishBounds(): void {
    if (!this.boundsDirty) return;
    this.boundsDirty = false;
    const b = this.bounds;
    const min = b?.min ?? ([0, 0, 0] as [number, number, number]);
    const max = b?.max ?? ([0, 0, 0] as [number, number, number]);
    const key = `${min},${max}`;
    if (key === this.lastBounds) return;
    this.lastBounds = key;
    this.sink.setBounds(min, max);
  }

  private upload(chunk: Chunk, bricks: Uint8Array): void {
    const key = chunk.key;
    let res = this.resident.get(key);
    if (!res) {
      res = { cx: chunk.cx, cy: chunk.cy, cz: chunk.cz, pointers: new Uint32Array(BRICKS_PER_CHUNK), occupied: 0 };
      this.resident.set(key, res);
    }
    const pointers = res.pointers;

    // 1. Classify bricks, pack mixed ones into staging, (re)assign slots.
    const writes: number[] = []; // [slot, stagingIndex] pairs
    const byBase = chunk.cy * BRICKS_PER_AXIS - this.size.minBrickY;
    for (let b = 0; b < BRICKS_PER_CHUNK; b++) {
      if (!bricks[b]) continue;
      const old = pointers[b]!;
      // Bricks above/below the grid (chunk straddling a world bound) have no cell.
      const gy = byBase + Math.floor(b / (BRICKS_PER_AXIS * BRICKS_PER_AXIS));
      const result = gy < 0 || gy >= this.size.y ? 0 : packBrick(chunk, b, this.staging, b * BRICK_BYTES, this.stagingMasks, b * BRICK_OCCUPANCY_WORDS);
      let next: number;
      if (result === MIXED) {
        const slot = isSlot(old) ? old : this.allocSlot();
        if (slot === 0) {
          this.dropped++;
          next = EMPTY_BRICK;
        } else {
          next = slot;
          writes.push(slot, b);
        }
      } else {
        if (isSlot(old)) this.freeSlot(old);
        next = uniformPointer(result);
      }
      this.count(res, old, gy, -1);
      this.count(res, next, gy, +1);
      pointers[b] = next;
    }

    // 2. Pool: coalesce runs of consecutive slots into one write each.
    for (let i = 0; i < writes.length; ) {
      const slot = writes[i]!;
      let run = 1;
      while (i + run * 2 < writes.length && writes[i + run * 2] === slot + run) run++;
      const words = new Uint32Array(run * BRICK_STRIDE_WORDS);
      for (let r = 0; r < run; r++) {
        const b = writes[i + r * 2 + 1]!;
        const at = r * BRICK_STRIDE_WORDS;
        words.set(this.stagingWords.subarray(b * BRICK_VOXEL_WORDS, (b + 1) * BRICK_VOXEL_WORDS), at);
        words.set(this.stagingMasks.subarray(b * BRICK_OCCUPANCY_WORDS, (b + 1) * BRICK_OCCUPANCY_WORDS), at + BRICK_VOXEL_WORDS);
      }
      this.sink.writePool(slot, words);
      this.uploaded += words.byteLength;
      i += run * 2;
    }

    // 3. Grid: each brick row of the chunk (fixed y, z) is contiguous in X.
    this.writeRows(chunk.cx, chunk.cy, chunk.cz, pointers, bricks);
  }

  /** Writes the chunk's grid rows (only rows with a flagged brick unless `bricks` is null). */
  private writeRows(cx: number, cy: number, cz: number, pointers: Uint32Array, bricks: Uint8Array | null): void {
    const P = BRICKS_PER_AXIS;
    const bx0 = cx * P;
    for (let y = 0; y < P; y++) {
      for (let z = 0; z < P; z++) {
        const first = (y * P + z) * P;
        if (bricks && !bricks.subarray(first, first + P).some((f) => f)) continue;
        const cell = this.cell(bx0, cy * P + y, cz * P + z);
        if (cell < 0) continue;
        for (let x = 0; x < P; x++) {
          const p = pointers[first + x]!;
          this.rowPointers[x] = p;
          this.grid[cell + x] = p;
        }
        this.sink.writeGrid(cell, this.rowPointers);
        this.uploaded += this.rowPointers.byteLength;
      }
    }
  }

  private evict(key: string): void {
    const res = this.resident.get(key);
    if (!res) return;
    this.resident.delete(key);
    const { pointers } = res;
    const byBase = res.cy * BRICKS_PER_AXIS - this.size.minBrickY;
    for (let b = 0; b < BRICKS_PER_CHUNK; b++) {
      const p = pointers[b]!;
      if (isSlot(p)) this.freeSlot(p);
      this.count(res, p, byBase + Math.floor(b / (BRICKS_PER_AXIS * BRICKS_PER_AXIS)), -1);
      pointers[b] = EMPTY_BRICK;
    }
    this.writeRows(res.cx, res.cy, res.cz, pointers, null);
  }

  private allocSlot(): number {
    const reused = this.free.pop();
    if (reused !== undefined) return reused;
    if (this.nextSlot >= this.capacity) {
      const limit = this.opts.maxPoolBricks + 1;
      if (this.capacity >= limit) return 0;
      this.capacity = Math.min(limit, Math.ceil(this.capacity * this.opts.poolGrowth));
      this.sink.growPool(this.capacity);
    }
    return this.nextSlot++;
  }

  private freeSlot(slot: number): void {
    this.free.push(slot);
  }

  private count(res: ResidentChunk, pointer: number, gy: number, delta: number): void {
    if (pointer === EMPTY_BRICK) return;
    if (isSlot(pointer)) this.mixed += delta;
    else this.uniform += delta;
    res.occupied += delta;
    this.layerCounts[gy]! += delta;
    this.boundsDirty = true;
  }
}

/** Grid dimensions for the configured world bounds and window width. */
export function brickGridSize(gridChunksXZ: number, minY: number, maxY: number): BrickGridSize {
  const minBrickY = Math.floor(minY / BRICK_SIZE);
  const maxBrickY = Math.ceil(maxY / BRICK_SIZE);
  const xz = gridChunksXZ * BRICKS_PER_AXIS;
  return { x: xz, y: maxBrickY - minBrickY, z: xz, minBrickY };
}

