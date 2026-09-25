import { BlockId } from './blocks';
import { CHUNK_VOLUME, chunkKey, localIndex } from './coords';

/** Dense block storage: one byte per block while every id fits, two otherwise. */
export type BlockArray = Uint8Array | Uint16Array;

const MAX_NARROW_ID = 0xff;

/** Allocates dense storage filled with `fill`, as narrow as `fill` and `next` allow. */
function allocate(fill: number, next: number): BlockArray {
  return fill <= MAX_NARROW_ID && next <= MAX_NARROW_ID
    ? new Uint8Array(CHUNK_VOLUME).fill(fill)
    : new Uint16Array(CHUNK_VOLUME).fill(fill);
}

/**
 * A CHUNK_SIZE³ block of voxels storing uint16 block ids.
 *
 * Memory: chunks made of a single block type (open air, deep stone) keep no array at
 * all — just the id. The dense array is allocated on the first write that breaks
 * uniformity (and dropped again when the chunk becomes all air). While every id is
 * < 256 it is a Uint8Array (32 KB instead of 64 KB); writing a larger id widens it.
 */
export class Chunk {
  readonly key: string;
  private data: BlockArray | null = null;
  private uniformId: number;
  private nonAir: number;

  constructor(
    readonly cx: number,
    readonly cy: number,
    readonly cz: number,
    fill: number = BlockId.air,
  ) {
    this.key = chunkKey(cx, cy, cz);
    this.uniformId = fill;
    this.nonAir = fill === BlockId.air ? 0 : CHUNK_VOLUME;
  }

  /** Builds a chunk from dense data; stays compact (uniform or 8-bit) when possible. */
  static fromArray(cx: number, cy: number, cz: number, data: BlockArray): Chunk {
    if (data.length !== CHUNK_VOLUME) {
      throw new Error(`Chunk data must have ${CHUNK_VOLUME} entries, got ${data.length}`);
    }
    const first = data[0] ?? BlockId.air;
    const chunk = new Chunk(cx, cy, cz, first);
    let uniform = true;
    let nonAir = 0;
    let max = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i]!;
      if (v !== first) uniform = false;
      if (v !== BlockId.air) nonAir++;
      if (v > max) max = v;
    }
    if (uniform) return chunk;
    chunk.data = data instanceof Uint16Array && max <= MAX_NARROW_ID ? Uint8Array.from(data) : data;
    chunk.nonAir = nonAir;
    return chunk;
  }

  /**
   * Adopts already-analysed data (e.g. from a terrain worker) without rescanning it.
   * `data` null means every block is `uniform`; otherwise `nonAir` must be exact.
   */
  static fromParts(
    cx: number,
    cy: number,
    cz: number,
    data: BlockArray | null,
    uniform: number,
    nonAir: number,
  ): Chunk {
    const chunk = new Chunk(cx, cy, cz, data ? BlockId.air : uniform);
    if (data) {
      if (data.length !== CHUNK_VOLUME) {
        throw new Error(`Chunk data must have ${CHUNK_VOLUME} entries, got ${data.length}`);
      }
      chunk.data = data;
      chunk.nonAir = nonAir;
    }
    return chunk;
  }

  /** True if every block is air. */
  get isEmpty(): boolean {
    return this.nonAir === 0;
  }

  /** Non-null when every block has the same id. */
  get uniformBlock(): number | null {
    return this.data ? null : this.uniformId;
  }

  get nonAirCount(): number {
    return this.nonAir;
  }

  /** Bytes held by the dense block array (0 for uniform chunks). */
  get byteLength(): number {
    return this.data?.byteLength ?? 0;
  }

  get(lx: number, ly: number, lz: number): number {
    return this.data ? (this.data[localIndex(lx, ly, lz)] ?? BlockId.air) : this.uniformId;
  }

  /** Sets a block by local coordinates. Returns true if the value changed. */
  set(lx: number, ly: number, lz: number, id: number): boolean {
    let data = this.data;
    if (!data) {
      if (id === this.uniformId) return false;
      data = this.data = allocate(this.uniformId, id);
    } else if (id > MAX_NARROW_ID && data instanceof Uint8Array) {
      data = this.data = Uint16Array.from(data);
    }
    const i = localIndex(lx, ly, lz);
    const prev = data[i] ?? BlockId.air;
    if (prev === id) return false;
    data[i] = id;
    this.nonAir += (id !== BlockId.air ? 1 : 0) - (prev !== BlockId.air ? 1 : 0);
    if (this.nonAir === 0) {
      this.data = null;
      this.uniformId = BlockId.air;
    }
    return true;
  }

  /** Replaces every block with `id`, releasing the dense array. */
  fill(id: number): void {
    this.data = null;
    this.uniformId = id;
    this.nonAir = id === BlockId.air ? 0 : CHUNK_VOLUME;
  }

  /** Dense uint16 copy of the chunk (for GPU packing / worker transfer). */
  toArray(): Uint16Array {
    return this.data ? Uint16Array.from(this.data) : new Uint16Array(CHUNK_VOLUME).fill(this.uniformId);
  }
}
