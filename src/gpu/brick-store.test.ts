import { describe, expect, it } from 'vitest';
import { config } from '../config';
import { BlockId } from '../world/blocks';
import { Chunk } from '../world/chunk';
import { BRICK_SIZE, BRICKS_PER_AXIS, CHUNK_SIZE, voxelIndexInBrick } from '../world/coords';
import { TerrainGenerator } from '../world/gen/terrain';
import { World } from '../world/world';
import { BRICK_STRIDE_WORDS, BRICK_VOXEL_WORDS, EMPTY_FLAG, isEmptyPointer, isSlot, UNIFORM_FLAG, UNIFORM_ID_MASK } from './brick-pack';
import { brickGridSize, BrickStore, type BrickGridSize, type BrickSink } from './brick-store';

const S = CHUNK_SIZE;

/** Array-backed sink that mirrors exactly what the GPU buffers would contain. */
class ArraySink implements BrickSink {
  grid: Uint32Array;
  pool = new Uint32Array(0);
  origin: readonly [number, number, number] = [0, 0, 0];
  poolWrites = 0;
  poolWordsWritten = 0;
  gridWrites = 0;

  constructor(readonly size: BrickGridSize) {
    this.grid = new Uint32Array(size.x * size.y * size.z);
  }
  writeGrid(cell: number, pointers: Uint32Array): void {
    this.grid.set(pointers, cell);
    this.gridWrites++;
  }
  writePool(slot: number, words: Uint32Array): void {
    this.pool.set(words, slot * BRICK_STRIDE_WORDS);
    this.poolWrites++;
    this.poolWordsWritten += words.length;
  }
  growPool(capacity: number): void {
    const next = new Uint32Array(capacity * BRICK_STRIDE_WORDS);
    next.set(this.pool);
    this.pool = next;
  }
  setOrigin(origin: readonly [number, number, number]): void {
    this.origin = origin;
  }
  bounds: { min: readonly number[]; max: readonly number[] } = { min: [0, 0, 0], max: [0, 0, 0] };
  setBounds(min: readonly [number, number, number], max: readonly [number, number, number]): void {
    this.bounds = { min: [...min], max: [...max] };
  }

  /** CPU port of getVoxel() in brickmap.wgsl. */
  getVoxel(x: number, y: number, z: number): number {
    const b = [x >> 3, y >> 3, z >> 3] as const;
    const o = this.origin;
    const { size } = this;
    if (b[0] < o[0] || b[0] >= o[0] + size.x || b[1] < o[1] || b[1] >= o[1] + size.y || b[2] < o[2] || b[2] >= o[2] + size.z) return 0;
    const cell = ((b[1] - size.minBrickY) * size.z + (b[2] & (size.z - 1))) * size.x + (b[0] & (size.x - 1));
    const ptr = this.grid[cell]!;
    if (isEmptyPointer(ptr)) return 0;
    if (ptr & UNIFORM_FLAG) return ptr & UNIFORM_ID_MASK;
    const i = voxelIndexInBrick(x & 7, y & 7, z & 7);
    return (this.pool[ptr * BRICK_STRIDE_WORDS + (i >> 2)]! >>> ((i & 3) * 8)) & 0xff;
  }

  /** CPU port of the sub-cell occupancy test in trace.wgsl. */
  subcellOccupied(x: number, y: number, z: number): boolean {
    const { size } = this;
    const b = [x >> 3, y >> 3, z >> 3] as const;
    const ptr = this.grid[((b[1] - size.minBrickY) * size.z + (b[2] & (size.z - 1))) * size.x + (b[0] & (size.x - 1))]!;
    if (!isSlot(ptr)) return !isEmptyPointer(ptr);
    const s = (((y & 7) >> 1) * 4 + ((z & 7) >> 1)) * 4 + ((x & 7) >> 1);
    return ((this.pool[ptr * BRICK_STRIDE_WORDS + BRICK_VOXEL_WORDS + (s >> 5)]! >>> (s & 31)) & 1) === 1;
  }
}

function setup(gridChunks = 8, initialPoolBricks = 64, minY = -64, maxY = 128) {
  const size = brickGridSize(gridChunks, minY, maxY);
  const sink = new ArraySink(size);
  const store = new BrickStore(sink, { grid: size, initialPoolBricks, poolGrowth: 2, maxPoolBricks: 1 << 20 });
  const world = new World();
  return { size, sink, store, world };
}

function sync(store: BrickStore, world: World) {
  store.applyChanges(world.takeChanges());
  store.flush(world);
}

describe('BrickStore', () => {
  it('rejects grids that cannot be addressed toroidally', () => {
    const size = { x: 12, y: 4, z: 16, minBrickY: 0 };
    expect(() => new BrickStore(new ArraySink(size), { grid: size, initialPoolBricks: 8, poolGrowth: 2, maxPoolBricks: 64 })).toThrow();
  });

  it('encodes empty, uniform and mixed bricks; only mixed bricks use pool slots', () => {
    const { sink, store, world } = setup();
    store.setCenter(0, 0, world);
    world.addChunk(new Chunk(0, 0, 0)); // air
    world.addChunk(new Chunk(1, 0, 0, BlockId.stone)); // solid
    const mixed = new Chunk(0, 1, 0);
    mixed.set(1, 2, 3, BlockId.glass); // one mixed brick, 63 empty
    world.addChunk(mixed);
    sync(store, world);

    expect(store.stats).toMatchObject({ residentChunks: 3, mixedBricks: 1, uniformBricks: 64 });
    expect(sink.getVoxel(S + 5, 5, 5)).toBe(BlockId.stone);
    expect(sink.getVoxel(1, S + 2, 3)).toBe(BlockId.glass);
    expect(sink.getVoxel(2, S + 2, 3)).toBe(BlockId.air);
    expect(sink.poolWrites).toBe(1);
  });

  it('matches the world voxel-for-voxel on generated terrain (incl. negative coordinates)', () => {
    const { sink, store, world } = setup(8, 256, config.world.minY, config.world.maxY);
    const gen = new TerrainGenerator(config.terrain, config.world, 64);
    store.setCenter(0, 0, world);
    const top = gen.surfaceY(0, 0);
    for (let cz = -2; cz <= 1; cz++) {
      for (let cx = -2; cx <= 1; cx++) {
        for (let cy = Math.floor(top / S) - 2; cy <= Math.floor(top / S) + 1; cy++) {
          const g = gen.generate(cx, cy, cz);
          world.addChunk(Chunk.fromParts(cx, cy, cz, g.data, g.uniform, g.nonAir));
        }
      }
    }
    sync(store, world);
    expect(store.stats.mixedBricks).toBeGreaterThan(0);

    let mismatches = 0;
    let solid = 0;
    const base = (Math.floor(top / S) - 2) * S;
    for (let i = 0; i < 200_000; i++) {
      const x = -2 * S + ((i * 7919) % (4 * S));
      const z = -2 * S + ((i * 104729) % (4 * S));
      const y = base + ((i * 31337) % (4 * S));
      const expected = world.getBlock(x, y, z);
      if (expected !== BlockId.air) solid++;
      if (sink.getVoxel(x, y, z) !== expected) mismatches++;
    }
    expect(mismatches).toBe(0);
    expect(solid).toBeGreaterThan(1000);

    // Sub-cell masks: a sub-cell is marked occupied exactly when one of its voxels is solid.
    let maskErrors = 0;
    for (let i = 0; i < 20_000; i++) {
      const x = (-2 * S + ((i * 7919) % (4 * S))) & ~1;
      const z = (-2 * S + ((i * 104729) % (4 * S))) & ~1;
      const y = (base + ((i * 31337) % (4 * S))) & ~1;
      let any = false;
      for (let d = 0; d < 8; d++) any ||= world.getBlock(x + (d & 1), y + ((d >> 1) & 1), z + (d >> 2)) !== BlockId.air;
      if (sink.subcellOccupied(x, y, z) !== any) maskErrors++;
    }
    expect(maskErrors).toBe(0);
  });

  it('re-uploads only the brick touched by a block edit', () => {
    const { sink, store, world } = setup();
    store.setCenter(0, 0, world);
    world.addChunk(new Chunk(0, 0, 0, BlockId.stone));
    sync(store, world);
    const before = { pool: sink.poolWrites, words: sink.poolWordsWritten, grid: sink.gridWrites };

    world.setBlock(9, 9, 9, BlockId.air); // carve one voxel: brick becomes mixed
    sync(store, world);
    expect(sink.poolWrites - before.pool).toBe(1);
    expect(sink.poolWordsWritten - before.words).toBe(BRICK_STRIDE_WORDS);
    expect(sink.gridWrites - before.grid).toBe(1); // one row of BRICKS_PER_AXIS cells
    expect(sink.getVoxel(9, 9, 9)).toBe(BlockId.air);
    expect(sink.getVoxel(10, 9, 9)).toBe(BlockId.stone);

    world.setBlock(9, 9, 9, BlockId.stone); // uniform again: slot freed, no pool write
    sync(store, world);
    expect(sink.poolWrites - before.pool).toBe(1);
    expect(store.stats).toMatchObject({ mixedBricks: 0, uniformBricks: 64 });
  });

  it('frees slots on removal and reuses them', () => {
    const { store, world } = setup();
    store.setCenter(0, 0, world);
    const make = (cx: number) => {
      const c = new Chunk(cx, 0, 0);
      for (let b = 0; b < 8; b++) c.set(b * BRICK_SIZE, 0, 0, BlockId.dirt);
      return c;
    };
    world.addChunk(make(0));
    sync(store, world);
    const capacity = store.stats.poolCapacity;
    for (let i = 0; i < 50; i++) {
      world.removeChunk(0, 0, 0);
      world.addChunk(make(0));
      sync(store, world);
    }
    expect(store.stats.mixedBricks).toBe(BRICKS_PER_AXIS);
    expect(store.stats.poolCapacity).toBe(capacity);
  });

  it('grows the pool without losing data', () => {
    const { sink, store, world } = setup(8, 4);
    store.setCenter(0, 0, world);
    for (let cx = -2; cx < 2; cx++) {
      const c = new Chunk(cx, 0, 0);
      for (let i = 0; i < 64; i++) c.set((i % 4) * 8 + 1, Math.floor(i / 16) * 8, (Math.floor(i / 4) % 4) * 8, BlockId.sand);
      world.addChunk(c);
    }
    sync(store, world);
    expect(store.stats.mixedBricks).toBe(4 * 64);
    expect(store.stats.poolCapacity).toBeGreaterThanOrEqual(4 * 64);
    for (let cx = -2; cx < 2; cx++) expect(sink.getVoxel(cx * S + 1, 0, 0)).toBe(BlockId.sand);
  });

  it('shifts the window toroidally: keeps overlapping chunks, evicts and reuses cells', () => {
    const { sink, store, world, size } = setup(8); // window = 8 chunks
    const W = size.x / BRICKS_PER_AXIS;
    store.setCenter(0, 0, world);
    world.addChunk(new Chunk(-4, 0, 0, BlockId.stone)); // west edge of window [-4, 4)
    world.addChunk(new Chunk(0, 0, 0, BlockId.dirt));
    sync(store, world);
    expect(sink.getVoxel(-4 * S, 0, 0)).toBe(BlockId.stone);
    const uploadsBefore = sink.gridWrites;

    // Move one chunk east: window [-3, 5). Chunk -4 leaves, its cells alias chunk 4 (= -4 + W).
    expect(store.setCenter(1, 0, world)).toBe(true);
    expect(store.stats.residentChunks).toBe(1);
    expect(sink.getVoxel(-4 * S, 0, 0)).toBe(BlockId.air); // out of window
    expect(sink.getVoxel(0, 0, 0)).toBe(BlockId.dirt); // untouched, no re-upload
    world.addChunk(new Chunk(-4 + W, 0, 0, BlockId.sand));
    sync(store, world);
    expect(sink.getVoxel((-4 + W) * S, 0, 0)).toBe(BlockId.sand);
    // Only the evicted chunk's rows and the new chunk's rows were written.
    expect(sink.gridWrites - uploadsBefore).toBe(2 * BRICKS_PER_AXIS * BRICKS_PER_AXIS);

    // Moving back re-queues the still-loaded chunk that re-enters the window.
    world.removeChunk(-4 + W, 0, 0);
    store.applyChanges(world.takeChanges());
    store.setCenter(0, 0, world);
    store.flush(world);
    expect(sink.getVoxel(-4 * S, 0, 0)).toBe(BlockId.stone);
  });

  it('reports a clipping box around non-empty bricks (incl. the highest terrain layer)', () => {
    const { sink, store, world, size } = setup();
    store.setCenter(0, 0, world);
    expect(sink.bounds).toEqual({ min: [0, 0, 0], max: [0, 0, 0] }); // empty world
    world.addChunk(new Chunk(-1, 0, 2)); // all air: does not widen the box
    const c = new Chunk(1, 0, -2);
    c.set(3, 17, 3, BlockId.stone); // brick layer 2 of chunk layer 0
    world.addChunk(c);
    world.addChunk(new Chunk(0, -1, 0, BlockId.stone)); // full chunk one layer lower
    sync(store, world);
    const P = BRICKS_PER_AXIS;
    expect(sink.bounds).toEqual({ min: [0, -P, -2 * P], max: [2 * P, 3, 1 * P] });
    expect(size.minBrickY).toBeLessThanOrEqual(-P);

    world.removeChunk(1, 0, -2);
    sync(store, world);
    expect(sink.bounds).toEqual({ min: [0, -P, 0], max: [P, 0, P] });
    world.setBlock(0, 0, 0, BlockId.air); // carving keeps the box (brick still occupied)
    sync(store, world);
    expect(sink.bounds.max[1]).toBe(0);
  });

  it('reads distance-annotated empty cells (written by the GPU pass) as air', () => {
    const { sink, store, world } = setup();
    store.setCenter(0, 0, world);
    world.addChunk(new Chunk(0, 0, 0));
    sync(store, world);
    sink.grid.fill((EMPTY_FLAG | 5) >>> 0);
    expect(sink.getVoxel(3, 3, 3)).toBe(BlockId.air);
    expect(isSlot((EMPTY_FLAG | 5) >>> 0)).toBe(false);
  });

  it('ignores chunk changes outside the window', () => {
    const { store, world } = setup(8);
    store.setCenter(0, 0, world);
    world.addChunk(new Chunk(40, 0, 0, BlockId.stone));
    sync(store, world);
    expect(store.stats.residentChunks).toBe(0);
  });

  it('respects the upload time budget but always makes progress', () => {
    const { store, world } = setup();
    store.setCenter(0, 0, world);
    for (let cx = -3; cx <= 3; cx++) world.addChunk(new Chunk(cx, 0, 0, BlockId.stone));
    store.applyChanges(world.takeChanges());
    let t = 0;
    store.process(world, 1, () => (t += 10)); // every chunk "takes" 10 ms
    expect(store.stats.residentChunks).toBe(1);
    expect(store.stats.pendingChunks).toBe(6);
  });

  it('leaves no pointer to a freed slot after regeneration (clear)', () => {
    const { sink, store, world } = setup();
    store.setCenter(0, 0, world);
    const c = new Chunk(0, 0, 0);
    c.set(0, 0, 0, BlockId.glass);
    world.addChunk(c);
    sync(store, world);
    world.clear();
    sync(store, world);
    expect(sink.grid.some((p) => isSlot(p))).toBe(false);
    expect(store.stats).toMatchObject({ residentChunks: 0, mixedBricks: 0, uniformBricks: 0 });
  });
});
