import { afterEach, describe, expect, it } from 'vitest';
import { config } from '../config';
import { BlockId } from './blocks';
import { Chunk } from './chunk';
import { BRICK_SIZE, BRICKS_PER_CHUNK, brickIndexInChunk, CHUNK_SIZE } from './coords';
import { World } from './world';

const S = CHUNK_SIZE;

function worldWith(...coords: [number, number, number][]): World {
  const w = new World();
  for (const [x, y, z] of coords) w.addChunk(new Chunk(x, y, z));
  w.takeChanges();
  return w;
}

describe('World', () => {
  it('sets and gets blocks at positive and negative world coordinates', () => {
    const w = worldWith([0, 0, 0], [-1, -1, -1], [-1, 0, 0]);
    expect(w.setBlock(5, 6, 7, BlockId.stone)).toBe(true);
    expect(w.setBlock(-1, -1, -1, BlockId.dirt)).toBe(true);
    expect(w.setBlock(-S, -S, -S, BlockId.sand)).toBe(true);
    expect(w.setBlock(-1, 0, 0, BlockId.glass)).toBe(true);

    expect(w.getBlock(5, 6, 7)).toBe(BlockId.stone);
    expect(w.getBlock(-1, -1, -1)).toBe(BlockId.dirt);
    expect(w.getBlock(-S, -S, -S)).toBe(BlockId.sand);
    expect(w.getBlock(-1, 0, 0)).toBe(BlockId.glass);
    expect(w.getBlock(0, 0, 0)).toBe(BlockId.air);

    // Written into the right chunk and local slot.
    expect(w.getChunk(-1, -1, -1)?.get(S - 1, S - 1, S - 1)).toBe(BlockId.dirt);
    expect(w.getChunk(-1, -1, -1)?.get(0, 0, 0)).toBe(BlockId.sand);
  });

  it('floors fractional positions', () => {
    const w = worldWith([-1, 0, 0]);
    w.setBlock(-1, 0, 0, BlockId.gravel);
    expect(w.getBlock(-0.25, 0.9, 0.1)).toBe(BlockId.gravel);
  });

  it('returns air / false for unloaded chunks', () => {
    const w = new World();
    expect(w.getBlock(100, 100, 100)).toBe(BlockId.air);
    expect(w.setBlock(100, 100, 100, BlockId.stone)).toBe(false);
  });

  it('tracks changes per brick and clears them on takeChanges', () => {
    const w = worldWith([0, 0, 0], [1, 0, 0]);
    w.setBlock(10, 10, 10, BlockId.stone);
    w.setBlock(11, 10, 10, BlockId.dirt); // same brick
    expect(w.isDirty(0, 0, 0)).toBe(true);
    expect(w.isDirty(1, 0, 0)).toBe(false);
    const { changed, removed } = w.takeChanges();
    expect(removed).toEqual([]);
    expect(changed.map((c) => c.chunk.key)).toEqual(['0,0,0']);
    const flagged = [...changed[0]!.bricks.keys()].filter((i) => changed[0]!.bricks[i]);
    expect(flagged).toEqual([brickIndexInChunk(10, 10, 10)]);
    expect(w.takeChanges()).toEqual({ changed: [], removed: [] });
  });

  it('marks every brick of an added chunk as changed', () => {
    const w = new World();
    w.addChunk(new Chunk(0, 0, 0));
    const [change] = w.takeChanges().changed;
    expect(change?.bricks.length).toBe(BRICKS_PER_CHUNK);
    expect(change?.bricks.every((b) => b === 1)).toBe(true);
  });

  it('does not dirty the neighbour on a border edit (the ray tracer reads it directly)', () => {
    const w = worldWith([0, 0, 0], [1, 0, 0]);
    w.setBlock(S - 1, 0, 5, BlockId.stone);
    expect(w.isDirty(1, 0, 0)).toBe(false);
    expect(w.takeChanges().changed).toHaveLength(1);
  });

  it('flags the right bricks at brick borders', () => {
    const w = worldWith([0, 0, 0]);
    w.setBlock(BRICK_SIZE - 1, 0, 0, BlockId.stone);
    w.setBlock(BRICK_SIZE, 0, 0, BlockId.stone);
    const bricks = w.takeChanges().changed[0]!.bricks;
    expect(bricks[0]).toBe(1);
    expect(bricks[1]).toBe(1);
    expect(bricks.reduce((a, b) => a + b, 0)).toBe(2);
  });

  it('does not report anything for a no-op write', () => {
    const w = worldWith([0, 0, 0]);
    expect(w.setBlock(1, 1, 1, BlockId.air)).toBe(false);
    expect(w.takeChanges().changed).toEqual([]);
  });

  it('reports removals and forgets pending changes of removed chunks', () => {
    const w = worldWith([0, 0, 0], [0, 1, 0]);
    w.setBlock(0, S, 0, BlockId.dirt);
    w.removeChunk(0, 1, 0);
    expect(w.hasChunk(0, 1, 0)).toBe(false);
    expect(w.getBlock(0, S, 0)).toBe(BlockId.air);
    expect(w.takeChanges()).toEqual({ changed: [], removed: ['0,1,0'] });
  });

  it('reports a removal followed by a re-add as both (removal first, then full chunk)', () => {
    const w = worldWith([0, 0, 0]);
    w.removeChunk(0, 0, 0);
    w.addChunk(new Chunk(0, 0, 0, BlockId.stone));
    const { removed, changed } = w.takeChanges();
    expect(removed).toEqual(['0,0,0']);
    expect(changed[0]?.chunk.uniformBlock).toBe(BlockId.stone);
  });

  it('reports every chunk as removed on clear()', () => {
    const w = worldWith([0, 0, 0], [1, 0, 0]);
    w.clear();
    expect(w.takeChanges().removed.sort()).toEqual(['0,0,0', '1,0,0']);
    expect(w.chunkCount).toBe(0);
  });

  describe('vertical bounds', () => {
    const saved = { ...config.world };
    afterEach(() => Object.assign(config.world, saved));

    it('treats Y outside [minY, maxY) as air and refuses writes there', () => {
      Object.assign(config.world, { minY: 0, maxY: S - 4 });
      const w = worldWith([0, 0, 0], [0, -1, 0]);
      expect(w.setBlock(1, S - 5, 1, BlockId.stone)).toBe(true);
      expect(w.setBlock(1, S - 4, 1, BlockId.stone)).toBe(false);
      expect(w.setBlock(1, -1, 1, BlockId.stone)).toBe(false);
      w.getChunk(0, -1, 0)?.set(1, S - 1, 1, BlockId.stone);
      expect(w.getBlock(1, -1, 1)).toBe(BlockId.air);
    });
  });

  it('does not serve a stale chunk from the lookup cache after replacement', () => {
    const w = worldWith([0, 0, 0]);
    w.getBlock(0, 0, 0);
    w.addChunk(new Chunk(0, 0, 0, BlockId.stone));
    expect(w.getBlock(0, 0, 0)).toBe(BlockId.stone);
  });
});
