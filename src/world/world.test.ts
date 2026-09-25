import { afterEach, describe, expect, it } from 'vitest';
import { config } from '../config';
import { BlockId } from './blocks';
import { Chunk } from './chunk';
import { CHUNK_SIZE } from './coords';
import { World } from './world';

const S = CHUNK_SIZE;

function worldWith(...coords: [number, number, number][]): World {
  const w = new World();
  for (const [x, y, z] of coords) w.addChunk(new Chunk(x, y, z));
  w.takeDirty();
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

  it('tracks dirty chunks and clears them on takeDirty', () => {
    const w = worldWith([0, 0, 0], [1, 0, 0]);
    w.setBlock(10, 10, 10, BlockId.stone);
    expect(w.isDirty(0, 0, 0)).toBe(true);
    expect(w.isDirty(1, 0, 0)).toBe(false);
    expect(w.takeDirty().map((c) => c.key)).toEqual(['0,0,0']);
    expect(w.takeDirty()).toEqual([]);
  });

  it('dirties the neighbour when editing a border block', () => {
    const w = worldWith([0, 0, 0], [1, 0, 0], [-1, 0, 0]);
    w.setBlock(S - 1, 0, 5, BlockId.stone);
    expect(w.isDirty(1, 0, 0)).toBe(true);
    expect(w.isDirty(-1, 0, 0)).toBe(false);
  });

  it('does not dirty anything for a no-op write', () => {
    const w = worldWith([0, 0, 0]);
    expect(w.setBlock(1, 1, 1, BlockId.air)).toBe(false);
    expect(w.takeDirty()).toEqual([]);
  });

  it('dirties neighbours on add/remove and forgets removed chunks', () => {
    const w = worldWith([0, 0, 0]);
    w.addChunk(new Chunk(0, 1, 0));
    expect(w.takeDirty().map((c) => c.key).sort()).toEqual(['0,0,0', '0,1,0']);
    w.setBlock(0, S, 0, BlockId.dirt);
    w.removeChunk(0, 1, 0);
    expect(w.hasChunk(0, 1, 0)).toBe(false);
    expect(w.getBlock(0, S, 0)).toBe(BlockId.air);
    expect(w.takeDirty().map((c) => c.key)).toEqual(['0,0,0']);
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
