import { describe, expect, it } from 'vitest';
import { BlockId } from './blocks';
import { Chunk } from './chunk';
import { CHUNK_SIZE } from './coords';
import { EmitterRegistry, scanChunk } from './emitters';
import { World } from './world';

const S = CHUNK_SIZE;

describe('EmitterRegistry', () => {
  it('finds emissive blocks at their absolute cells', () => {
    const c = new Chunk(-1, 2, 0);
    c.set(3, 4, 5, BlockId.torch);
    c.set(0, 0, 0, BlockId.stone);
    c.set(S - 1, S - 1, S - 1, BlockId.lava);
    const found = scanChunk(c).sort((a, b) => a.id - b.id);
    expect(found).toEqual([
      { x: -S + 3, y: 2 * S + 4, z: 5, id: BlockId.torch },
      { x: -1, y: 3 * S - 1, z: S - 1, id: BlockId.lava },
    ]);
  });

  it('mirrors world changes and removals', () => {
    const world = new World();
    world.addChunk(new Chunk(0, 0, 0));
    const reg = new EmitterRegistry();
    reg.apply(world.takeChanges());
    expect(reg.total).toBe(0);

    world.setBlock(1, 2, 3, BlockId.torch);
    world.setBlock(10, 2, 3, BlockId.torch);
    const v = reg.version;
    reg.apply(world.takeChanges());
    expect(reg.total).toBe(2);
    expect(reg.version).toBeGreaterThan(v);

    world.setBlock(1, 2, 3, BlockId.air);
    reg.apply(world.takeChanges());
    expect(reg.total).toBe(1);

    world.removeChunk(0, 0, 0);
    reg.apply(world.takeChanges());
    expect(reg.total).toBe(0);
  });

  it('returns the nearest emitters within a radius', () => {
    const world = new World();
    world.addChunk(new Chunk(0, 0, 0));
    for (const x of [2, 20, 8]) world.setBlock(x, 0, 0, BlockId.torch);
    const reg = new EmitterRegistry();
    reg.apply(world.takeChanges());
    expect(reg.nearest(0, 0, 0, 15, 8).map((e) => e.x)).toEqual([2, 8]);
    expect(reg.nearest(0, 0, 0, 100, 1).map((e) => e.x)).toEqual([2]);
  });
});
