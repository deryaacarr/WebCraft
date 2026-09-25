import { describe, expect, it } from 'vitest';
import { BlockId } from './blocks';
import { Chunk } from './chunk';
import { raycast } from './raycast';
import { World } from './world';

function world(): World {
  const w = new World();
  for (const [x, y, z] of [[0, 0, 0], [-1, 0, 0], [0, 0, -1], [-1, 0, -1], [0, -1, 0], [-1, -1, 0]] as const) w.addChunk(new Chunk(x, y, z));
  return w;
}

describe('raycast', () => {
  it('hits the first solid voxel with the entered face normal', () => {
    const w = world();
    w.setBlock(5, 2, 2, BlockId.stone);
    const h = raycast(w, [0.5, 2.5, 2.5], [1, 0, 0], 100);
    expect(h).toMatchObject({ cell: [5, 2, 2], normal: [-1, 0, 0], id: BlockId.stone });
    expect(h?.t).toBeCloseTo(4.5);
  });

  it('works along negative directions and across chunk borders', () => {
    const w = world();
    w.setBlock(-3, 1, -2, BlockId.glass);
    const h = raycast(w, [2.5, 1.5, -1.5], [-5.5, 0, -0.5], 100);
    expect(h).toMatchObject({ cell: [-3, 1, -2], id: BlockId.glass });
    expect(h?.normal).toEqual([1, 0, 0]);
  });

  it('reports the top face when looking down', () => {
    const w = world();
    w.setBlock(3, -4, 3, BlockId.grass);
    expect(raycast(w, [3.5, 10.2, 3.5], [0, -1, 0], 100)?.normal).toEqual([0, 1, 0]);
  });

  it('misses beyond maxT and handles axis-aligned rays', () => {
    const w = world();
    w.setBlock(0, 0, 20, BlockId.stone);
    expect(raycast(w, [0.5, 0.5, 0.5], [0, 0, 1], 10)).toBeNull();
    expect(raycast(w, [0.5, 0.5, 0.5], [0, 0, 1], 30)?.cell).toEqual([0, 0, 20]);
  });

  it('reports a zero normal when starting inside a block', () => {
    const w = world();
    w.setBlock(1, 1, 1, BlockId.dirt);
    expect(raycast(w, [1.5, 1.5, 1.5], [1, 0, 0], 10)).toMatchObject({ t: 0, normal: [0, 0, 0] });
  });
});
