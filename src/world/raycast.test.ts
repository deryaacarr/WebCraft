import { describe, expect, it } from 'vitest';
import { BlockId } from './blocks';
import { Chunk } from './chunk';
import { raycast } from './raycast';
import { MICRO, microSolid } from './shapes';
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

  it('hits shaped blocks only on their micro voxels', () => {
    const w = world();
    const cell = [4, 2, 4] as const;
    w.setBlock(...cell, BlockId.broken_stone);
    // Straight down through every micro column: the hit is the top of that column.
    for (let z = 0; z < MICRO; z++) {
      for (let x = 0; x < MICRO; x++) {
        let top = 0;
        while (top < MICRO && microSolid(BlockId.broken_stone, cell, x, top, z)) top++;
        const o = [cell[0] + (x + 0.5) / MICRO, 10.5, cell[2] + (z + 0.5) / MICRO] as const;
        const h = raycast(w, [...o], [0, -1, 0], 100);
        expect(h).toMatchObject({ cell: [...cell], normal: [0, 1, 0], id: BlockId.broken_stone });
        expect(h!.t).toBeCloseTo(o[1] - (cell[1] + top / MICRO), 9);
      }
    }
    // The top layer is open: a level ray through it passes the block.
    w.setBlock(9, 2, 4, BlockId.stone);
    const h = raycast(w, [0.5, 2 + 15.5 / 16, 4.5], [1, 0, 0], 100);
    expect(h).toMatchObject({ cell: [9, 2, 4], id: BlockId.stone });
    // Horizontal hit on a micro side face (bottom layer is full).
    const s = raycast(w, [0.5, 2 + 0.5 / MICRO, 4.5], [1, 0, 0], 100);
    expect(s).toMatchObject({ cell: [4, 2, 4], normal: [-1, 0, 0] });
    expect(s!.t).toBeCloseTo(3.5, 9);
  });
});
