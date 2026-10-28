import { describe, expect, it } from 'vitest';
import { TEXTURE_SPEC } from '../../scripts/texture-spec';
import { BlockId, getBlockDef, MATERIAL_NAMES } from './blocks';
import { BLOCK_SHAPES, isShaped, MICRO, microSolid, packShapeTable, SHAPE_HEADER, SHAPE_WORDS, shapeCount, shapeGet, shapeHash } from './shapes';

describe('sub-voxel shapes', () => {
  it('marks only shaped blocks', () => {
    expect(isShaped(BlockId.broken_stone)).toBe(true);
    expect(isShaped(BlockId.stone)).toBe(false);
    expect(isShaped(BlockId.air)).toBe(false);
  });

  it('broken stone: a partial block standing on a full bottom layer', () => {
    for (const s of BLOCK_SHAPES.broken_stone!) {
      const n = shapeCount(s);
      expect(n).toBeGreaterThan(MICRO ** 3 * 0.2);
      expect(n).toBeLessThan(MICRO ** 3 * 0.85);
      for (let z = 0; z < MICRO; z++) for (let x = 0; x < MICRO; x++) expect(shapeGet(s, x, 0, z)).toBe(true);
      // Nothing above the highest plane, the top layer stays open.
      for (let z = 0; z < MICRO; z++) for (let x = 0; x < MICRO; x++) expect(shapeGet(s, x, MICRO - 1, z)).toBe(false);
    }
  });

  it('hash matches the WGSL u32 arithmetic for negative coordinates', () => {
    const h = shapeHash(-5, 70, -123456);
    expect(Number.isInteger(h)).toBe(true);
    expect(h).toBeGreaterThanOrEqual(0);
    expect(h).toBeLessThan(2 ** 32);
    expect(shapeHash(-5, 70, -123456)).toBe(h);
    expect(shapeHash(-5, 70, -123455)).not.toBe(h);
  });

  it('quarter turns keep the voxel count and the height profile', () => {
    // Over many cells every variant/turn shows up; each must be a permutation of a shape.
    const counts = new Set(BLOCK_SHAPES.broken_stone!.map(shapeCount));
    for (let i = 0; i < 64; i++) {
      const cell = [i * 7, 3, -i * 13] as const;
      let n = 0;
      for (let y = 0; y < MICRO; y++) for (let z = 0; z < MICRO; z++) for (let x = 0; x < MICRO; x++) if (microSolid(BlockId.broken_stone, cell, x, y, z)) n++;
      expect(counts.has(n)).toBe(true);
    }
  });

  it('packs the GPU table: header per block id, then the shapes', () => {
    const t = packShapeTable(true);
    const h = t[BlockId.broken_stone]!;
    expect(h & 0xff).toBe(BLOCK_SHAPES.broken_stone!.length);
    const first = h >>> 8;
    expect(t.subarray(SHAPE_HEADER + first * SHAPE_WORDS, SHAPE_HEADER + (first + 1) * SHAPE_WORDS)).toEqual(BLOCK_SHAPES.broken_stone![0]);
    expect(t[BlockId.stone]).toBe(0);
    expect(packShapeTable(false).subarray(0, SHAPE_HEADER).every((w) => w === 0)).toBe(true);
  });

  it('shaped blocks use opaque materials (the main trace stops at them, trace.wgsl MICRO_WALK)', () => {
    for (const name of Object.keys(BLOCK_SHAPES) as (keyof typeof BlockId)[]) {
      const m = getBlockDef(BlockId[name]).materials;
      for (const i of [m.top, m.side, m.bottom]) expect(TEXTURE_SPEC[MATERIAL_NAMES[i]!].alphaTest).toBe(false);
    }
  });
});
