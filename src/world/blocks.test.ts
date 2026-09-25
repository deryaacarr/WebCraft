import { describe, expect, it } from 'vitest';
import { BLOCKS, BlockId, MATERIAL_NAMES, NO_MATERIAL, getBlockDef, isSolid, isTransparent } from './blocks';

describe('block registry', () => {
  it('has dense, uint16 ids matching array positions', () => {
    BLOCKS.forEach((b, i) => {
      expect(b.id).toBe(i);
      expect(b.id).toBeLessThanOrEqual(0xffff);
    });
    expect(BLOCKS).toHaveLength(Object.keys(BlockId).length);
  });

  it('contains the starter blocks', () => {
    const names = BLOCKS.map((b) => b.name);
    for (const n of ['air', 'grass', 'dirt', 'stone', 'cobblestone', 'gravel', 'sand', 'water',
      'oak_log', 'oak_leaves', 'oak_planks', 'glass', 'torch', 'lava']) {
      expect(names).toContain(n);
    }
  });

  it('references only valid materials', () => {
    for (const b of BLOCKS) {
      for (const m of [b.materials.top, b.materials.side, b.materials.bottom]) {
        if (b.id === BlockId.air) {
          expect(m).toBe(NO_MATERIAL);
        } else {
          expect(m).toBeGreaterThanOrEqual(0);
          expect(m).toBeLessThan(MATERIAL_NAMES.length);
        }
      }
    }
  });

  it('uses per-face materials where needed', () => {
    const grass = getBlockDef(BlockId.grass).materials;
    expect(new Set([grass.top, grass.side, grass.bottom]).size).toBe(3);
  });

  it('keeps physical flags consistent', () => {
    expect(isSolid(BlockId.air)).toBe(false);
    expect(isSolid(BlockId.stone)).toBe(true);
    expect(isTransparent(BlockId.glass)).toBe(true);
    expect(isTransparent(BlockId.stone)).toBe(false);
    for (const b of BLOCKS) {
      if (b.solid) expect(b.footstep).not.toBeNull();
    }
    expect(getBlockDef(BlockId.torch).emissive).toBeGreaterThan(0);
    expect(getBlockDef(BlockId.lava).emissive).toBeGreaterThan(0);
    expect(() => getBlockDef(9999)).toThrow();
  });
});
