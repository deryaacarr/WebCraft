import { describe, expect, it } from 'vitest';
import { MATERIAL_NAMES } from '../src/world/blocks.ts';
import { ambientCgIds, TEXTURE_RESOLUTIONS, TEXTURE_SPEC } from './texture-spec.ts';
import { TEXTURE_RESOLUTIONS as RUNTIME_RESOLUTIONS } from '../src/config.ts';

describe('texture spec', () => {
  it('covers every material with at least one variant', () => {
    expect(Object.keys(TEXTURE_SPEC).sort()).toEqual([...MATERIAL_NAMES].sort());
    for (const spec of Object.values(TEXTURE_SPEC)) expect(spec.variants.length).toBeGreaterThan(0);
  });

  it('uses valid crops and only alpha-tests leaves', () => {
    for (const [name, spec] of Object.entries(TEXTURE_SPEC)) {
      for (const v of spec.variants) {
        if (v.kind !== 'ambientcg') continue;
        expect(v.crop).toBeGreaterThan(0);
        expect(v.crop).toBeLessThanOrEqual(1);
      }
      expect(spec.alphaTest).toBe(name === 'oak_leaves');
    }
  });

  it('builds the resolutions the runtime offers', () => {
    expect([...TEXTURE_RESOLUTIONS]).toEqual([...RUNTIME_RESOLUTIONS]);
    expect(ambientCgIds().length).toBeGreaterThan(10);
  });
});
