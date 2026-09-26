import { describe, expect, it } from 'vitest';
import { MATERIAL_NAMES } from '../world/blocks';
import { fallbackSet, meanCoverage, meanEmission } from './materials';

describe('fallback material set', () => {
  it('has one flat layer per material in the pack layout [kind][layer][texel]', () => {
    const set = fallbackSet();
    const texels = set.resolution * set.resolution;
    expect(set.layers).toBe(MATERIAL_NAMES.length);
    expect(set.pack.byteLength).toBe(3 * set.layers * texels * 4);
    set.materials.forEach((m, i) => expect(m).toEqual({ firstLayer: i, count: 1, flags: 0 }));
  });

  it('encodes a flat normal and LabPBR "no emission" for ordinary materials', () => {
    const set = fallbackSet();
    const texels = set.resolution * set.resolution;
    const normal = set.layers * texels * 4; // normal kind, layer 0
    expect([...set.pack.subarray(normal, normal + 4)]).toEqual([128, 128, 255, 128]);
    const specular = 2 * set.layers * texels * 4;
    expect(set.pack[specular + 3]).toBe(255);
    const torch = MATERIAL_NAMES.indexOf('torch');
    expect(set.pack[(2 * set.layers + torch) * texels * 4 + 3]).toBeLessThan(255);
  });

  it('averages emission (linear albedo × emission) only for emissive materials', () => {
    const set = fallbackSet();
    const e = meanEmission(set);
    const torch = MATERIAL_NAMES.indexOf('torch');
    const stone = MATERIAL_NAMES.indexOf('stone');
    expect(e[torch * 4]!).toBeGreaterThan(0.5);
    expect([e[stone * 4], e[stone * 4 + 1], e[stone * 4 + 2]]).toEqual([0, 0, 0]);
  });

  it('reports full coverage for opaque flat layers', () => {
    const c = meanCoverage(fallbackSet());
    for (const v of c) expect(v).toBeCloseTo(1);
  });
});
