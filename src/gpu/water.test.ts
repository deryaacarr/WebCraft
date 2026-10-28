import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { config } from '../config';
import { BlockId } from '../world/blocks';
import { waterCoefficients } from './water';

describe('water', () => {
  it('WATER_ID in water.wgsl is BlockId.water', () => {
    const wgsl = readFileSync(new URL('../shaders/water.wgsl', import.meta.url), 'utf8');
    expect(wgsl).toMatch(new RegExp(`const WATER_ID: u32 = ${BlockId.water}u;`));
  });

  it('absorption reproduces the configured colour after the reference distance', () => {
    const w = { ...config.water, absorptionColor: [0.3, 0.79, 0.87] as [number, number, number], absorptionDistance: 4 };
    const { absorption } = waterCoefficients(w);
    absorption.forEach((a, i) => expect(Math.exp(-a * 4)).toBeCloseTo(w.absorptionColor[i]!, 6));
    // Red is absorbed first.
    expect(absorption[0]).toBeGreaterThan(absorption[1]);
  });

  it('clarity moves scattering between the murky and clear limits', () => {
    const at = (clarity: number) => waterCoefficients({ ...config.water, clarity }).scattering[0];
    expect(at(0)).toBeCloseTo(config.water.maxScattering);
    expect(at(1)).toBeCloseTo(config.water.minScattering);
    expect(at(0.5)).toBeLessThan(at(0.2));
  });
});
