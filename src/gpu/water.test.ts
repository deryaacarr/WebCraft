import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { config } from '../config';
import { BlockId } from '../world/blocks';
import { waterAt, waterCoefficients, waterTraceDistance, waveHeight } from './water';

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

  it('stops refracted rays where the clearest channel keeps 1 % of the light', () => {
    const w = config.water;
    const { absorption, scattering } = waterCoefficients(w);
    const d = waterTraceDistance(w);
    expect(d).toBeLessThanOrEqual(w.maxDistance);
    const left = Math.max(...absorption.map((a, i) => Math.exp(-(a + scattering[i]!) * d)));
    expect(d < w.maxDistance ? left : 0.01).toBeCloseTo(0.01, 4);
    // Murky water: shorter.
    expect(waterTraceDistance({ ...w, clarity: 0 })).toBeLessThan(d);
  });

  it('finds the camera under the wavy surface of the water column above it', () => {
    // Water from y = 60 to 64 at every column, air above.
    const getBlock = (_x: number, y: number) => (y >= 60 && y <= 64 ? BlockId.water : 0);
    const surface = 64 + config.water.level;
    const t = 3;
    const h = waveHeight(10.5, -3.25, t);
    expect(Math.abs(h)).toBeLessThan(0.05);
    expect(waterAt(getBlock, 10.5, 62.2, -3.25, t)).toEqual({ under: true, surfaceY: surface });
    // In the top voxel: under or not depending on the waves at that spot and moment.
    expect(waterAt(getBlock, 10.5, surface + h - 1e-4, -3.25, t).under).toBe(true);
    expect(waterAt(getBlock, 10.5, surface + h + 1e-4, -3.25, t).under).toBe(false);
    expect(waterAt(getBlock, 10.5, 66, -3.25, t).under).toBe(false);
  });
});
