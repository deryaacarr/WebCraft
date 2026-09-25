import { describe, expect, it } from 'vitest';
import { config } from '../../config';
import { BlockId } from '../blocks';
import { Chunk } from '../chunk';
import { CHUNK_SIZE, toChunkCoord } from '../coords';
import { World } from '../world';
import { TerrainGenerator } from './terrain';

const params = config.terrain;
const S = CHUNK_SIZE;

/**
 * Region tests run on a pinned seed whose chunks -2..2 hold both a lake and forested
 * land (found by scanning seeds 1..400 for 20–50 % underwater columns). If terrain
 * params change and the lake disappears, the water assertions fail on purpose:
 * re-scan and pick a new seed rather than weakening the test.
 */
const REGION_SEED = 3;
const MIN_WET_COLUMNS = 500;
const regionParams = { ...params, seed: REGION_SEED };

/** Generates every chunk from the surface's lowest cave level to the canopy for a square of columns. */
function generateRegion(gen: TerrainGenerator, c0: number, c1: number, depthChunks = 1): World {
  const world = new World();
  for (let cz = c0; cz <= c1; cz++) {
    for (let cx = c0; cx <= c1; cx++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (let z = cz * S; z < (cz + 1) * S; z += 4) {
        for (let x = cx * S; x < (cx + 1) * S; x += 4) {
          const h = gen.surfaceY(x, z);
          lo = Math.min(lo, h);
          hi = Math.max(hi, h);
        }
      }
      const top = toChunkCoord(Math.max(hi, params.seaLevel) + params.trees.trunkMax + 4);
      for (let cy = toChunkCoord(lo) - depthChunks; cy <= top; cy++) {
        const g = gen.generate(cx, cy, cz);
        world.addChunk(Chunk.fromParts(cx, cy, cz, g.data, g.uniform, g.nonAir));
      }
    }
  }
  return world;
}

function sameChunk(a: ReturnType<TerrainGenerator['generate']>, b: ReturnType<TerrainGenerator['generate']>) {
  expect(a.uniform).toBe(b.uniform);
  expect(a.nonAir).toBe(b.nonAir);
  expect(a.data === null).toBe(b.data === null);
  if (a.data && b.data) expect(a.data).toEqual(b.data);
}

describe('TerrainGenerator', () => {
  const coords: [number, number, number][] = [];
  for (let cy = 0; cy <= 5; cy++) for (const [cx, cz] of [[0, 0], [-1, 3], [7, -2]] as const) coords.push([cx, cy, cz]);

  it('is deterministic: same seed → identical chunks, independent of cache state and order', () => {
    const a = new TerrainGenerator(params, config.world, 64);
    const b = new TerrainGenerator(params, config.world, 1);
    const resultsA = coords.map(([x, y, z]) => a.generate(x, y, z));
    const resultsB = [...coords].reverse().map(([x, y, z]) => b.generate(x, y, z)).reverse();
    resultsA.forEach((r, i) => sameChunk(r, resultsB[i]!));
  });

  it('produces a different world for a different seed', () => {
    const a = new TerrainGenerator(params, config.world);
    const b = new TerrainGenerator({ ...params, seed: params.seed + 1 }, config.world);
    let differs = 0;
    for (let x = 0; x < 256; x += 16) if (a.surfaceY(x, 0) !== b.surfaceY(x, 0)) differs++;
    expect(differs).toBeGreaterThan(8);
  });

  it('generates nothing outside the world bounds and keeps a solid floor', () => {
    const bounds = { minY: -40, maxY: 100 }; // deliberately not chunk-aligned
    const gen = new TerrainGenerator(params, bounds);
    expect(gen.generate(0, toChunkCoord(bounds.maxY) + 1, 0)).toMatchObject({ data: null, uniform: BlockId.air });
    expect(gen.generate(0, toChunkCoord(bounds.minY) - 1, 0)).toMatchObject({ data: null, uniform: BlockId.air });

    const bottom = gen.generate(0, toChunkCoord(bounds.minY), 0);
    const top = gen.generate(0, toChunkCoord(bounds.maxY - 1), 0);
    const at = (c: typeof bottom, cy: number, y: number, x = 5, z = 5) =>
      c.data ? c.data[((y - cy * S) << 10) | (z << 5) | x] : c.uniform;
    const cyB = toChunkCoord(bounds.minY);
    const cyT = toChunkCoord(bounds.maxY - 1);
    expect(at(bottom, cyB, bounds.minY - 1)).toBe(BlockId.air);
    for (let x = 0; x < S; x++) for (let z = 0; z < S; z++) expect(at(bottom, cyB, bounds.minY, x, z)).not.toBe(BlockId.air);
    expect(at(top, cyT, bounds.maxY)).toBe(BlockId.air);
  });

  it('has real relief: deep valleys under the sea level and high mountains', () => {
    const gen = new TerrainGenerator(params, config.world);
    const hs: number[] = [];
    for (let z = -4096; z < 4096; z += 64) for (let x = -4096; x < 4096; x += 64) hs.push(gen.surfaceY(x, z));
    expect(Math.min(...hs)).toBeLessThan(params.seaLevel - 20);
    expect(Math.max(...hs)).toBeGreaterThan(params.seaLevel + 120);
  });

  describe('in a generated region', () => {
    const gen = new TerrainGenerator(regionParams, config.world, 128);
    const C0 = -2;
    const C1 = 2;
    const world = generateRegion(gen, C0, C1, 3);
    const x0 = C0 * S;
    const x1 = (C1 + 1) * S - 1;

    it('applies surface rules: grass over 3-4 dirt over stone on gentle land, sand under water', () => {
      const { surface } = params;
      let grassColumns = 0;
      let wetColumns = 0;
      for (let z = x0 + 1; z < x1; z++) {
        for (let x = x0 + 1; x < x1; x++) {
          const h = gen.surfaceY(x, z);
          const slope = Math.max(...[[1, 0], [-1, 0], [0, 1], [0, -1]].map(([dx, dz]) => Math.abs(gen.surfaceY(x + dx!, z + dz!) - h)));
          const top = world.getBlock(x, h, z);
          if (h < params.seaLevel) {
            wetColumns++;
            expect(top).toBe(BlockId.sand);
            expect(world.getBlock(x, params.seaLevel, z)).toBe(BlockId.water);
            expect(world.getBlock(x, params.seaLevel + 1, z)).toBe(BlockId.air);
          } else if (h > params.seaLevel + surface.beachHeight && h < surface.rockAltitude && slope < surface.gravelSlope) {
            if (top === BlockId.dirt && world.getBlock(x, h + 1, z) === BlockId.oak_log) continue; // under a trunk
            grassColumns++;
            expect(top).toBe(BlockId.grass);
            for (let d = 1; d <= surface.dirtDepthMin; d++) {
              const b = world.getBlock(x, h - d, z);
              if (b !== BlockId.air) expect(b).toBe(BlockId.dirt); // air = cave tunnel
            }
            const deep = world.getBlock(x, h - surface.dirtDepthMax - 1, z);
            if (deep !== BlockId.air) expect(deep).toBe(BlockId.stone);
          } else if (slope >= surface.stoneSlope) {
            expect(top).toBe(BlockId.stone);
          }
        }
      }
      expect(grassColumns).toBeGreaterThan(100);
      // The pinned region contains a lake; without it the underwater rules above went unchecked.
      expect(wetColumns).toBeGreaterThan(MIN_WET_COLUMNS);
    });

    it('carves caves below the surface', () => {
      let caveAir = 0;
      for (let z = x0; z <= x1; z += 2) {
        for (let x = x0; x <= x1; x += 2) {
          const h = gen.surfaceY(x, z);
          for (let y = h - 3 * S; y < h - params.caves.cheeseMinDepth; y++) {
            if (world.hasChunk(toChunkCoord(x), toChunkCoord(y), toChunkCoord(z)) && world.getBlock(x, y, z) === BlockId.air) caveAir++;
          }
        }
      }
      expect(caveAir).toBeGreaterThan(0);
    });

    it('places whole trees, including canopies that cross chunk borders', () => {
      const R = params.trees.leafRadius;
      const trunks: { x: number; z: number; top: number }[] = [];
      for (let z = x0; z <= x1; z++) {
        for (let x = x0; x <= x1; x++) {
          const h = gen.surfaceY(x, z);
          if (world.getBlock(x, h + 1, z) !== BlockId.oak_log) continue;
          let top = h + 1;
          while (world.getBlock(x, top + 1, z) === BlockId.oak_log) top++;
          trunks.push({ x, z, top });
        }
      }
      expect(trunks.length).toBeGreaterThan(5);

      let crossing = 0;
      for (const t of trunks) {
        const len = t.top - gen.surfaceY(t.x, t.z);
        expect(len).toBeGreaterThanOrEqual(params.trees.trunkMin);
        expect(len).toBeLessThanOrEqual(params.trees.trunkMax);
        // The four side neighbours of the second-highest trunk block are always leaves (or another trunk).
        for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const b = world.getBlock(t.x + dx, t.top - 1, t.z + dz);
          const inRegion = t.x + dx >= x0 && t.x + dx <= x1 && t.z + dz >= x0 && t.z + dz <= x1;
          if (inRegion) expect([BlockId.oak_leaves, BlockId.oak_log]).toContain(b);
        }
        const trunkChunk = toChunkCoord(t.x);
        for (let dx = -R; dx <= R; dx++) {
          if (toChunkCoord(t.x + dx) !== trunkChunk && t.x + dx >= x0 && t.x + dx <= x1 &&
              world.getBlock(t.x + dx, t.top - 1, t.z) === BlockId.oak_leaves) {
            crossing++;
          }
        }
      }
      expect(crossing).toBeGreaterThan(0);

      // Every leaf belongs to some trunk within canopy reach.
      for (let z = x0 + R; z <= x1 - R; z++) {
        for (let x = x0 + R; x <= x1 - R; x++) {
          const h = gen.surfaceY(x, z);
          for (let y = h + 1; y <= h + params.trees.trunkMax + 4; y++) {
            if (world.getBlock(x, y, z) !== BlockId.oak_leaves) continue;
            const owner = trunks.some((t) => Math.abs(t.x - x) <= R && Math.abs(t.z - z) <= R && y >= t.top - 2 && y <= t.top + 1);
            expect(owner).toBe(true);
          }
        }
      }
    });
  });
});
