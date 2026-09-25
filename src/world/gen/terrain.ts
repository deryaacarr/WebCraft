import FastNoiseLite from 'fastnoise-lite';
import type { TerrainParams } from '../../config';
import { BLOCKS, BlockId } from '../blocks';
import type { BlockArray } from '../chunk';
import { CHUNK_BITS, CHUNK_SIZE, CHUNK_VOLUME, chunkOrigin } from '../coords';
import { hash01, hash3 } from './hash';
import { Spline } from './spline';

export interface GeneratedChunk {
  /** Dense block data (8-bit while all ids fit), or null when every block equals `uniform`. */
  data: BlockArray | null;
  uniform: number;
  nonAir: number;
}

export interface Tree {
  x: number;
  z: number;
  /** Y of the ground block the trunk stands on. */
  base: number;
  trunkHeight: number;
}

/** Per chunk-column data shared by every chunk in that vertical stack. */
interface Column {
  /** Surface heights incl. a padding ring of `pad` columns (row-major, z outer). */
  heights: Int32Array;
  /** Trees whose canopy reaches into this column's XZ footprint. */
  trees: Tree[];
  minH: number;
  maxH: number;
  /** Lowest / highest Y occupied by any tree block (±Infinity if none). */
  treeBottom: number;
  treeTop: number;
}

// Independent hash streams for the different random decisions.
const SALT_TREE_X = 101;
const SALT_TREE_Z = 102;
const SALT_TREE_ACCEPT = 103;
const SALT_TRUNK = 104;
const SALT_LEAF = 105;
const SALT_DIRT = 106;
// Per-field seed offsets so the noise fields are uncorrelated.
const enum Field {
  Continentalness = 1,
  Erosion,
  PeaksValleys,
  Detail,
  WarpX,
  WarpZ,
  Cheese,
  SpaghettiA,
  SpaghettiB,
  Forest,
}

/** Vertical extent of the world: blocks outside [minY, maxY) are air. */
export interface WorldBounds {
  minY: number;
  maxY: number;
}

const S = CHUNK_SIZE;
// All registered ids fit in a byte → generate straight into 8-bit storage.
const DenseArray = BLOCKS.length <= 0x100 ? Uint8Array : Uint16Array;
const ROW = CHUNK_BITS;
const LAYER = 2 * CHUNK_BITS;

function makeNoise(
  seed: number,
  field: Field,
  frequency: number,
  octaves: number,
  fractal: 'FBm' | 'Ridged' = 'FBm',
): FastNoiseLite {
  const n = new FastNoiseLite((seed + field * 7919) | 0);
  n.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
  n.SetFrequency(frequency);
  if (octaves > 1 || fractal === 'Ridged') {
    n.SetFractalType(FastNoiseLite.FractalType[fractal]);
    n.SetFractalOctaves(octaves);
  }
  return n;
}

/**
 * Deterministic terrain: a pure function of (params, chunk coordinates).
 * Runs inside the terrain workers; the main thread only uses `surfaceY` (spawn).
 */
export class TerrainGenerator {
  private readonly continentalness: FastNoiseLite;
  private readonly erosion: FastNoiseLite;
  private readonly peaksValleys: FastNoiseLite;
  private readonly detail: FastNoiseLite;
  private readonly warpX: FastNoiseLite;
  private readonly warpZ: FastNoiseLite;
  private readonly cheese: FastNoiseLite;
  private readonly spaghettiA: FastNoiseLite;
  private readonly spaghettiB: FastNoiseLite;
  private readonly forest: FastNoiseLite;
  private readonly contSpline: Spline;
  private readonly erosionSpline: Spline;
  private readonly pvSpline: Spline;
  private readonly densitySpline: Spline;
  /** Padding ring around a chunk column: tree canopy reach + 1 for slope. */
  private readonly pad: number;
  private readonly padW: number;
  private readonly columns = new Map<string, Column>();

  constructor(
    private readonly p: TerrainParams,
    private readonly bounds: WorldBounds,
    private readonly cacheColumns = 1,
  ) {
    const { seed, caves } = p;
    this.continentalness = makeNoise(seed, Field.Continentalness, p.continentalness.frequency, p.continentalness.octaves);
    this.erosion = makeNoise(seed, Field.Erosion, p.erosion.frequency, p.erosion.octaves);
    this.peaksValleys = makeNoise(seed, Field.PeaksValleys, p.peaksValleys.frequency, p.peaksValleys.octaves, 'Ridged');
    this.detail = makeNoise(seed, Field.Detail, p.detail.frequency, p.detail.octaves);
    this.warpX = makeNoise(seed, Field.WarpX, p.warp.frequency, 1);
    this.warpZ = makeNoise(seed, Field.WarpZ, p.warp.frequency, 1);
    this.cheese = makeNoise(seed, Field.Cheese, caves.cheeseFrequency, 1);
    this.spaghettiA = makeNoise(seed, Field.SpaghettiA, caves.spaghettiFrequency, 1);
    this.spaghettiB = makeNoise(seed, Field.SpaghettiB, caves.spaghettiFrequency, 1);
    this.forest = makeNoise(seed, Field.Forest, p.trees.forestFrequency, 2);
    this.contSpline = new Spline(p.continentalnessSpline);
    this.erosionSpline = new Spline(p.erosionSpline);
    this.pvSpline = new Spline(p.peaksValleysSpline);
    this.densitySpline = new Spline(p.trees.densitySpline);
    this.pad = p.trees.leafRadius + 1;
    this.padW = S + 2 * this.pad;

    if (S % caves.latticeStep !== 0) {
      throw new Error(`caves.latticeStep (${caves.latticeStep}) must divide the chunk size (${S})`);
    }
  }

  /** Continuous terrain height at a column (before flooring). */
  heightAt(x: number, z: number): number {
    const { warp, detail } = this.p;
    const wx = x + warp.amplitude * this.warpX.GetNoise(x, z);
    const wz = z + warp.amplitude * this.warpZ.GetNoise(x, z);
    const c = this.continentalness.GetNoise(wx, wz);
    const e = this.erosion.GetNoise(wx, wz);
    // Ridged fractal: connected crest lines near +1, valleys between them.
    const pv = this.peaksValleys.GetNoise(wx, wz);
    return (
      this.contSpline.evaluate(c) +
      this.erosionSpline.evaluate(e) * this.pvSpline.evaluate(pv) +
      detail.amplitude * this.detail.GetNoise(x, z)
    );
  }

  /** Y of the topmost ground block at a column. */
  surfaceY(x: number, z: number): number {
    return Math.floor(this.heightAt(x, z));
  }

  generate(cx: number, cy: number, cz: number): GeneratedChunk {
    const { seaLevel } = this.p;
    const col = this.column(cx, cz);
    const y0 = chunkOrigin(cy);
    const y1 = y0 + S - 1;

    if (y0 > Math.max(col.maxH, seaLevel, col.treeTop)) return uniform(BlockId.air);
    if (y0 > col.maxH && y1 <= seaLevel) return uniform(BlockId.water);

    const { minY, maxY } = this.bounds;
    if (y1 < minY || y0 >= maxY) return uniform(BlockId.air);

    const data = new DenseArray(CHUNK_VOLUME);
    this.fillTerrain(data, col, cx, cz, y0);
    if (y0 <= col.maxH) this.carveCaves(data, col, cx, cz, y0);
    if (y1 >= col.treeBottom && y0 <= col.treeTop) this.placeTrees(data, col, cx, cz, y0);
    // A chunk straddling a bound: nothing exists outside the world.
    if (y0 < minY) data.fill(BlockId.air, 0, (minY - y0) << LAYER);
    if (y1 >= maxY) data.fill(BlockId.air, (maxY - y0) << LAYER);
    return compact(data);
  }

  // ---------------------------------------------------------------- columns

  private column(cx: number, cz: number): Column {
    const key = `${cx},${cz}`;
    const hit = this.columns.get(key);
    if (hit) {
      // LRU: re-insert to mark as most recently used.
      this.columns.delete(key);
      this.columns.set(key, hit);
      return hit;
    }
    const col = this.buildColumn(cx, cz);
    this.columns.set(key, col);
    if (this.columns.size > this.cacheColumns) {
      const oldest = this.columns.keys().next().value;
      if (oldest !== undefined) this.columns.delete(oldest);
    }
    return col;
  }

  private buildColumn(cx: number, cz: number): Column {
    const { pad, padW } = this;
    const x0 = chunkOrigin(cx);
    const z0 = chunkOrigin(cz);
    const heights = new Int32Array(padW * padW);
    let minH = Infinity;
    let maxH = -Infinity;
    for (let pz = 0; pz < padW; pz++) {
      for (let px = 0; px < padW; px++) {
        const h = this.surfaceY(x0 + px - pad, z0 + pz - pad);
        heights[pz * padW + px] = h;
        const inside = px >= pad && px < pad + S && pz >= pad && pz < pad + S;
        if (inside) {
          if (h < minH) minH = h;
          if (h > maxH) maxH = h;
        }
      }
    }
    const col: Column = { heights, trees: [], minH, maxH, treeBottom: Infinity, treeTop: -Infinity };
    this.findTrees(col, x0, z0);
    return col;
  }

  /** Height at world (x, z), which must lie inside the column's padded grid. */
  private h(col: Column, x0: number, z0: number, x: number, z: number): number {
    return col.heights[(z - z0 + this.pad) * this.padW + (x - x0 + this.pad)] ?? 0;
  }

  private slope(col: Column, x0: number, z0: number, x: number, z: number): number {
    const h = this.h(col, x0, z0, x, z);
    return Math.max(
      Math.abs(this.h(col, x0, z0, x + 1, z) - h),
      Math.abs(this.h(col, x0, z0, x - 1, z) - h),
      Math.abs(this.h(col, x0, z0, x, z + 1) - h),
      Math.abs(this.h(col, x0, z0, x, z - 1) - h),
    );
  }

  // ---------------------------------------------------------------- terrain

  private fillTerrain(data: BlockArray, col: Column, cx: number, cz: number, y0: number): void {
    const { seaLevel, seed, surface } = this.p;
    const x0 = chunkOrigin(cx);
    const z0 = chunkOrigin(cz);
    const dirtRange = surface.dirtDepthMax - surface.dirtDepthMin + 1;

    for (let lz = 0; lz < S; lz++) {
      for (let lx = 0; lx < S; lx++) {
        const x = x0 + lx;
        const z = z0 + lz;
        const h = this.h(col, x0, z0, x, z);
        const slope = this.slope(col, x0, z0, x, z);
        const dirtDepth = surface.dirtDepthMin + (hash3(seed, x, SALT_DIRT, z) % dirtRange);
        for (let ly = 0; ly < S; ly++) {
          const y = y0 + ly;
          const block =
            y > h ? (y <= seaLevel ? BlockId.water : BlockId.air)
            : this.groundBlock(h - y, h, slope, dirtDepth);
          data[(ly << LAYER) | (lz << ROW) | lx] = block;
        }
      }
    }
  }

  /** Surface rules for a ground block `depth` blocks below the column top `h`. */
  private groundBlock(depth: number, h: number, slope: number, dirtDepth: number): number {
    const { seaLevel, surface: s } = this.p;
    if (h <= seaLevel + s.beachHeight) return depth < s.sandDepth ? BlockId.sand : BlockId.stone;
    if (h >= s.rockAltitude || slope >= s.stoneSlope) return BlockId.stone;
    if (slope >= s.gravelSlope) return depth < s.gravelDepth ? BlockId.gravel : BlockId.stone;
    if (depth === 0) return BlockId.grass;
    return depth <= dirtDepth ? BlockId.dirt : BlockId.stone;
  }

  // ---------------------------------------------------------------- caves

  private carveCaves(data: BlockArray, col: Column, cx: number, cz: number, y0: number): void {
    const { caves, seaLevel } = this.p;
    const L = caves.latticeStep;
    const n = S / L + 1;
    const x0 = chunkOrigin(cx);
    const z0 = chunkOrigin(cz);

    // Noise on a coarse lattice aligned to world multiples of L.
    const cheese = new Float32Array(n * n * n);
    const spA = new Float32Array(n * n * n);
    const spB = new Float32Array(n * n * n);
    for (let iy = 0; iy < n; iy++) {
      const y = y0 + iy * L;
      for (let iz = 0; iz < n; iz++) {
        const z = z0 + iz * L;
        for (let ix = 0; ix < n; ix++) {
          const x = x0 + ix * L;
          const i = (iy * n + iz) * n + ix;
          cheese[i] = this.cheese.GetNoise(x, y * caves.cheeseVerticalScale, z);
          spA[i] = this.spaghettiA.GetNoise(x, y, z);
          spB[i] = this.spaghettiB.GetNoise(x, y, z);
        }
      }
    }

    const w = caves.spaghettiWidth;
    for (let lz = 0; lz < S; lz++) {
      for (let lx = 0; lx < S; lx++) {
        const x = x0 + lx;
        const z = z0 + lz;
        const h = this.h(col, x0, z0, x, z);
        if (y0 > h) continue;
        // Keep a solid roof under (or next to) water: there is no fluid simulation.
        const wet =
          h < seaLevel ||
          this.h(col, x0, z0, x + 1, z) < seaLevel ||
          this.h(col, x0, z0, x - 1, z) < seaLevel ||
          this.h(col, x0, z0, x, z + 1) < seaLevel ||
          this.h(col, x0, z0, x, z - 1) < seaLevel;
        const minSpaghetti = wet ? caves.underwaterMinDepth : 1;
        const minCheese = wet ? Math.max(caves.cheeseMinDepth, caves.underwaterMinDepth) : caves.cheeseMinDepth;
        const topLy = Math.min(S - 1, h - y0);

        const fx = (lx % L) / L;
        const fz = (lz % L) / L;
        const ix = (lx / L) | 0;
        const iz = (lz / L) | 0;
        // Never carve the world's bottom layer: caves must not open into the void.
        const bottomLy = Math.max(0, this.bounds.minY + 1 - y0);
        for (let ly = bottomLy; ly <= topLy; ly++) {
          const depth = h - (y0 + ly);
          if (depth < minSpaghetti) continue;
          const fy = (ly % L) / L;
          const base = (((ly / L) | 0) * n + iz) * n + ix;
          const carve =
            (depth >= minCheese && trilinear(cheese, base, n, fx, fy, fz) > caves.cheeseThreshold) ||
            (Math.abs(trilinear(spA, base, n, fx, fy, fz)) < w &&
              Math.abs(trilinear(spB, base, n, fx, fy, fz)) < w);
          if (carve) data[(ly << LAYER) | (lz << ROW) | lx] = BlockId.air;
        }
      }
    }
  }

  // ---------------------------------------------------------------- trees

  /** Collects trees (from any cell) whose canopy overlaps this chunk column. */
  private findTrees(col: Column, x0: number, z0: number): void {
    const { seed, seaLevel, surface, trees: t } = this.p;
    const R = t.leafRadius;
    const C = t.cellSize;
    const minX = x0 - R;
    const maxX = x0 + S - 1 + R;
    const minZ = z0 - R;
    const maxZ = z0 + S - 1 + R;
    const trunkRange = t.trunkMax - t.trunkMin + 1;

    for (let cj = Math.floor(minZ / C); cj <= Math.floor(maxZ / C); cj++) {
      for (let ci = Math.floor(minX / C); ci <= Math.floor(maxX / C); ci++) {
        const x = ci * C + Math.floor(hash01(seed, ci, SALT_TREE_X, cj) * C);
        const z = cj * C + Math.floor(hash01(seed, ci, SALT_TREE_Z, cj) * C);
        if (x < minX || x > maxX || z < minZ || z > maxZ) continue;

        const base = this.h(col, x0, z0, x, z);
        if (base <= seaLevel + surface.beachHeight || base >= t.treeline) continue;
        if (this.slope(col, x0, z0, x, z) > t.maxSlope) continue;
        const altitude = Math.min(1, (t.treeline - base) / t.treelineFade);
        const density = this.densitySpline.evaluate(this.forest.GetNoise(x, z)) * altitude;
        if (hash01(seed, x, SALT_TREE_ACCEPT, z) >= density) continue;

        const trunkHeight = t.trunkMin + (hash3(seed, x, SALT_TRUNK, z) % trunkRange);
        col.trees.push({ x, z, base, trunkHeight });
        // Blocks span the dirt under the trunk up to the canopy's top layer.
        col.treeBottom = Math.min(col.treeBottom, base);
        col.treeTop = Math.max(col.treeTop, base + trunkHeight + 1);
      }
    }
  }

  private placeTrees(data: BlockArray, col: Column, cx: number, cz: number, y0: number): void {
    const { seed, trees: t } = this.p;
    const x0 = chunkOrigin(cx);
    const z0 = chunkOrigin(cz);
    const R = t.leafRadius;
    const put = (x: number, y: number, z: number, id: number, over: readonly number[]) => {
      const lx = x - x0;
      const ly = y - y0;
      const lz = z - z0;
      if (lx < 0 || lx >= S || ly < 0 || ly >= S || lz < 0 || lz >= S) return;
      // Only above the ground: caves open air pockets that a canopy must not fill.
      if (id === BlockId.oak_leaves && y <= this.h(col, x0, z0, x, z)) return;
      const i = (ly << LAYER) | (lz << ROW) | lx;
      if (over.includes(data[i] ?? BlockId.air)) data[i] = id;
    };

    // Leaves first, trunks second: the result does not depend on tree order.
    for (const tree of col.trees) {
      const top = tree.base + tree.trunkHeight;
      for (let dy = -2; dy <= 1; dy++) {
        const r = dy >= 0 ? R - 1 : R;
        for (let dz = -r; dz <= r; dz++) {
          for (let dx = -r; dx <= r; dx++) {
            const corner = Math.abs(dx) === r && Math.abs(dz) === r;
            if (corner && (dy === 1 || hash01(seed, tree.x + dx, top + dy + SALT_LEAF, tree.z + dz) < 0.5)) {
              continue;
            }
            put(tree.x + dx, top + dy, tree.z + dz, BlockId.oak_leaves, [BlockId.air]);
          }
        }
      }
    }
    for (const tree of col.trees) {
      put(tree.x, tree.base, tree.z, BlockId.dirt, [BlockId.grass]);
      for (let y = tree.base + 1; y <= tree.base + tree.trunkHeight; y++) {
        put(tree.x, y, tree.z, BlockId.oak_log, [BlockId.air, BlockId.oak_leaves]);
      }
    }
  }
}

function uniform(id: number): GeneratedChunk {
  return { data: null, uniform: id, nonAir: id === BlockId.air ? 0 : CHUNK_VOLUME };
}

function compact(data: BlockArray): GeneratedChunk {
  const first = data[0] ?? BlockId.air;
  let same = true;
  let nonAir = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v !== first) same = false;
    if (v !== BlockId.air) nonAir++;
  }
  return same ? uniform(first) : { data, uniform: BlockId.air, nonAir };
}

/** Trilinear interpolation inside the lattice cell whose min corner is `base`. */
function trilinear(v: Float32Array, base: number, n: number, fx: number, fy: number, fz: number): number {
  const row = n;
  const layer = n * n;
  const c000 = v[base]!;
  const c100 = v[base + 1]!;
  const c010 = v[base + layer]!;
  const c110 = v[base + layer + 1]!;
  const c001 = v[base + row]!;
  const c101 = v[base + row + 1]!;
  const c011 = v[base + layer + row]!;
  const c111 = v[base + layer + row + 1]!;
  const x00 = c000 + (c100 - c000) * fx;
  const x01 = c001 + (c101 - c001) * fx;
  const x10 = c010 + (c110 - c010) * fx;
  const x11 = c011 + (c111 - c011) * fx;
  const y0 = x00 + (x10 - x00) * fy;
  const y1 = x01 + (x11 - x01) * fy;
  return y0 + (y1 - y0) * fz;
}
