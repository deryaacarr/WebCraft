/**
 * Sub-voxel block shapes: a block may be an 8×8×8 grid of micro voxels instead of a full
 * cube. Each shaped block type has a few shape variants; a cell picks one variant and a
 * quarter turn about Y from a hash of its coordinates (same hash on the GPU, brickmap.wgsl),
 * so neighbouring rocks differ without storing anything per block.
 *
 * GPU table (`packShapeTable`, brickmap.wgsl `shape_table`):
 *   words 0..255          per block id: 0 = full cube, else (first shape << 8) | variant count
 *   words 256 + 16·s ..   shape s: 512 occupancy bits, bit (y·64 + z·8 + x)
 */
import { BlockId, type BlockName } from './blocks';

/** Micro voxels per block edge. */
export const MICRO = 8;
/** u32 words per shape (MICRO³ bits). */
export const SHAPE_WORDS = (MICRO * MICRO * MICRO) / 32;
/** Header entries: GPU voxels are 8-bit block ids. */
export const SHAPE_HEADER = 256;
/** Shapes the GPU table holds (a fixed-size uniform array, brickmap.wgsl SHAPE_TABLE_VEC4). */
export const MAX_SHAPES = 64;
export const SHAPE_TABLE_WORDS = SHAPE_HEADER + MAX_SHAPES * SHAPE_WORDS;

export type Shape = Uint32Array;

export function emptyShape(): Shape {
  return new Uint32Array(SHAPE_WORDS);
}

function bitIndex(x: number, y: number, z: number): number {
  return (y * MICRO + z) * MICRO + x;
}

export function shapeGet(s: Shape, x: number, y: number, z: number): boolean {
  const i = bitIndex(x, y, z);
  return ((s[i >>> 5]! >>> (i & 31)) & 1) === 1;
}

export function shapeSet(s: Shape, x: number, y: number, z: number): void {
  const i = bitIndex(x, y, z);
  s[i >>> 5]! |= 1 << (i & 31);
}

/** Shape from a predicate on micro voxel centres in [0, 1)³ block space. */
export function shapeFrom(inside: (x: number, y: number, z: number) => boolean): Shape {
  const s = emptyShape();
  for (let y = 0; y < MICRO; y++)
    for (let z = 0; z < MICRO; z++)
      for (let x = 0; x < MICRO; x++) if (inside((x + 0.5) / MICRO, (y + 0.5) / MICRO, (z + 0.5) / MICRO)) shapeSet(s, x, y, z);
  return s;
}

export function shapeCount(s: Shape): number {
  let n = 0;
  for (const w of s) for (let v = w; v; v &= v - 1) n++;
  return n;
}

/** Small deterministic PRNG for shape generation (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Half broken stone: the lower half of a block whose top was snapped off. The top is the
 * lowest of three tilted planes (flat facets meeting in sharp ridges — angular, never
 * rounded); one vertical plane chips a corner off above the bottom layer.
 */
function brokenStone(seed: number): Shape {
  const r = rng(seed);
  const planes = Array.from({ length: 3 }, () => ({ h: 0.42 + r() * 0.3, a: (r() - 0.5) * 1.6, b: (r() - 0.5) * 1.6 }));
  const corner = Math.floor(r() * 4);
  const cx = corner & 1 ? 1 : 0;
  const cz = corner & 2 ? 1 : 0;
  const chip = 0.3 + r() * 0.2;
  return shapeFrom((x, y, z) => {
    const top = Math.max(Math.min(...planes.map((p) => p.h + p.a * (x - 0.5) + p.b * (z - 0.5))), 0.25);
    if (y >= top) return false;
    // Corner chip: |x − corner| + |z − corner| below the chip size, not on the bottom layer.
    return y < 1 / MICRO || Math.abs(x - cx) + Math.abs(z - cz) > chip;
  });
}

/** Shape variants per shaped block type. */
export const BLOCK_SHAPES: Partial<Record<BlockName, readonly Shape[]>> = {
  broken_stone: [1, 2, 3, 4].map(brokenStone),
};

/** Same as brickmap.wgsl `shapeHash`. */
export function shapeHash(x: number, y: number, z: number): number {
  let h = (Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(z, 0xc2b2ae3d)) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h ^= h >>> 12;
  h = Math.imul(h, 0x297a2d39) >>> 0;
  h ^= h >>> 15;
  return h >>> 0;
}

const SHAPE_LIST: Shape[] = [];
/** Per block id: [first shape, count], or undefined for full cubes. */
const SHAPE_RANGE: ([number, number] | undefined)[] = [];
for (const [name, shapes] of Object.entries(BLOCK_SHAPES) as [BlockName, readonly Shape[]][]) {
  SHAPE_RANGE[BlockId[name]] = [SHAPE_LIST.length, shapes.length];
  SHAPE_LIST.push(...shapes);
}
if (SHAPE_LIST.length > MAX_SHAPES) throw new Error(`more than ${MAX_SHAPES} sub-voxel shapes`);

/**
 * Bit i set = block id i (< 32) is shaped. A pipeline override (brickmap.wgsl SHAPED_IDS),
 * so tracers test a constant instead of loading the table for every voxel they cross.
 */
export function shapedIdMask(): number {
  let mask = 0;
  SHAPE_RANGE.forEach((range, id) => {
    if (!range) return;
    if (id >= 32) throw new Error(`shaped block id ${id} >= 32 (SHAPED_IDS mask)`);
    mask |= 1 << id;
  });
  return mask >>> 0;
}

export function isShaped(id: number): boolean {
  return SHAPE_RANGE[id] !== undefined;
}

/**
 * Whether micro voxel (x, y, z) of the shaped block `id` at `cell` is solid: the cell's
 * variant, turned by (hash >> 30) quarter turns about Y. Matches brickmap.wgsl `microSolid`.
 */
export function microSolid(id: number, cell: readonly [number, number, number], x: number, y: number, z: number): boolean {
  const range = SHAPE_RANGE[id];
  if (!range) return true;
  const h = shapeHash(cell[0], cell[1], cell[2]);
  const s = SHAPE_LIST[range[0] + ((h & 0xffff) % range[1])]!;
  const m = MICRO - 1;
  switch (h >>> 30) {
    case 1:
      return shapeGet(s, z, y, m - x);
    case 2:
      return shapeGet(s, m - x, y, m - z);
    case 3:
      return shapeGet(s, m - z, y, x);
    default:
      return shapeGet(s, x, y, z);
  }
}

/** GPU shape table; `enabled` false = every block is a full cube (header all zero). */
export function packShapeTable(enabled: boolean): Uint32Array {
  const out = new Uint32Array(SHAPE_TABLE_WORDS);
  if (!enabled) return out;
  SHAPE_RANGE.forEach((range, id) => {
    if (range) out[id] = (range[0] << 8) | range[1];
  });
  SHAPE_LIST.forEach((s, i) => out.set(s, SHAPE_HEADER + i * SHAPE_WORDS));
  return out;
}
