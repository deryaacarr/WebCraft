import { BLOCKS } from './blocks';
import type { Chunk } from './chunk';
import { CHUNK_SIZE } from './coords';
import type { WorldChanges } from './world';

/** Block ids that emit light. */
const EMISSIVE_IDS = BLOCKS.filter((b) => b.emissive > 0).map((b) => b.id);
const EMISSIVE = Uint8Array.from(BLOCKS, (b) => (b.emissive > 0 ? 1 : 0));

/** One emissive block: absolute cell and block id. */
export interface Emitter {
  x: number;
  y: number;
  z: number;
  id: number;
}

/**
 * Emissive blocks of the loaded world (light sources for ReSTIR DI). Mirrors world
 * changes: a changed chunk is rescanned (`indexOf` per emissive id, so chunks without
 * emitters cost a few native scans). `nearest` picks the light list for the GPU.
 */
export class EmitterRegistry {
  private readonly byChunk = new Map<string, Emitter[]>();
  private count = 0;
  /** Bumped whenever the set of emitters changes. */
  version = 0;

  get total(): number {
    return this.count;
  }

  apply(changes: WorldChanges): void {
    for (const key of changes.removed) this.set(key, []);
    for (const { chunk } of changes.changed) this.set(chunk.key, scanChunk(chunk));
  }

  clear(): void {
    this.byChunk.clear();
    this.count = 0;
    this.version++;
  }

  /** Up to `max` emitters within `radius` of (x, y, z), nearest first. */
  nearest(x: number, y: number, z: number, radius: number, max: number): Emitter[] {
    const r2 = radius * radius;
    const found: { e: Emitter; d: number }[] = [];
    for (const list of this.byChunk.values()) {
      for (const e of list) {
        const dx = e.x + 0.5 - x;
        const dy = e.y + 0.5 - y;
        const dz = e.z + 0.5 - z;
        const d = dx * dx + dy * dy + dz * dz;
        if (d <= r2) found.push({ e, d });
      }
    }
    found.sort((a, b) => a.d - b.d);
    return found.slice(0, max).map((f) => f.e);
  }

  private set(key: string, list: Emitter[]): void {
    const prev = this.byChunk.get(key);
    if (!prev && list.length === 0) return;
    this.count += list.length - (prev?.length ?? 0);
    if (list.length) this.byChunk.set(key, list);
    else this.byChunk.delete(key);
    this.version++;
  }
}

/** Emissive blocks of one chunk (a uniform chunk of an emissive block is ignored: it is
 *  buried or a lava sea; its surface blocks light the scene through GI bounces). */
export function scanChunk(chunk: Chunk): Emitter[] {
  const data = chunk.denseData;
  if (!data) return [];
  const out: Emitter[] = [];
  const x0 = chunk.cx * CHUNK_SIZE;
  const y0 = chunk.cy * CHUNK_SIZE;
  const z0 = chunk.cz * CHUNK_SIZE;
  for (const id of EMISSIVE_IDS) {
    for (let i = data.indexOf(id); i !== -1; i = data.indexOf(id, i + 1)) {
      // Local index layout: y << 10 | z << 5 | x (chunk.ts).
      const lx = i & (CHUNK_SIZE - 1);
      const lz = (i / CHUNK_SIZE) & (CHUNK_SIZE - 1);
      const ly = Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE));
      out.push({ x: x0 + lx, y: y0 + ly, z: z0 + lz, id });
    }
  }
  return out;
}

export function isEmissive(id: number): boolean {
  return EMISSIVE[id] === 1;
}
