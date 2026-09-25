import { config } from '../config';
import type { Chunk } from './chunk';
import { chunkKey, toChunkCoord, worldChunkYRange } from './coords';
import type { World } from './world';

/** Produces chunk contents (terrain generator, save file…). May be async / worker-backed. */
export interface ChunkProvider {
  request(cx: number, cy: number, cz: number, signal: AbortSignal): Promise<Chunk>;
}

type Offset = readonly [dx: number, dy: number, dz: number];

export interface StreamerStats {
  loaded: number;
  inFlight: number;
  /** Chunks inside the load radius that are not loaded yet. */
  missing: number;
}

/**
 * Keeps the chunks inside a cylinder around the player loaded: circular radius on XZ,
 * and vertically the full world column between world.minY and world.maxY (chunks
 * outside those bounds are never requested). Requests are issued nearest-first and
 * capped at `maxInFlight`; chunks beyond radius + unloadMargin are unloaded (the margin
 * prevents thrashing when the player walks back and forth across a chunk border).
 */
export class ChunkStreamer {
  private center: Offset | null = null;
  /** Inputs the current `offsets` were built from; rebuilt when they change. */
  private offsetsKey = '';
  private offsets: Offset[] = [];
  /** Settings the last unload pass used; a change triggers a new pass. */
  private keepKey = '';
  /** Next position in `offsets` to consider; everything before it is loaded or in flight. */
  private cursor = 0;
  private readonly inFlight = new Map<string, AbortController>();

  constructor(
    private readonly world: World,
    private readonly provider: ChunkProvider,
  ) {}

  /** Call every tick with the player position in world units. */
  update(x: number, y: number, z: number): void {
    const cx = toChunkCoord(x);
    const cy = toChunkCoord(y);
    const cz = toChunkCoord(z);
    const h = config.streaming.horizontalRadius;
    const [cyMin, cyMax] = worldChunkYRange();
    // Offsets are relative to the player, so their vertical span shifts with the player's layer.
    const offsetsKey = `${h},${cyMin - cy},${cyMax - cy}`;

    const c = this.center;
    const moved = !c || c[0] !== cx || c[1] !== cy || c[2] !== cz;
    const rebuilt = offsetsKey !== this.offsetsKey;
    if (rebuilt) {
      this.offsets = sortedOffsets(h, cyMin - cy, cyMax - cy);
      this.offsetsKey = offsetsKey;
    }
    if (moved || rebuilt || this.keepKey !== this.currentKeepKey()) {
      this.center = [cx, cy, cz];
      this.keepKey = this.currentKeepKey();
      this.cursor = 0;
      this.unloadOutOfRange();
    }
    this.issueRequests();
  }

  private currentKeepKey(): string {
    const { horizontalRadius, unloadMargin } = config.streaming;
    const { minY, maxY } = config.world;
    return `${horizontalRadius},${unloadMargin},${minY},${maxY}`;
  }

  get stats(): StreamerStats {
    let missing = 0;
    const c = this.center;
    if (c) {
      for (const [dx, dy, dz] of this.offsets) {
        if (!this.world.hasChunk(c[0] + dx, c[1] + dy, c[2] + dz)) missing++;
      }
    }
    return { loaded: this.world.chunkCount, inFlight: this.inFlight.size, missing };
  }

  /** Aborts all pending requests. */
  dispose(): void {
    for (const ctrl of this.inFlight.values()) ctrl.abort();
    this.inFlight.clear();
  }

  /** Aborts pending requests and forgets the center; the next update() starts from scratch. */
  reset(): void {
    this.dispose();
    this.center = null;
    this.keepKey = '';
    this.cursor = 0;
  }

  private inKeepRange(cx: number, cy: number, cz: number): boolean {
    const c = this.center;
    if (!c) return false;
    const { horizontalRadius: h, unloadMargin: m } = config.streaming;
    const [cyMin, cyMax] = worldChunkYRange();
    const dx = cx - c[0];
    const dz = cz - c[2];
    return dx * dx + dz * dz <= (h + m) * (h + m) && cy >= cyMin && cy <= cyMax;
  }

  private unloadOutOfRange(): void {
    const drop: Chunk[] = [];
    for (const chunk of this.world.chunkValues()) {
      if (!this.inKeepRange(chunk.cx, chunk.cy, chunk.cz)) drop.push(chunk);
    }
    for (const chunk of drop) this.world.removeChunk(chunk.cx, chunk.cy, chunk.cz);

    for (const [key, ctrl] of this.inFlight) {
      const [x, y, z] = key.split(',').map(Number) as [number, number, number];
      if (!this.inKeepRange(x, y, z)) {
        ctrl.abort();
        this.inFlight.delete(key);
      }
    }
  }

  private issueRequests(): void {
    const c = this.center;
    if (!c) return;
    const max = config.streaming.maxInFlight;
    while (this.inFlight.size < max && this.cursor < this.offsets.length) {
      const off = this.offsets[this.cursor++];
      if (!off) break;
      const cx = c[0] + off[0];
      const cy = c[1] + off[1];
      const cz = c[2] + off[2];
      const key = chunkKey(cx, cy, cz);
      if (this.inFlight.has(key) || this.world.hasChunk(cx, cy, cz)) continue;
      this.request(cx, cy, cz, key);
    }
  }

  private request(cx: number, cy: number, cz: number, key: string): void {
    const ctrl = new AbortController();
    this.inFlight.set(key, ctrl);
    this.provider
      .request(cx, cy, cz, ctrl.signal)
      .then((chunk) => {
        if (ctrl.signal.aborted || this.inFlight.get(key) !== ctrl) return;
        this.inFlight.delete(key);
        if (this.inKeepRange(cx, cy, cz)) this.world.addChunk(chunk);
      })
      .catch((err: unknown) => {
        if (this.inFlight.get(key) === ctrl) this.inFlight.delete(key);
        // Not retried until the player changes chunk (cursor reset), to avoid hot loops.
        if (!ctrl.signal.aborted) console.error(`[streamer] chunk ${key} failed:`, err);
      });
  }
}

/** All offsets inside the load cylinder (radius h on XZ, dy in [dyMin, dyMax]), nearest first. */
export function sortedOffsets(h: number, dyMin: number, dyMax: number): Offset[] {
  const out: Offset[] = [];
  for (let dy = dyMin; dy <= dyMax; dy++) {
    for (let dz = -h; dz <= h; dz++) {
      for (let dx = -h; dx <= h; dx++) {
        if (dx * dx + dz * dz <= h * h) out.push([dx, dy, dz]);
      }
    }
  }
  const dist = ([dx, dy, dz]: Offset) => dx * dx + dy * dy + dz * dz;
  return out.sort((a, b) => dist(a) - dist(b));
}
