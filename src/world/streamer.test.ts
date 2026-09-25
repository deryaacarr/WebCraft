import { afterEach, describe, expect, it } from 'vitest';
import { config } from '../config';
import { Chunk } from './chunk';
import { CHUNK_SIZE } from './coords';
import { ChunkStreamer, sortedOffsets, type ChunkProvider } from './streamer';
import { World } from './world';

const saved = { streaming: { ...config.streaming }, world: { ...config.world } };
afterEach(() => {
  Object.assign(config.streaming, saved.streaming);
  Object.assign(config.world, saved.world);
});

/** World bounds covering chunk layers [cyMin, cyMax]. */
function layers(cyMin: number, cyMax: number): void {
  Object.assign(config.world, { minY: cyMin * CHUNK_SIZE, maxY: (cyMax + 1) * CHUNK_SIZE });
}

class RecordingProvider implements ChunkProvider {
  readonly order: string[] = [];
  async request(cx: number, cy: number, cz: number): Promise<Chunk> {
    this.order.push(`${cx},${cy},${cz}`);
    return new Chunk(cx, cy, cz);
  }
}

const flush = () => new Promise((r) => setTimeout(r, 0));

async function settle(streamer: ChunkStreamer, x: number, y: number, z: number): Promise<void> {
  for (let i = 0; i < 1000; i++) {
    streamer.update(x, y, z);
    await flush();
    const s = streamer.stats;
    if (s.missing === 0 && s.inFlight === 0) return;
  }
  throw new Error('streamer did not settle');
}

describe('sortedOffsets', () => {
  it('covers the cylinder, nearest first', () => {
    const offs = sortedOffsets(2, -1, 1);
    expect(offs[0]).toEqual([0, 0, 0]);
    // 13 columns inside radius 2 on XZ, 3 layers.
    expect(offs).toHaveLength(13 * 3);
    const d = offs.map(([x, y, z]) => x * x + y * y + z * z);
    expect(d).toEqual([...d].sort((a, b) => a - b));
    expect(offs.every(([x, , z]) => x * x + z * z <= 4)).toBe(true);
  });
});

describe('ChunkStreamer', () => {
  it('loads every chunk in range, nearest first, respecting maxInFlight', async () => {
    Object.assign(config.streaming, { horizontalRadius: 3, unloadMargin: 0, maxInFlight: 4 });
    layers(-1, 1);
    const world = new World();
    const provider = new RecordingProvider();
    const streamer = new ChunkStreamer(world, provider);

    streamer.update(0, 0, 0);
    expect(streamer.stats.inFlight).toBe(4);
    expect(provider.order[0]).toBe('0,0,0');

    await settle(streamer, 0, 0, 0);
    expect(world.chunkCount).toBe(sortedOffsets(3, -1, 1).length);
    expect(provider.order).toHaveLength(world.chunkCount); // no duplicate requests
  });

  it('uses the chunk containing a negative, fractional player position', async () => {
    Object.assign(config.streaming, { horizontalRadius: 0, unloadMargin: 0 });
    layers(-1, -1);
    const world = new World();
    const provider = new RecordingProvider();
    await settle(new ChunkStreamer(world, provider), -0.5, -0.5, -0.5);
    expect(provider.order).toEqual(['-1,-1,-1']);
  });

  it('unloads chunks outside radius + margin when the player moves', async () => {
    Object.assign(config.streaming, { horizontalRadius: 2, unloadMargin: 1 });
    layers(0, 0);
    const world = new World();
    const streamer = new ChunkStreamer(world, new RecordingProvider());
    await settle(streamer, 0, 0, 0);
    expect(world.hasChunk(-2, 0, 0)).toBe(true);

    // One chunk east: (-2,0,0) is now 3 away = radius + margin → kept.
    await settle(streamer, CHUNK_SIZE, 0, 0);
    expect(world.hasChunk(-2, 0, 0)).toBe(true);

    // Two chunks east: 4 away → unloaded.
    await settle(streamer, 2 * CHUNK_SIZE, 0, 0);
    expect(world.hasChunk(-2, 0, 0)).toBe(false);
    expect(world.hasChunk(4, 0, 0)).toBe(true);
  });

  it('loads the whole column between the world bounds, wherever the player is', async () => {
    Object.assign(config.streaming, { horizontalRadius: 0, unloadMargin: 0 });
    layers(-2, 9);
    for (const playerY of [-10_000, 0, 150, 10_000]) {
      const provider = new RecordingProvider();
      await settle(new ChunkStreamer(new World(), provider), 0, playerY, 0);
      const ys = provider.order.map((k) => Number(k.split(',')[1])).sort((a, b) => a - b);
      expect(ys).toEqual([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
  });

  it('requests the layers nearest to the player first', async () => {
    Object.assign(config.streaming, { horizontalRadius: 0, unloadMargin: 0, maxInFlight: 64 });
    layers(-2, 9);
    const provider = new RecordingProvider();
    new ChunkStreamer(new World(), provider).update(0, 6 * CHUNK_SIZE + 1, 0);
    expect(provider.order.slice(0, 3)).toEqual(['0,6,0', '0,5,0', '0,7,0']);
  });

  it('unloads chunks that fall outside new world bounds', async () => {
    Object.assign(config.streaming, { horizontalRadius: 0, unloadMargin: 0 });
    layers(0, 3);
    const world = new World();
    const streamer = new ChunkStreamer(world, new RecordingProvider());
    await settle(streamer, 0, 0, 0);
    expect(world.chunkCount).toBe(4);
    layers(0, 1);
    await settle(streamer, 0, 0, 0);
    expect(world.hasChunk(0, 3, 0)).toBe(false);
    expect(world.chunkCount).toBe(2);
  });

  it('drops results that arrive after the chunk went out of range', async () => {
    Object.assign(config.streaming, { horizontalRadius: 0, unloadMargin: 0 });
    layers(0, 0);
    let resolve: ((c: Chunk) => void) | undefined;
    const provider: ChunkProvider = {
      request: (cx, cy, cz) => new Promise((r) => { resolve = () => r(new Chunk(cx, cy, cz)); }),
    };
    const world = new World();
    const streamer = new ChunkStreamer(world, provider);
    streamer.update(0, 0, 0);
    streamer.update(10 * CHUNK_SIZE, 0, 0); // moves away, aborts (0,0,0)
    resolve?.(new Chunk(0, 0, 0));
    await flush();
    expect(world.hasChunk(0, 0, 0)).toBe(false);
  });
});
