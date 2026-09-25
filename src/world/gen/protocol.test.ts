import { describe, expect, it } from 'vitest';
import { config } from '../../config';
import { CHUNK_VOLUME, toChunkCoord } from '../coords';
import { WorkerHandler } from './protocol';
import { TerrainGenerator } from './terrain';

const params = config.terrain;

/** A chunk that straddles the surface at the origin, so it is guaranteed to be dense. */
function surfaceChunkY(): number {
  return toChunkCoord(new TerrainGenerator(params, config.world).surfaceY(0, 0));
}

describe('WorkerHandler', () => {
  it('returns dense data as a transferable ArrayBuffer (moved, not copied)', () => {
    const h = new WorkerHandler();
    expect(h.handle({ type: 'init', generation: 1, params, bounds: config.world, heightCacheColumns: 4 })).toBeNull();
    const reply = h.handle({ type: 'generate', generation: 1, id: 7, cx: 0, cy: surfaceChunkY(), cz: 0 });
    if (reply?.response.type !== 'chunk') throw new Error('expected a chunk');
    const buffer = reply.response.data;
    expect(buffer).toBeInstanceOf(ArrayBuffer);
    // All block ids fit in a byte, so the worker sends 8-bit data (half the transfer).
    expect(reply.response.bytesPerBlock).toBe(1);
    expect(buffer?.byteLength).toBe(CHUNK_VOLUME);
    expect(reply.transfer).toEqual([buffer]);

    // Same semantics as postMessage(msg, transfer): the sender's buffer is detached.
    const received = structuredClone(reply.response, { transfer: reply.transfer });
    expect(buffer?.byteLength).toBe(0);
    expect(received.type === 'chunk' && received.data?.byteLength).toBe(CHUNK_VOLUME);
  });

  it('sends uniform chunks without a buffer', () => {
    const h = new WorkerHandler();
    h.handle({ type: 'init', generation: 1, params, bounds: config.world, heightCacheColumns: 4 });
    const reply = h.handle({ type: 'generate', generation: 1, id: 1, cx: 0, cy: 40, cz: 0 });
    expect(reply?.response).toMatchObject({ type: 'chunk', data: null, nonAir: 0 });
    expect(reply?.transfer).toEqual([]);
  });

  it('rejects requests from a stale generation', () => {
    const h = new WorkerHandler();
    h.handle({ type: 'init', generation: 2, params, bounds: config.world, heightCacheColumns: 4 });
    const reply = h.handle({ type: 'generate', generation: 1, id: 1, cx: 0, cy: 0, cz: 0 });
    expect(reply?.response.type).toBe('error');
  });
});
