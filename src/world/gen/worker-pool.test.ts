import { describe, expect, it } from 'vitest';
import { config } from '../../config';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { WorkerHandler } from './protocol';
import { TerrainGenerator } from './terrain';
import { TerrainWorkerPool, type WorkerLike } from './worker-pool';

const params = config.terrain;

/** In-process stand-in for a Worker: same handler, async replies, real transfer semantics. */
class FakeWorker implements WorkerLike {
  static all: FakeWorker[] = [];
  readonly handler = new WorkerHandler();
  readonly received: WorkerRequest[] = [];
  /** When set, replies are held until release() is called. */
  hold = false;
  private held: (() => void)[] = [];
  private onMessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null;

  constructor() {
    FakeWorker.all.push(this);
  }

  postMessage(msg: WorkerRequest): void {
    this.received.push(msg);
    const reply = this.handler.handle(structuredClone(msg));
    if (!reply) return;
    const deliver = () => {
      const data = structuredClone(reply.response, { transfer: reply.transfer });
      this.onMessage?.({ data } as MessageEvent<WorkerResponse>);
    };
    if (this.hold) this.held.push(deliver);
    else setTimeout(deliver, 0);
  }

  release(): void {
    for (const d of this.held.splice(0)) d();
  }

  addEventListener(type: 'message' | 'error', listener: (e: never) => void): void {
    if (type === 'message') this.onMessage = listener as (e: MessageEvent<WorkerResponse>) => void;
  }

  terminate(): void {}
}

function makePool(count: number): { pool: TerrainWorkerPool; workers: FakeWorker[] } {
  FakeWorker.all = [];
  const pool = new TerrainWorkerPool(params, config.world, 8, count, () => new FakeWorker());
  return { pool, workers: FakeWorker.all };
}

const generates = (w: FakeWorker) => w.received.filter((m) => m.type === 'generate');
const signal = () => new AbortController().signal;

describe('TerrainWorkerPool', () => {
  it('returns the same chunk as direct generation', async () => {
    const { pool } = makePool(2);
    const cy = Math.floor(new TerrainGenerator(params, config.world).surfaceY(0, 0) / 32);
    const chunk = await pool.request(0, cy, 0, signal());
    const direct = new TerrainGenerator(params, config.world).generate(0, cy, 0);
    expect(chunk.nonAirCount).toBe(direct.nonAir);
    const expected = direct.data;
    if (!expected) throw new Error('expected a dense surface chunk');
    for (let i = 0; i < expected.length; i += 97) {
      const y = i >> 10;
      const z = (i >> 5) & 31;
      const x = i & 31;
      expect(chunk.get(x, y, z)).toBe(expected[i]);
    }
  });

  it('runs at most one job per worker and queues the rest FIFO', async () => {
    const { pool, workers } = makePool(2);
    const all = [0, 1, 2, 3, 4].map((i) => pool.request(i, 0, 0, signal()));
    expect(pool.stats).toMatchObject({ busy: 2, queued: 3 });
    expect(workers.flatMap(generates).map((m) => m.type === 'generate' && m.cx)).toEqual([0, 1]);
    const chunks = await Promise.all(all);
    expect(chunks.map((c) => c.cx)).toEqual([0, 1, 2, 3, 4]);
    expect(pool.stats).toMatchObject({ busy: 0, queued: 0 });
  });

  it('drops aborted queued jobs without dispatching them', async () => {
    const { pool, workers } = makePool(1);
    const first = pool.request(0, 0, 0, signal());
    const ctrl = new AbortController();
    const second = pool.request(1, 0, 0, ctrl.signal);
    ctrl.abort();
    await expect(second).rejects.toMatchObject({ name: 'AbortError' });
    await first;
    expect(generates(workers[0]!)).toHaveLength(1);
  });

  it('rejects in-flight jobs on reset and ignores their late results', async () => {
    const { pool, workers } = makePool(1);
    const w = workers[0]!;
    w.hold = true;
    const old = pool.request(0, 0, 0, signal());
    pool.reset({ ...params, seed: params.seed + 1 }, 8);
    await expect(old).rejects.toMatchObject({ name: 'AbortError' });
    w.hold = false;
    const next = pool.request(2, 0, 0, signal());
    w.release(); // stale reply for the old job arrives now
    const chunk = await next;
    expect(chunk.cx).toBe(2);
  });
});
