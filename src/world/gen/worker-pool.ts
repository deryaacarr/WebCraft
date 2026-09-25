import type { TerrainParams } from '../../config';
import { Chunk } from '../chunk';
import type { ChunkProvider } from '../streamer';
import type { WorkerRequest, WorkerResponse } from './protocol';
import type { WorldBounds } from './terrain';

/** Minimal worker surface the pool needs; lets tests substitute an in-process fake. */
export interface WorkerLike {
  postMessage(message: WorkerRequest): void;
  addEventListener(type: 'message', listener: (e: MessageEvent<WorkerResponse>) => void): void;
  addEventListener(type: 'error', listener: (e: ErrorEvent) => void): void;
  terminate(): void;
}

interface Job {
  id: number;
  cx: number;
  cy: number;
  cz: number;
  resolve(chunk: Chunk): void;
  reject(err: unknown): void;
  /** Set when the caller aborted while the job was already running. */
  cancelled: boolean;
  cleanup(): void;
}

interface Slot {
  worker: WorkerLike;
  job: Job | null;
}

export interface PoolStats {
  workers: number;
  busy: number;
  queued: number;
  /** Rolling average generation time per chunk inside a worker (ms). */
  avgChunkMs: number;
}

const AVG_WINDOW = 64;

function abortError(): DOMException {
  return new DOMException('Chunk request aborted', 'AbortError');
}

export function defaultWorkerCount(configured: number): number {
  if (configured > 0) return configured;
  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4;
  return Math.max(1, cores - 1);
}

/**
 * Chunk provider backed by a pool of terrain workers. Jobs run FIFO (the streamer
 * already submits nearest-first), one per worker at a time.
 */
export class TerrainWorkerPool implements ChunkProvider {
  private readonly slots: Slot[] = [];
  private readonly queue: Job[] = [];
  private generation = 0;
  private nextId = 1;
  private avgMs = 0;

  constructor(
    params: TerrainParams,
    private readonly bounds: WorldBounds,
    heightCacheColumns: number,
    count: number,
    createWorker: () => WorkerLike = () =>
      new Worker(new URL('./terrain.worker.ts', import.meta.url), { type: 'module', name: 'terrain' }),
  ) {
    for (let i = 0; i < count; i++) {
      const slot: Slot = { worker: createWorker(), job: null };
      slot.worker.addEventListener('message', (e) => this.onMessage(slot, e.data));
      slot.worker.addEventListener('error', (e) => this.onWorkerError(slot, e));
      this.slots.push(slot);
    }
    this.init(params, heightCacheColumns);
  }

  get stats(): PoolStats {
    return {
      workers: this.slots.length,
      busy: this.slots.filter((s) => s.job).length,
      queued: this.queue.length,
      avgChunkMs: this.avgMs,
    };
  }

  request(cx: number, cy: number, cz: number, signal: AbortSignal): Promise<Chunk> {
    if (signal.aborted) return Promise.reject(abortError());
    return new Promise<Chunk>((resolve, reject) => {
      const job: Job = {
        id: this.nextId++,
        cx,
        cy,
        cz,
        resolve,
        reject,
        cancelled: false,
        cleanup: () => signal.removeEventListener('abort', onAbort),
      };
      const onAbort = () => {
        const i = this.queue.indexOf(job);
        if (i >= 0) this.queue.splice(i, 1);
        else job.cancelled = true; // running: the worker finishes, the result is dropped
        job.cleanup();
        reject(abortError());
      };
      signal.addEventListener('abort', onAbort, { once: true });
      this.queue.push(job);
      this.pump();
    });
  }

  /** Switches every worker to new terrain params; queued and running jobs are rejected. */
  reset(params: TerrainParams, heightCacheColumns: number): void {
    for (const job of this.queue.splice(0)) {
      job.cleanup();
      job.reject(abortError());
    }
    for (const slot of this.slots) {
      if (slot.job && !slot.job.cancelled) {
        slot.job.cancelled = true;
        slot.job.cleanup();
        slot.job.reject(abortError());
      }
    }
    this.init(params, heightCacheColumns);
  }

  dispose(): void {
    for (const job of this.queue.splice(0)) job.reject(abortError());
    for (const slot of this.slots) {
      slot.job?.reject(abortError());
      slot.worker.terminate();
    }
    this.slots.length = 0;
  }

  private init(params: TerrainParams, heightCacheColumns: number): void {
    this.generation++;
    this.avgMs = 0;
    for (const slot of this.slots) {
      slot.worker.postMessage({
        type: 'init',
        generation: this.generation,
        params,
        bounds: this.bounds,
        heightCacheColumns,
      });
    }
    this.pump();
  }

  private pump(): void {
    for (const slot of this.slots) {
      if (slot.job) continue;
      const job = this.queue.shift();
      if (!job) return;
      slot.job = job;
      slot.worker.postMessage({
        type: 'generate',
        generation: this.generation,
        id: job.id,
        cx: job.cx,
        cy: job.cy,
        cz: job.cz,
      });
    }
  }

  private onMessage(slot: Slot, msg: WorkerResponse): void {
    const job = slot.job;
    if (!job || job.id !== msg.id) return;
    slot.job = null;

    if (!job.cancelled && msg.generation === this.generation) {
      job.cleanup();
      if (msg.type === 'chunk') {
        this.avgMs += (msg.ms - this.avgMs) / (this.avgMs === 0 ? 1 : AVG_WINDOW);
        const data = !msg.data ? null : msg.bytesPerBlock === 2 ? new Uint16Array(msg.data) : new Uint8Array(msg.data);
        job.resolve(Chunk.fromParts(job.cx, job.cy, job.cz, data, msg.uniform, msg.nonAir));
      } else {
        job.reject(new Error(msg.message));
      }
    }
    this.pump();
  }

  private onWorkerError(slot: Slot, e: ErrorEvent): void {
    console.error('[terrain worker]', e.message || e);
    const job = slot.job;
    slot.job = null;
    if (job && !job.cancelled) {
      job.cleanup();
      job.reject(new Error(`Terrain worker failed: ${e.message}`));
    }
    this.pump();
  }
}
