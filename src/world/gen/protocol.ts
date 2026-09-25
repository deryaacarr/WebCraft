import type { TerrainParams } from '../../config';
import { TerrainGenerator, type WorldBounds } from './terrain';

export type WorkerRequest =
  | { type: 'init'; generation: number; params: TerrainParams; bounds: WorldBounds; heightCacheColumns: number }
  | { type: 'generate'; generation: number; id: number; cx: number; cy: number; cz: number };

export type WorkerResponse =
  | {
      type: 'chunk';
      generation: number;
      id: number;
      /** Dense block data (transferred, not copied), or null for a uniform chunk. */
      data: ArrayBuffer | null;
      /** Bytes per block in `data`: 1 = Uint8Array, 2 = Uint16Array. */
      bytesPerBlock: 1 | 2;
      uniform: number;
      nonAir: number;
      /** Generation time inside the worker. */
      ms: number;
    }
  | { type: 'error'; generation: number; id: number; message: string };

export interface Reply {
  response: WorkerResponse;
  transfer: Transferable[];
}

/** Worker-side state machine, kept free of worker globals so it can be unit tested. */
export class WorkerHandler {
  private generator: TerrainGenerator | null = null;
  private generation = -1;

  handle(req: WorkerRequest): Reply | null {
    if (req.type === 'init') {
      this.generator = new TerrainGenerator(req.params, req.bounds, req.heightCacheColumns);
      this.generation = req.generation;
      return null;
    }
    const { id, cx, cy, cz, generation } = req;
    if (!this.generator || generation !== this.generation) {
      return { response: { type: 'error', generation, id, message: 'stale or uninitialised generation' }, transfer: [] };
    }
    try {
      const start = performance.now();
      const chunk = this.generator.generate(cx, cy, cz);
      const ms = performance.now() - start;
      const buffer = chunk.data ? (chunk.data.buffer as ArrayBuffer) : null;
      const bytesPerBlock = chunk.data?.BYTES_PER_ELEMENT === 2 ? 2 : 1;
      return {
        response: {
          type: 'chunk',
          generation,
          id,
          data: buffer,
          bytesPerBlock,
          uniform: chunk.uniform,
          nonAir: chunk.nonAir,
          ms,
        },
        transfer: buffer ? [buffer] : [],
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { response: { type: 'error', generation, id, message }, transfer: [] };
    }
  }
}
