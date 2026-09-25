import { config } from '../config';

const NS_PER_MS = 1e6;
const TIMESTAMP_BYTES = 8;

type Timestamps = GPUComputePassTimestampWrites & GPURenderPassTimestampWrites;

/**
 * Measures GPU time per pass with timestamp queries.
 *
 * Usage per frame: `beginFrame()`, pass `timestampWrites(name)` into each pass
 * descriptor, then `resolve(encoder)` before submit and `afterSubmit()` after.
 * Results arrive asynchronously a few frames later and are exponentially smoothed.
 * When 'timestamp-query' is unavailable every call is a no-op.
 */
export class GpuProfiler {
  readonly enabled: boolean;
  /** Smoothed GPU time per pass in milliseconds. */
  readonly timings = new Map<string, number>();
  /** Smoothed sum of all passes in milliseconds. */
  totalMs = 0;

  private readonly maxPasses: number;
  private readonly querySet: GPUQuerySet | null = null;
  private readonly resolveBuffer: GPUBuffer | null = null;
  private readonly freeReadbacks: GPUBuffer[] = [];
  private frameNames: string[] = [];
  private frameReadback: GPUBuffer | null = null;

  constructor(device: GPUDevice, supported: boolean) {
    this.enabled = supported;
    this.maxPasses = config.debug.profilerMaxPasses;
    if (!supported) return;

    const count = this.maxPasses * 2;
    const size = count * TIMESTAMP_BYTES;
    this.querySet = device.createQuerySet({ label: 'profiler', type: 'timestamp', count });
    this.resolveBuffer = device.createBuffer({
      label: 'profiler-resolve',
      size,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    for (let i = 0; i < config.debug.profilerReadbackBuffers; i++) {
      this.freeReadbacks.push(
        device.createBuffer({
          label: `profiler-readback-${i}`,
          size,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
      );
    }
  }

  beginFrame(): void {
    this.frameNames = [];
    // Skip measuring this frame if every readback buffer is still being mapped.
    this.frameReadback = this.enabled ? (this.freeReadbacks.pop() ?? null) : null;
  }

  /** Timestamp writes for a pass, or undefined when profiling is off for this frame. */
  timestampWrites(name: string): Timestamps | undefined {
    if (!this.querySet || !this.frameReadback || this.frameNames.length >= this.maxPasses) {
      return undefined;
    }
    const index = this.frameNames.length * 2;
    this.frameNames.push(name);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: index,
      endOfPassWriteIndex: index + 1,
    };
  }

  resolve(encoder: GPUCommandEncoder): void {
    const readback = this.frameReadback;
    if (!this.querySet || !this.resolveBuffer || !readback) return;
    if (this.frameNames.length === 0) {
      this.freeReadbacks.push(readback);
      this.frameReadback = null;
      return;
    }
    const count = this.frameNames.length * 2;
    encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, readback, 0, count * TIMESTAMP_BYTES);
  }

  /** Call after queue.submit(); starts the async readback for this frame. */
  afterSubmit(): void {
    const readback = this.frameReadback;
    const names = this.frameNames;
    this.frameReadback = null;
    if (!readback) return;
    if (names.length === 0) return;

    readback
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const times = new BigInt64Array(readback.getMappedRange(0, names.length * 2 * TIMESTAMP_BYTES));
        this.record(names, times);
        readback.unmap();
        this.freeReadbacks.push(readback);
      })
      .catch(() => {
        // Device lost or buffer destroyed; drop the sample.
      });
  }

  private record(names: string[], times: BigInt64Array): void {
    const alpha = 1 / Math.max(1, config.debug.profilerSmoothing);
    let total = 0;
    names.forEach((name, i) => {
      const begin = times[i * 2] ?? 0n;
      const end = times[i * 2 + 1] ?? 0n;
      // Some drivers return 0 or out-of-order values for a pass; treat as 0.
      const ms = end > begin ? Number(end - begin) / NS_PER_MS : 0;
      total += ms;
      const prev = this.timings.get(name);
      this.timings.set(name, prev === undefined ? ms : prev + (ms - prev) * alpha);
    });
    this.totalMs = this.totalMs === 0 ? total : this.totalMs + (total - this.totalMs) * alpha;
  }
}
