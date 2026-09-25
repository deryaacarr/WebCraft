import { config } from '../config';
import { toChunkCoord } from '../world/coords';
import type { World } from '../world/world';
import { BRICK_BYTES } from './brick-pack';
import { brickGridSize, BrickStore, type BrickSink } from './brick-store';

// BrickmapParams in brickmap.wgsl: origin vec3i, pad, size vec3u, min_brick_y i32.
const PARAMS_SIZE = 32;

export interface BrickmapMemory {
  gridBytes: number;
  poolBytes: number;
  poolUsedBytes: number;
}

/**
 * GPU side of the brickmap: owns the grid, pool and params buffers and applies the
 * writes produced by BrickStore. Consumers bind it at @group(1) via `bindGroup()`.
 */
export class GpuBrickmap implements BrickSink {
  readonly store: BrickStore;
  private readonly grid: GPUBuffer;
  private readonly params: GPUBuffer;
  private pool: GPUBuffer | null = null;
  private poolCapacity = 0;
  /** Bumped whenever a buffer is replaced; cached bind groups are rebuilt. */
  private version = 0;
  private readonly bindGroups = new Map<GPUBindGroupLayout, { version: number; group: GPUBindGroup }>();
  private readonly paramData = new ArrayBuffer(PARAMS_SIZE);

  constructor(private readonly device: GPUDevice) {
    const { world, brickmap } = config;
    const size = brickGridSize(brickmap.gridChunksXZ, world.minY, world.maxY);
    const streamedDiameter = 2 * (config.streaming.horizontalRadius + config.streaming.unloadMargin) + 1;
    if (streamedDiameter > brickmap.gridChunksXZ) {
      console.warn(
        `[brickmap] grid (${brickmap.gridChunksXZ} chunks) is narrower than the streamed area ` +
          `(${streamedDiameter} chunks); distant chunks will not reach the GPU.`,
      );
    }

    this.grid = device.createBuffer({
      label: 'brick-grid',
      size: size.x * size.y * size.z * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.params = device.createBuffer({
      label: 'brickmap-params',
      size: PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const u32 = new Uint32Array(this.paramData);
    const i32 = new Int32Array(this.paramData);
    u32[4] = size.x;
    u32[5] = size.y;
    u32[6] = size.z;
    i32[7] = size.minBrickY;

    const maxBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.store = new BrickStore(this, {
      grid: size,
      initialPoolBricks: brickmap.initialPoolBricks,
      poolGrowth: brickmap.poolGrowth,
      maxPoolBricks: Math.floor(maxBytes / BRICK_BYTES) - 1,
    });
  }

  /** Per-frame: follow the player, then upload world changes within the time budget. */
  sync(world: World, x: number, z: number): void {
    this.store.setCenter(toChunkCoord(x), toChunkCoord(z), world);
    this.store.applyChanges(world.takeChanges());
    this.store.process(world, config.brickmap.uploadBudgetMs);
  }

  bindGroup(layout: GPUBindGroupLayout): GPUBindGroup {
    const cached = this.bindGroups.get(layout);
    if (cached && cached.version === this.version) return cached.group;
    if (!this.pool) throw new Error('brick pool not allocated');
    const group = this.device.createBindGroup({
      label: 'brickmap',
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.grid } },
        { binding: 2, resource: { buffer: this.pool } },
      ],
    });
    this.bindGroups.set(layout, { version: this.version, group });
    return group;
  }

  get memory(): BrickmapMemory {
    return {
      gridBytes: this.grid.size,
      poolBytes: this.pool?.size ?? 0,
      poolUsedBytes: this.store.stats.mixedBricks * BRICK_BYTES,
    };
  }

  // ------------------------------------------------------------ BrickSink

  writeGrid(cell: number, pointers: Uint32Array): void {
    this.device.queue.writeBuffer(this.grid, cell * Uint32Array.BYTES_PER_ELEMENT, pointers);
  }

  writePool(slot: number, words: Uint32Array): void {
    if (!this.pool) return;
    this.device.queue.writeBuffer(this.pool, slot * BRICK_BYTES, words);
  }

  growPool(capacity: number): void {
    const next = this.device.createBuffer({
      label: `brick-pool (${capacity} bricks)`,
      size: capacity * BRICK_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const old = this.pool;
    if (old) {
      const encoder = this.device.createCommandEncoder({ label: 'brick-pool-grow' });
      encoder.copyBufferToBuffer(old, 0, next, 0, this.poolCapacity * BRICK_BYTES);
      this.device.queue.submit([encoder.finish()]);
      void this.device.queue.onSubmittedWorkDone().then(() => old.destroy());
      console.info(`[brickmap] pool grown to ${capacity} bricks (${((capacity * BRICK_BYTES) / 2 ** 20).toFixed(1)} MB)`);
    }
    this.pool = next;
    this.poolCapacity = capacity;
    this.version++;
  }

  setOrigin(origin: readonly [number, number, number]): void {
    const i32 = new Int32Array(this.paramData);
    i32[0] = origin[0];
    i32[1] = origin[1];
    i32[2] = origin[2];
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }
}
