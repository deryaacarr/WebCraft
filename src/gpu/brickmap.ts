import { config } from '../config';
import { toChunkCoord } from '../world/coords';
import type { World } from '../world/world';
import { BRICK_STRIDE_BYTES, EMPTY_FLAG } from './brick-pack';
import { BrickDistanceField, type CellBox } from './brick-distance';
import { brickGridSize, BrickStore, type BrickGridSize, type BrickSink } from './brick-store';

// BrickmapParams in brickmap.wgsl: origin vec3i, pad, size vec3u, min_brick_y i32,
// bounds_min vec3i, pad, bounds_max vec3i, pad.
const PARAMS_SIZE = 64;

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
  /** Bindings of brickmap.wgsl: 0 params, 1 grid, 2 pool. */
  static readonly ALL_BINDINGS = [0, 1, 2] as const;
  private readonly paramData = new ArrayBuffer(PARAMS_SIZE);
  private readonly size: BrickGridSize;
  private distance: BrickDistanceField | null = null;
  /** Grid cells written since the last distance update (cell coords, max inclusive). */
  private dirty: CellBox | null = null;

  constructor(private readonly device: GPUDevice) {
    const { world, brickmap } = config;
    const size = brickGridSize(brickmap.gridChunksXZ, world.minY, world.maxY);
    this.size = size;
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
      // Slots must stay below the pointer flag bits (2^30).
      maxPoolBricks: Math.min(Math.floor(maxBytes / BRICK_STRIDE_BYTES) - 1, EMPTY_FLAG - 1),
    });
  }

  /** Creates GPU-side helpers that need async pipeline compilation. */
  async init(): Promise<void> {
    this.distance = await BrickDistanceField.create(this.device, [this.size.x, this.size.y, this.size.z], this.grid);
  }

  /** Per-frame: follow the player, upload world changes within the time budget, then
   *  refresh the distance field around the cells that changed. */
  sync(world: World, x: number, z: number): void {
    this.store.setCenter(toChunkCoord(x), toChunkCoord(z), world);
    this.store.applyChanges(world.takeChanges());
    this.store.process(world, config.brickmap.uploadBudgetMs);
    this.flushDistance();
  }

  /** Recomputes the distance field for cells written so far (also used by debug tools). */
  flushDistance(): void {
    if (this.distance && this.dirty) this.distance.update(this.dirty);
    this.dirty = null;
  }

  /**
   * Bind group for @group(1). `bindings` lists the brickmap bindings the pipeline actually
   * uses: 'auto' layouts drop unused ones (e.g. a pass that never reads voxels has no pool).
   */
  bindGroup(layout: GPUBindGroupLayout, bindings: readonly number[] = GpuBrickmap.ALL_BINDINGS): GPUBindGroup {
    const cached = this.bindGroups.get(layout);
    if (cached && cached.version === this.version) return cached.group;
    const pool = this.pool;
    if (!pool) throw new Error('brick pool not allocated');
    const buffers = [this.params, this.grid, pool];
    const group = this.device.createBindGroup({
      label: 'brickmap',
      layout,
      entries: bindings.map((binding) => ({ binding, resource: { buffer: buffers[binding]! } })),
    });
    this.bindGroups.set(layout, { version: this.version, group });
    return group;
  }

  get memory(): BrickmapMemory {
    return {
      gridBytes: this.grid.size,
      poolBytes: this.pool?.size ?? 0,
      poolUsedBytes: this.store.stats.mixedBricks * BRICK_STRIDE_BYTES,
    };
  }

  // ------------------------------------------------------------ BrickSink

  writeGrid(cell: number, pointers: Uint32Array): void {
    this.device.queue.writeBuffer(this.grid, cell * Uint32Array.BYTES_PER_ELEMENT, pointers);
    if (!this.distance) return;
    const { x: sx, z: sz } = this.size;
    const x = cell % sx;
    const z = Math.floor(cell / sx) % sz;
    const y = Math.floor(cell / (sx * sz));
    const last = x + pointers.length - 1;
    const d = this.dirty;
    if (!d) {
      this.dirty = { min: [x, y, z], max: [last, y, z] };
    } else {
      d.min = [Math.min(d.min[0], x), Math.min(d.min[1], y), Math.min(d.min[2], z)];
      d.max = [Math.max(d.max[0], last), Math.max(d.max[1], y), Math.max(d.max[2], z)];
    }
  }

  writePool(slot: number, words: Uint32Array): void {
    if (!this.pool) return;
    this.device.queue.writeBuffer(this.pool, slot * BRICK_STRIDE_BYTES, words);
  }

  growPool(capacity: number): void {
    const next = this.device.createBuffer({
      label: `brick-pool (${capacity} bricks)`,
      size: capacity * BRICK_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const old = this.pool;
    if (old) {
      const encoder = this.device.createCommandEncoder({ label: 'brick-pool-grow' });
      encoder.copyBufferToBuffer(old, 0, next, 0, this.poolCapacity * BRICK_STRIDE_BYTES);
      this.device.queue.submit([encoder.finish()]);
      void this.device.queue.onSubmittedWorkDone().then(() => old.destroy());
      console.info(`[brickmap] pool grown to ${capacity} bricks (${((capacity * BRICK_STRIDE_BYTES) / 2 ** 20).toFixed(1)} MB)`);
    }
    this.pool = next;
    this.poolCapacity = capacity;
    this.version++;
  }

  setOrigin(origin: readonly [number, number, number]): void {
    new Int32Array(this.paramData).set(origin, 0);
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  setBounds(min: readonly [number, number, number], max: readonly [number, number, number]): void {
    const i32 = new Int32Array(this.paramData);
    i32.set(min, 8);
    i32.set(max, 12);
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }
}
