import { config } from '../../config';
import { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import type { MaterialSystem } from '../materials';
import { clearMicroFlags, createMicroFlags, createMicroPipelines, microFlagDummy } from '../micro-fixup';
import { traceConstants } from '../trace-constants';
import type { WaterSystem } from '../water';
import type { FrameContext, RenderPass } from './pass';

// Camera struct in camera.wgsl: 3 × mat4x4f (192 B) + 6 × 16 B.
const CAMERA_SIZE = 288;
const DEG = Math.PI / 180;
/** u32 words before the tile marks in the tile list buffer (primary.wgsl TILE_HEADER). */
const TILE_HEADER = 4;
/** Frames between read-backs of "water seen" (idle frame rate policy). */
const WATER_SEEN_INTERVAL = 10;

/**
 * Traces one ray per pixel through the brickmap and fills the G-buffer. A depth prepass
 * at 1/prepassTile resolution first finds, per tile, how far all its rays can safely skip.
 * Pixels whose ray stopped at a shaped block or water are redone by a fix-up pipeline
 * (micro-fixup.ts; water: refraction to the bottom, water.wgsl).
 */
export class PrimaryPass implements RenderPass {
  readonly name = 'primary';

  private pipeline!: GPUComputePipeline;
  private fixup: GPUComputePipeline | null = null;
  private fixupBindGroup: GPUBindGroup | null = null;
  /** Camera under water: replaces the main pipeline and fix-ups (primary.wgsl UNDERWATER). */
  private underwater!: GPUComputePipeline;
  private underwaterGroup: GPUBindGroup | null = null;
  private underwaterLayouts!: [GPUBindGroupLayout, GPUBindGroupLayout];
  private fixupLayouts: [GPUBindGroupLayout, GPUBindGroupLayout] | null = null;
  private flags: GPUTexture | null = null;
  /** Water surface under the waves (primary.wgsl `surface`, read by the fix-up). */
  private waterSurface: GPUTexture | null = null;
  private surfacePipeline!: GPUComputePipeline;
  private surfaceGroup: GPUBindGroup | null = null;
  /** Its brickmap and material layouts, looked up once (getBindGroupLayout() returns a new
   *  object per call, which would grow the bind group caches keyed by layout). */
  private surfaceLayouts!: [GPUBindGroupLayout, GPUBindGroupLayout];
  /** Tiles with flagged pixels (primary.wgsl micro_tiles): indirect args, marks, list. */
  private tiles: GPUBuffer | null = null;
  private tileCount = 0;
  /** Water seen (tile list word 3), copied for reading back every few frames. */
  private readonly waterSeen: GPUBuffer;
  /** 'copied': recorded in a frame, to map after its submit; 'mapping': read in flight. */
  private waterSeenState: 'idle' | 'copied' | 'mapping' = 'idle';
  private waterSeenFrame = 0;
  /** Whether water was on screen when last read back (a few frames late). */
  waterVisible = false;
  /** Stand-in for the tile binding a pipeline does not use. */
  private readonly dummyTiles: GPUBuffer;
  private prepassPipeline!: GPUComputePipeline;
  private prepassBindGroup: GPUBindGroup | null = null;
  private coarse: GPUTexture | null = null;
  private brickmapLayout!: GPUBindGroupLayout;
  private prepassLayout!: GPUBindGroupLayout;
  /** Camera uniform (camera.wgsl), shared with the lighting passes. */
  camera!: GPUBuffer;
  private materialLayout!: GPUBindGroupLayout;
  private bindGroup: GPUBindGroup | null = null;
  private readonly cameraData = new ArrayBuffer(CAMERA_SIZE);
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly brickmap: GpuBrickmap,
    private readonly gbuffer: GBuffer,
    private readonly materials: MaterialSystem,
    private readonly water: WaterSystem,
  ) {
    this.dummyTiles = device.createBuffer({ label: `${this.name}-micro-tiles-dummy`, size: 16, usage: GPUBufferUsage.STORAGE });
    this.waterSeen = device.createBuffer({ label: `${this.name}-water-seen`, size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const [module, prepassModule] = await Promise.all([
      createShaderModule(this.device, 'primary.wgsl'),
      createShaderModule(this.device, 'prepass.wgsl'),
    ]);
    const [wx, wy] = this.workgroup;
    const constants = { WORKGROUP_X: wx, WORKGROUP_Y: wy, ...traceConstants() };
    let pipelines;
    [pipelines, this.prepassPipeline, this.surfacePipeline, this.underwater] = await Promise.all([
      createMicroPipelines(this.device, this.name, module, constants, true),
      this.device.createComputePipelineAsync({
        label: `${this.name}-prepass`,
        layout: 'auto',
        compute: { module: prepassModule, entryPoint: 'main', constants },
      }),
      this.device.createComputePipelineAsync({
        label: `${this.name}-water-surface`,
        layout: 'auto',
        compute: { module, entryPoint: 'surface', constants: { ...constants, MICRO_WALK: 1 } },
      }),
      this.device.createComputePipelineAsync({
        label: `${this.name}-underwater`,
        layout: 'auto',
        compute: { module, entryPoint: 'main', constants: { ...constants, MICRO_WALK: 1, UNDERWATER: 1 } },
      }),
    ]);
    this.pipeline = pipelines.main;
    this.fixup = pipelines.fixup;
    this.fixupLayouts = this.fixup && [this.fixup.getBindGroupLayout(1), this.fixup.getBindGroupLayout(2)];
    this.prepassLayout = this.prepassPipeline.getBindGroupLayout(1);
    this.surfaceLayouts = [this.surfacePipeline.getBindGroupLayout(1), this.surfacePipeline.getBindGroupLayout(2)];
    this.underwaterLayouts = [this.underwater.getBindGroupLayout(1), this.underwater.getBindGroupLayout(2)];
    this.brickmapLayout = this.pipeline.getBindGroupLayout(1);
    this.camera = this.device.createBuffer({
      label: `${this.name}-camera`,
      size: CAMERA_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.materialLayout = this.pipeline.getBindGroupLayout(2);
  }

  resize(width: number, height: number): void {
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion || !g.water) return;
    const tile = config.trace.prepassTile;
    this.coarse?.destroy();
    this.coarse = this.device.createTexture({
      label: `${this.name}-coarse`,
      size: { width: Math.ceil(width / tile), height: Math.ceil(height / tile) },
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.prepassBindGroup = this.device.createBindGroup({
      label: `${this.name}-prepass`,
      layout: this.prepassPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera } },
        { binding: 1, resource: this.coarse.createView() },
      ],
    });
    this.flags?.destroy();
    this.flags = createMicroFlags(this.device, this.name, width, height);
    this.waterSurface?.destroy();
    this.waterSurface = this.device.createTexture({
      label: `${this.name}-water-surface`,
      size: { width, height },
      format: 'rg32uint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const [wx, wy] = this.workgroup;
    this.tileCount = Math.ceil(width / wx) * Math.ceil(height / wy);
    this.tiles?.destroy();
    this.tiles = this.device.createBuffer({
      label: `${this.name}-micro-tiles`,
      size: (TILE_HEADER + 2 * this.tileCount) * 4,
      // COPY_SRC: the "water seen" word is read back.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Indirect args y = z = 1 (x, the listed tile count, is cleared every frame).
    this.device.queue.writeBuffer(this.tiles, 4, new Uint32Array([1, 1]));
    const dummy = microFlagDummy(this.device).createView();
    const entries = (pipeline: GPUComputePipeline, flagOut: GPUTextureView, flagIn: GPUTextureView, tilesRw: GPUBuffer, tilesRead: GPUBuffer) =>
      this.device.createBindGroup({
        label: pipeline.label,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera } },
          { binding: 1, resource: g.gbuffer0!.createView() },
          { binding: 2, resource: g.depth!.createView() },
          { binding: 3, resource: g.motion!.createView() },
          { binding: 4, resource: this.coarse!.createView() },
          { binding: 5, resource: flagOut },
          { binding: 6, resource: flagIn },
          { binding: 7, resource: { buffer: tilesRw } },
          { binding: 8, resource: { buffer: tilesRead } },
          { binding: 9, resource: g.water!.createView() },
          { binding: 10, resource: { buffer: this.water.params } },
          { binding: 12, resource: this.waterSurface!.createView() },
        ],
      });
    this.bindGroup = entries(this.pipeline, this.flags.createView(), dummy, this.tiles, this.dummyTiles);
    this.fixupBindGroup = this.fixup && entries(this.fixup, dummy, this.flags.createView(), this.dummyTiles, this.tiles);
    this.underwaterGroup = entries(this.underwater, dummy, this.flags.createView(), this.dummyTiles, this.tiles);
    this.surfaceGroup = this.device.createBindGroup({
      label: this.surfacePipeline.label,
      layout: this.surfacePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera } },
        { binding: 6, resource: this.flags.createView() },
        { binding: 8, resource: { buffer: this.tiles } },
        { binding: 10, resource: { buffer: this.water.params } },
        { binding: 11, resource: this.waterSurface.createView() },
      ],
    });
  }

  /** Every few frames: copies the "water seen" word (read back in afterSubmit). */
  private readWaterSeen(encoder: GPUCommandEncoder): void {
    if (!this.tiles || this.waterSeenState !== 'idle' || ++this.waterSeenFrame < WATER_SEEN_INTERVAL) return;
    this.waterSeenFrame = 0;
    encoder.copyBufferToBuffer(this.tiles, 12, this.waterSeen, 0, 4);
    this.waterSeenState = 'copied';
  }

  /** After the frame's submit: maps the copied word (no stall; result a few frames late). */
  afterSubmit(): void {
    if (this.waterSeenState !== 'copied') return;
    this.waterSeenState = 'mapping';
    this.waterSeen.mapAsync(GPUMapMode.READ).then(
      () => {
        this.waterVisible = new Uint32Array(this.waterSeen.getMappedRange())[0] !== 0;
        this.waterSeen.unmap();
        this.waterSeenState = 'idle';
      },
      () => {
        this.waterSeenState = 'idle';
      },
    );
  }

  /** Pixels whose ray stopped at water or a shaped block (r32uint, non-zero), this frame. */
  get fixupFlags(): GPUTexture | null {
    return this.flags;
  }

  execute(ctx: FrameContext): void {
    this.encode(ctx, config.trace.prepass);
  }

  /** Records the pass; `prepass` can be forced on/off (the prepass verify tool compares both). */
  encode(ctx: FrameContext, prepass: boolean): void {
    if (!this.bindGroup || !this.prepassBindGroup || !this.coarse) return;
    const cam = ctx.camera;
    const f32 = new Float32Array(this.cameraData);
    const u32 = new Uint32Array(this.cameraData);
    const i32 = new Int32Array(this.cameraData);
    f32.set(cam.invViewProj, 0);
    f32.set(cam.viewProj, 16);
    f32.set(cam.prevViewProj, 32);
    i32.set(cam.cell, 48);
    u32[51] = config.trace.maxSteps;
    f32.set(cam.frac, 52);
    f32[55] = config.camera.far;
    f32.set(cam.prevDelta, 56);
    f32.set(cam.forward, 60);
    u32[64] = this.gbuffer.width;
    u32[65] = this.gbuffer.height;
    const tile = config.trace.prepassTile;
    // Every ray of a tile passes within half a tile diagonal (in pixels) of its centre ray;
    // at distance t that is at most t · (pixels · tan-space size of one pixel).
    const pixelTan = (2 * Math.tan((config.camera.fovY * DEG) / 2)) / this.gbuffer.height;
    // Ray cone for texture filtering: one pixel's footprint grows by pixelTan per block.
    this.materials.update(pixelTan);
    f32[59] = ((tile * Math.SQRT2) / 2) * pixelTan * config.trace.prepassConeMargin;
    f32[63] = config.trace.prepassSafety;
    // Under water the prepass finds nothing to skip (every ray starts inside water).
    const under = this.water.cameraUnder;
    if (under) prepass = false;
    u32[66] = prepass ? tile : 0;
    u32[67] = config.trace.prepassMaxSteps;
    f32.set(cam.prevForward, 68);
    u32[71] = cam.frameIndex;
    this.device.queue.writeBuffer(this.camera, 0, this.cameraData);
    this.water.update(ctx.time);

    // Only pixels that hit a shaped block or water write a flag (and a water sample).
    if (this.fixup && this.flags && this.tiles) {
      clearMicroFlags(ctx.encoder, this.flags);
      clearMicroFlags(ctx.encoder, this.gbuffer.water!);
      ctx.encoder.clearBuffer(this.tiles, 0, 4);
      ctx.encoder.clearBuffer(this.tiles, 12, 4);
      ctx.encoder.clearBuffer(this.tiles, TILE_HEADER * 4, this.tileCount * 4);
    }
    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    if (prepass) {
      pass.setPipeline(this.prepassPipeline);
      pass.setBindGroup(0, this.prepassBindGroup);
      // The prepass reads only pointers (params + grid), never voxels.
      pass.setBindGroup(1, this.brickmap.bindGroup(this.prepassLayout, [0, 1]));
      pass.dispatchWorkgroups(Math.ceil(this.coarse.width / this.workgroup[0]), Math.ceil(this.coarse.height / this.workgroup[1]));
    }
    const groups = [Math.ceil(this.gbuffer.width / this.workgroup[0]), Math.ceil(this.gbuffer.height / this.workgroup[1])] as const;
    if (under && this.underwaterGroup) {
      // Camera under water: every pixel through the water and out of (or mirrored at) the
      // surface, in one pipeline of its own.
      pass.setPipeline(this.underwater);
      pass.setBindGroup(0, this.underwaterGroup);
      pass.setBindGroup(1, this.brickmap.bindGroup(this.underwaterLayouts[0], GpuBrickmap.TRACE_BINDINGS));
      pass.setBindGroup(2, this.materials.bindGroup(this.underwaterLayouts[1]));
      pass.dispatchWorkgroups(...groups);
      pass.end();
      return;
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(1, this.brickmap.bindGroup(this.brickmapLayout, GpuBrickmap.TRACE_BINDINGS));
    pass.setBindGroup(2, this.materials.bindGroup(this.materialLayout));
    pass.dispatchWorkgroups(...groups);
    if (this.fixup && this.fixupBindGroup && this.fixupLayouts && this.tiles && this.surfaceGroup) {
      // Water surface under the waves first (same tiles), then the fix-up.
      pass.setPipeline(this.surfacePipeline);
      pass.setBindGroup(0, this.surfaceGroup);
      // getVoxel: params, grid, pool. Ripples: material params, sampler, noise volume.
      pass.setBindGroup(1, this.brickmap.bindGroup(this.surfaceLayouts[0], [0, 1, 2]));
      pass.setBindGroup(2, this.materials.bindGroup(this.surfaceLayouts[1], [0, 6, 10]));
      pass.dispatchWorkgroupsIndirect(this.tiles, 0);
      // One workgroup per listed tile; unflagged pixels in it return after one texel load.
      pass.setPipeline(this.fixup);
      pass.setBindGroup(0, this.fixupBindGroup);
      pass.setBindGroup(1, this.brickmap.bindGroup(this.fixupLayouts[0], GpuBrickmap.TRACE_BINDINGS));
      pass.setBindGroup(2, this.materials.bindGroup(this.fixupLayouts[1]));
      pass.dispatchWorkgroupsIndirect(this.tiles, 0);
    }
    pass.end();
    this.readWaterSeen(ctx.encoder);
  }
}
