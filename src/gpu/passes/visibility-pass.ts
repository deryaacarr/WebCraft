import { config } from '../../config';
import type { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import type { MaterialSystem } from '../materials';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import { traceConstants } from '../trace-constants';
import type { FrameContext, RenderPass } from './pass';

/**
 * Sun/moon shadow and sky visibility: one jittered ray of each per pixel and frame
 * (visibility.wgsl), accumulated over time with reprojection (temporal.wgsl).
 * `output` holds the accumulated result (r = light, g = sky).
 */
export class VisibilityPass implements RenderPass {
  readonly name = 'visibility';

  private tracePipeline!: GPUComputePipeline;
  private temporalPipeline!: GPUComputePipeline;
  private spatialPipeline!: GPUComputePipeline;
  /** Spatially filtered visibility (what lighting reads). */
  private filtered: GPUTexture | null = null;
  private spatialGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private layouts!: { brickmap: GPUBindGroupLayout; materials: GPUBindGroupLayout; sky: GPUBindGroupLayout };
  private readonly params: GPUBuffer;
  private readonly temporalParams: GPUBuffer;
  private raw: GPUTexture | null = null;
  /** Previous frame's G-buffer word and depth, for history validation. */
  private prevGbuffer: GPUTexture | null = null;
  private prevDepth: GPUTexture | null = null;
  private history: [GPUTexture, GPUTexture] | null = null;
  private traceGroup: GPUBindGroup | null = null;
  private temporalGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  /** Which history texture receives this frame's result. */
  private current = 0;
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    private readonly brickmap: GpuBrickmap,
    private readonly materials: MaterialSystem,
    private readonly sky: SkySystem,
  ) {
    const uniform = (label: string, size = 16) => device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.params = uniform('visibility-params', 32);
    this.temporalParams = uniform('temporal-params', 32);
  }

  /** Filtered visibility of the last executed frame (r light, g sky, a history length). */
  get output(): GPUTexture | null {
    return this.filtered;
  }

  async init(): Promise<void> {
    const [trace, temporal] = await Promise.all([
      createShaderModule(this.device, 'visibility.wgsl'),
      createShaderModule(this.device, 'temporal.wgsl'),
    ]);
    const [wx, wy] = this.workgroup;
    [this.tracePipeline, this.temporalPipeline, this.spatialPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: 'visibility',
        layout: 'auto',
        compute: { module: trace, entryPoint: 'main', constants: { WORKGROUP_X: wx, WORKGROUP_Y: wy, ...traceConstants() } },
      }),
      this.device.createComputePipelineAsync({
        label: 'temporal',
        layout: 'auto',
        compute: { module: temporal, entryPoint: 'main', constants: { WORKGROUP_X: wx, WORKGROUP_Y: wy } },
      }),
      this.device.createComputePipelineAsync({
        label: 'visibility-spatial',
        layout: 'auto',
        compute: { module: temporal, entryPoint: 'spatial', constants: { WORKGROUP_X: wx, WORKGROUP_Y: wy } },
      }),
    ]);
    this.layouts = {
      brickmap: this.tracePipeline.getBindGroupLayout(1),
      materials: this.tracePipeline.getBindGroupLayout(2),
      sky: this.tracePipeline.getBindGroupLayout(3),
    };
  }

  resize(width: number, height: number): void {
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion) return;
    for (const t of [this.raw, this.prevGbuffer, this.prevDepth, this.filtered, ...(this.history ?? [])]) t?.destroy();
    const copyTarget = (label: string, format: GPUTextureFormat) =>
      this.device.createTexture({ label, size: { width, height }, format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.prevGbuffer = copyTarget('prev-gbuffer0', 'rgba32uint');
    this.prevDepth = copyTarget('prev-depth', 'r32float');
    const make = (label: string) =>
      this.device.createTexture({
        label,
        size: { width, height },
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.raw = make('visibility-raw');
    this.filtered = make('visibility-filtered');
    this.history = [make('visibility-history-a'), make('visibility-history-b')];
    this.traceGroup = this.device.createBindGroup({
      label: 'visibility',
      layout: this.tracePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera() } },
        { binding: 1, resource: g.gbuffer0.createView() },
        { binding: 2, resource: g.depth.createView() },
        { binding: 3, resource: this.raw.createView() },
        { binding: 4, resource: { buffer: this.params } },
      ],
    });
    const temporal = (write: GPUTexture, read: GPUTexture) =>
      this.device.createBindGroup({
        label: 'temporal',
        layout: this.temporalPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.temporalParams } },
          { binding: 2, resource: this.raw!.createView() },
          { binding: 3, resource: read.createView() },
          { binding: 4, resource: g.depth!.createView() },
          { binding: 5, resource: g.motion!.createView() },
          { binding: 6, resource: write.createView() },
          { binding: 7, resource: g.gbuffer0!.createView() },
          { binding: 8, resource: this.prevGbuffer!.createView() },
          { binding: 9, resource: this.prevDepth!.createView() },
        ],
      });
    const [a, b] = this.history;
    this.temporalGroups = [temporal(a, b), temporal(b, a)];
    const spatial = (source: GPUTexture) =>
      this.device.createBindGroup({
        label: 'visibility-spatial',
        layout: this.spatialPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.temporalParams } },
          { binding: 4, resource: g.depth!.createView() },
          { binding: 7, resource: g.gbuffer0!.createView() },
          { binding: 10, resource: source.createView() },
          { binding: 11, resource: this.filtered!.createView() },
        ],
      });
    this.spatialGroups = [spatial(a), spatial(b)];
  }

  execute(ctx: FrameContext): void {
    if (!this.traceGroup || !this.temporalGroups) return;
    const l = config.lighting;
    const block = l.visibilityCheckerboard ? 2 : 1;
    const p = new ArrayBuffer(32);
    new Float32Array(p).set([l.shadowDistance, l.skyVisibilityDistance]);
    new Uint32Array(p).set([config.trace.maxSteps, block, l.skyVisibilitySteps], 2);
    new Float32Array(p)[5] = l.leafTransmission;
    new Uint32Array(p)[6] = config.gi.enabled ? 0 : 1;
    this.device.queue.writeBuffer(this.params, 0, p);
    // How fast the camera moves this frame (0 still … 1 fast): translation plus rotation.
    const c = ctx.camera;
    const turn = Math.acos(Math.min(1, c.forward[0] * c.prevForward[0] + c.forward[1] * c.prevForward[1] + c.forward[2] * c.prevForward[2]));
    const cameraMotion = Math.min(1, Math.hypot(...c.prevDelta) / l.historyCameraSpeed + (turn * 180) / Math.PI / l.historyCameraTurn);
    const t = new ArrayBuffer(32);
    new Float32Array(t).set([l.historyStill, l.historyMoving, cameraMotion, l.historyMotionPixels, l.temporalDepthTolerance, l.historyClipK]);
    new Uint32Array(t)[6] = block;
    this.device.queue.writeBuffer(this.temporalParams, 0, t);

    this.current = 1 - this.current;
    const w = Math.ceil(this.gbuffer.width / this.workgroup[0]);
    const h = Math.ceil(this.gbuffer.height / this.workgroup[1]);
    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.tracePipeline);
    pass.setBindGroup(0, this.traceGroup);
    pass.setBindGroup(1, this.brickmap.bindGroup(this.layouts.brickmap));
    // Alpha testing only reads albedo: params, tables, albedo array, sampler.
    pass.setBindGroup(2, this.materials.bindGroup(this.layouts.materials, [0, 1, 2, 3, 6]));
    pass.setBindGroup(3, this.sky.bindGroup(this.layouts.sky, [0]));
    pass.dispatchWorkgroups(Math.ceil(w / block), Math.ceil(h / block));
    pass.setPipeline(this.temporalPipeline);
    pass.setBindGroup(0, this.temporalGroups[this.current]!);
    pass.dispatchWorkgroups(w, h);
    pass.setPipeline(this.spatialPipeline);
    pass.setBindGroup(0, this.spatialGroups![this.current]!);
    pass.dispatchWorkgroups(w, h);
    pass.end();
    // Keep this frame's surface identity and depth for the next frame's validation.
    const g = this.gbuffer;
    const size = { width: g.width, height: g.height };
    ctx.encoder.copyTextureToTexture({ texture: g.gbuffer0! }, { texture: this.prevGbuffer! }, size);
    ctx.encoder.copyTextureToTexture({ texture: g.depth! }, { texture: this.prevDepth! }, size);
  }
}
