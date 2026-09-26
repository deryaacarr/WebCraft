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
  private layouts!: { brickmap: GPUBindGroupLayout; materials: GPUBindGroupLayout; sky: GPUBindGroupLayout };
  private readonly params: GPUBuffer;
  private readonly temporalParams: GPUBuffer;
  private raw: GPUTexture | null = null;
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
    this.temporalParams = uniform('temporal-params');
  }

  /** Index (0/1) of the history texture holding the last executed frame's result. */
  get outputIndex(): number {
    return this.current;
  }

  /** History texture 0 or 1 (consumers pre-build bind groups for both). */
  historyTexture(index: number): GPUTexture {
    const t = this.history?.[index];
    if (!t) throw new Error('visibility pass not sized');
    return t;
  }

  async init(): Promise<void> {
    const [trace, temporal] = await Promise.all([
      createShaderModule(this.device, 'visibility.wgsl'),
      createShaderModule(this.device, 'temporal.wgsl'),
    ]);
    const [wx, wy] = this.workgroup;
    [this.tracePipeline, this.temporalPipeline] = await Promise.all([
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
    for (const t of [this.raw, ...(this.history ?? [])]) t?.destroy();
    const make = (label: string) =>
      this.device.createTexture({
        label,
        size: { width, height },
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.raw = make('visibility-raw');
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
        ],
      });
    const [a, b] = this.history;
    this.temporalGroups = [temporal(a, b), temporal(b, a)];
  }

  execute(ctx: FrameContext): void {
    if (!this.traceGroup || !this.temporalGroups) return;
    const l = config.lighting;
    const block = l.visibilityCheckerboard ? 2 : 1;
    const p = new ArrayBuffer(32);
    new Float32Array(p).set([l.shadowDistance, l.skyVisibilityDistance]);
    new Uint32Array(p).set([config.trace.maxSteps, block, l.skyVisibilitySteps], 2);
    this.device.queue.writeBuffer(this.params, 0, p);
    const t = new ArrayBuffer(16);
    new Float32Array(t).set([l.temporalFrames, l.temporalDepthTolerance]);
    new Uint32Array(t)[2] = block;
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
    pass.end();
  }
}
