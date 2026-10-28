import { config } from '../../config';
import { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import type { MaterialSystem } from '../materials';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import { traceConstants } from '../trace-constants';
import type { WaterSystem } from '../water';
import type { FrameContext, RenderPass } from './pass';

// ReflectionParams in water-reflection.wgsl.
const PARAMS_SIZE = 32;

/**
 * Ray-traced reflections on water at 1/`config.water.reflectionScale` resolution
 * (water-reflection.wgsl), accumulated over frames; the lighting pass upsamples them and
 * adds the sun's highlight.
 */
export class WaterReflectionPass implements RenderPass {
  readonly name = 'water-reflection';

  private pipeline!: GPUComputePipeline;
  private temporalPipeline!: GPUComputePipeline;
  /** This frame's samples, and the accumulated result (ping-pong, `cur` = this frame's). */
  private raw: GPUTexture | null = null;
  private history: [GPUTexture, GPUTexture] | null = null;
  private cur = 0;
  private temporalGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private layouts!: { brickmap: GPUBindGroupLayout; materials: GPUBindGroupLayout; sky: GPUBindGroupLayout };
  private readonly params: GPUBuffer;
  /** Bind groups per last-frame lit image (the lighting pass re-creates it on resize). */
  private group: GPUBindGroup | null = null;
  private groupLit: GPUTexture | null = null;
  private groupFlags: GPUTexture | null = null;
  /** Bound until the lighting pass has an output. */
  private readonly fallbackLit: GPUTexture;
  private scale = 0;
  /** Called after the output was re-created outside a renderer resize (scale change). */
  onResize: (() => void) | null = null;
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    private readonly brickmap: GpuBrickmap,
    private readonly materials: MaterialSystem,
    private readonly sky: SkySystem,
    private readonly water: WaterSystem,
    /** The primary pass's fix-up flags (water / shaped pixels). */
    private readonly primaryFlags: () => GPUTexture | null,
    private readonly blockRadiance: () => GPUBuffer,
    /** Last frame's lit image (screen-space reuse of hits). */
    private readonly prevLit: () => GPUTexture | null,
  ) {
    this.params = device.createBuffer({ label: `${this.name}-params`, size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fallbackLit = device.createTexture({ label: `${this.name}-no-lit`, size: { width: 1, height: 1 }, format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING });
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'water-reflection.wgsl');
    const [wx, wy] = this.workgroup;
    const constants = { WORKGROUP_X: wx, WORKGROUP_Y: wy, ...traceConstants() };
    [this.pipeline, this.temporalPipeline] = await Promise.all(
      ['main', 'temporal'].map((entryPoint) =>
        this.device.createComputePipelineAsync({ label: `${this.name}-${entryPoint}`, layout: 'auto', compute: { module, entryPoint, constants } }),
      ),
    ) as [GPUComputePipeline, GPUComputePipeline];
    const p = this.pipeline;
    this.layouts = { brickmap: p.getBindGroupLayout(1), materials: p.getBindGroupLayout(2), sky: p.getBindGroupLayout(3) };
  }

  /** Accumulated reflections of the last executed frame (see water-reflection.wgsl). */
  get output(): GPUTexture | null {
    return this.history?.[this.cur] ?? null;
  }

  resize(width: number, height: number): void {
    this.scale = Math.max(1, Math.round(config.water.reflectionScale));
    for (const t of [this.raw, ...(this.history ?? [])]) t?.destroy();
    const make = (label: string) =>
      this.device.createTexture({
        label: `${this.name}-${label}`,
        size: { width: Math.max(1, Math.ceil(width / this.scale)), height: Math.max(1, Math.ceil(height / this.scale)) },
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.raw = make('raw');
    this.history = [make('history-a'), make('history-b')];
    const g = this.gbuffer;
    const temporal = (write: GPUTexture, read: GPUTexture) =>
      this.device.createBindGroup({
        label: `${this.name}-temporal`,
        layout: this.temporalPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 5, resource: g.motion!.createView() },
          { binding: 12, resource: this.raw!.createView() },
          { binding: 13, resource: read.createView() },
          { binding: 14, resource: write.createView() },
        ],
      });
    const [a, b] = this.history;
    this.temporalGroups = g.motion ? [temporal(a, b), temporal(b, a)] : null;
    this.group = null;
  }

  execute(ctx: FrameContext): void {
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion || !g.water) return;
    if (Math.max(1, Math.round(config.water.reflectionScale)) !== this.scale) {
      this.resize(g.width, g.height);
      this.onResize?.();
    }
    if (!this.raw || !this.temporalGroups) return;
    const lit = this.prevLit() ?? this.fallbackLit;
    const flags = this.primaryFlags();
    if (!flags) return;
    if (!this.group || lit !== this.groupLit || flags !== this.groupFlags) {
      this.groupLit = lit;
      this.groupFlags = flags;
      this.group = this.device.createBindGroup({
        label: this.name,
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 2, resource: g.gbuffer0.createView() },
          { binding: 3, resource: g.water.createView() },
          { binding: 4, resource: g.depth.createView() },
          { binding: 5, resource: g.motion.createView() },
          { binding: 6, resource: lit.createView() },
          { binding: 7, resource: this.raw.createView() },
          { binding: 9, resource: { buffer: this.blockRadiance() } },
          { binding: 10, resource: { buffer: this.water.params } },
          { binding: 11, resource: flags.createView() },
        ],
      });
    }
    const w = config.water;
    const p = new ArrayBuffer(PARAMS_SIZE);
    new Float32Array(p).set([w.reflectionDistance]);
    new Uint32Array(p)[1] = w.reflectionMaxSteps;
    new Float32Array(p)[2] = config.lighting.shadowDistance;
    new Uint32Array(p)[3] = config.trace.maxSteps;
    new Float32Array(p)[4] = Math.max(0, Math.log2(Math.max(this.materials.stats.resolution, 1) / config.gi.hitTexels));
    new Uint32Array(p).set([w.reflectionScreenReuse ? 1 : 0, this.scale], 5);
    new Float32Array(p)[7] = 1 / Math.max(1, w.reflectionHistory);
    this.device.queue.writeBuffer(this.params, 0, p);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.setBindGroup(1, this.brickmap.bindGroup(this.layouts.brickmap, GpuBrickmap.TRACE_BINDINGS));
    // Alpha testing and hit albedo: params, tables, albedo array, sampler, noise.
    pass.setBindGroup(2, this.materials.bindGroup(this.layouts.materials, [0, 1, 2, 3, 6, 10]));
    pass.setBindGroup(3, this.sky.bindGroup(this.layouts.sky));
    const groups = [Math.ceil(this.raw.width / this.workgroup[0]), Math.ceil(this.raw.height / this.workgroup[1])] as const;
    pass.dispatchWorkgroups(...groups);
    this.cur = 1 - this.cur;
    pass.setPipeline(this.temporalPipeline);
    pass.setBindGroup(0, this.temporalGroups[this.cur]);
    pass.dispatchWorkgroups(...groups);
    pass.end();
  }
}
