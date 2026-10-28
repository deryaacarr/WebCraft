import { config } from '../../config';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import type { FrameContext, RenderPass } from './pass';

// TaauParams in taau.wgsl.
const PARAMS_SIZE = 48;

/**
 * Temporal upsampling (taau.wgsl): accumulates the jittered internal-resolution frames
 * into a history at the output resolution (`config.taau.outputScale` × canvas). Runs after
 * lighting and before the exposure pass (the lit image is pre-exposed with the exposure
 * the exposure pass is about to update).
 */
export class TaauPass implements RenderPass {
  readonly name = 'taau';

  private pipeline!: GPUComputePipeline;
  private exposurePipeline!: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private readonly historyExposure: GPUBuffer;
  private readonly sampler: GPUSampler;
  private history: [GPUTexture, GPUTexture] | null = null;
  private prevDepth: GPUTexture | null = null;
  private cur = 0;
  /** The history is invalid until the next frame wrote it (resize, re-enabled). */
  private reset = true;
  private groups = new Map<string, GPUBindGroup>();
  private exposureGroup: GPUBindGroup | null = null;
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    /** This frame's lit image (internal resolution, pre-exposed). */
    private readonly color: () => GPUTexture | null,
    /** Exposure state buffer (exposure.wgsl). */
    private readonly exposure: GPUBuffer,
    /** Canvas backing size (the output follows it). */
    private readonly canvasSize: () => { width: number; height: number },
  ) {
    this.params = device.createBuffer({ label: `${this.name}-params`, size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.historyExposure = device.createBuffer({ label: `${this.name}-history-exposure`, size: 16, usage: GPUBufferUsage.STORAGE });
    this.sampler = device.createSampler({ label: this.name, magFilter: 'linear', minFilter: 'linear' });
  }

  /** Upsampled image of the last executed frame (pre-exposed HDR, output resolution). */
  get output(): GPUTexture | null {
    return this.history?.[this.cur] ?? null;
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'taau.wgsl');
    const [wx, wy] = this.workgroup;
    [this.pipeline, this.exposurePipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ label: this.name, layout: 'auto', compute: { module, entryPoint: 'main', constants: { WORKGROUP_X: wx, WORKGROUP_Y: wy } } }),
      this.device.createComputePipelineAsync({ label: `${this.name}-exposure`, layout: 'auto', compute: { module, entryPoint: 'storeExposure' } }),
    ]);
    this.exposureGroup = this.device.createBindGroup({
      label: `${this.name}-exposure`,
      layout: this.exposurePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 10, resource: { buffer: this.exposure } },
        { binding: 11, resource: { buffer: this.historyExposure } },
      ],
    });
  }

  resize(width: number, height: number): void {
    for (const t of [this.prevDepth, ...(this.history ?? [])]) t?.destroy();
    this.prevDepth = this.device.createTexture({
      label: `${this.name}-prev-depth`,
      size: { width, height },
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const canvas = this.canvasSize();
    const scale = config.taau.outputScale;
    const size = { width: Math.max(1, Math.round(canvas.width * scale)), height: Math.max(1, Math.round(canvas.height * scale)) };
    const make = (label: string) =>
      this.device.createTexture({ label: `${this.name}-${label}`, size, format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.history = [make('history-a'), make('history-b')];
    this.groups.clear();
    this.reset = true;
  }

  /** Forget the history (e.g. TAAU was off for a while). */
  invalidate(): void {
    this.reset = true;
  }

  execute(ctx: FrameContext): void {
    const g = this.gbuffer;
    const color = this.color();
    if (!this.history || !this.prevDepth || !color || !g.depth || !g.motion || !g.gbuffer0 || !this.exposureGroup) return;
    const read = this.history[this.cur]!;
    this.cur = 1 - this.cur;
    const write = this.history[this.cur]!;
    const pipeline = this.pipeline;
    const key = `${this.cur}|${color.label}|${color.width}|${pipeline.label}`;
    let group = this.groups.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        label: this.name,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 2, resource: color.createView() },
          { binding: 3, resource: g.depth.createView() },
          { binding: 4, resource: g.motion.createView() },
          { binding: 5, resource: g.gbuffer0.createView() },
          { binding: 6, resource: this.prevDepth.createView() },
          { binding: 7, resource: read.createView() },
          { binding: 8, resource: write.createView() },
          { binding: 9, resource: this.sampler },
          { binding: 10, resource: { buffer: this.exposure } },
          { binding: 11, resource: { buffer: this.historyExposure } },
        ],
      });
      this.groups.set(key, group);
    }
    const t = config.taau;
    const c = ctx.camera;
    const l = config.lighting;
    // How fast the camera moves this frame (0 still … 1 fast), as for the visibility history.
    const turn = Math.acos(Math.min(1, c.forward[0] * c.prevForward[0] + c.forward[1] * c.prevForward[1] + c.forward[2] * c.prevForward[2]));
    const cameraMotion = Math.min(1, Math.hypot(...c.prevDelta) / l.historyCameraSpeed + (turn * 180) / Math.PI / l.historyCameraTurn);
    const p = new ArrayBuffer(PARAMS_SIZE);
    const f = new Float32Array(p);
    f.set([c.jitter[0], c.jitter[1], t.framesStill, t.framesMoving, t.gamma, t.gammaWater, t.framesWater, t.depthTolerance, cameraMotion]);
    new Uint32Array(p)[9] = this.reset ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, p);
    this.reset = false;

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(write.width / this.workgroup[0]), Math.ceil(write.height / this.workgroup[1]));
    pass.setPipeline(this.exposurePipeline);
    pass.setBindGroup(0, this.exposureGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    ctx.encoder.copyTextureToTexture({ texture: g.depth }, { texture: this.prevDepth }, { width: g.width, height: g.height });
  }
}
