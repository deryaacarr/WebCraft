import { config } from '../../config';
import { createShaderModule } from '../shader';
import type { FrameContext, RenderPass } from './pass';

/** Upscales the internal-resolution scene texture onto the swap chain. */
export class BlitPass implements RenderPass {
  readonly name = 'blit';

  private pipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private bindGroup: GPUBindGroup | null = null;
  private params!: GPUBuffer;
  private tonemapKey = '';

  constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: GPUTextureFormat,
    /** Exposure state buffer (white-balance matrix). */
    private readonly exposureState: GPUBuffer,
  ) {}

  /** 'none' for debug views; AgX / ACES (with white balance) for pre-exposed HDR. */
  setTonemap(mode: 'none' | 'agx' | 'agx-punchy' | 'aces'): void {
    const value = { none: 0, agx: 1, aces: 2, 'agx-punchy': 3 }[mode];
    const r = config.render;
    const key = `${value},${r.agxPunchyContrast},${r.agxPunchySaturation}`;
    if (key === this.tonemapKey) return;
    this.tonemapKey = key;
    const data = new ArrayBuffer(16);
    new Uint32Array(data)[0] = value;
    new Float32Array(data).set([r.agxPunchyContrast, r.agxPunchySaturation], 1);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  async init(): Promise<void> {
    this.params = this.device.createBuffer({ label: `${this.name}-params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = await createShaderModule(this.device, 'blit.wgsl');
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: this.name,
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = this.device.createSampler({
      label: this.name,
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  /** The blit output always matches the canvas, so only the source binding matters. */
  resize(): void {}

  /** Rebinds the texture to upscale; call whenever the source is recreated. */
  setSource(source: GPUTexture): void {
    this.bindGroup = this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: source.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.params } },
        { binding: 3, resource: { buffer: this.exposureState } },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this.bindGroup) return;
    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginRenderPass({
      label: this.name,
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
      ...(timestampWrites && { timestampWrites }),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }
}
