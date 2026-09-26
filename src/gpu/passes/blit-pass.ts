import { createShaderModule } from '../shader';
import type { FrameContext, RenderPass } from './pass';

/** Upscales the internal-resolution scene texture onto the swap chain. */
export class BlitPass implements RenderPass {
  readonly name = 'blit';

  private pipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private bindGroup: GPUBindGroup | null = null;
  private params!: GPUBuffer;
  private tonemap = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: GPUTextureFormat,
  ) {}

  /** ACES tone mapping for pre-exposed HDR input (lit view) vs. plain display (debug views). */
  setTonemap(enabled: boolean): void {
    if (enabled === this.tonemap) return;
    this.tonemap = enabled;
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([enabled ? 1 : 0, 0, 0, 0]));
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
