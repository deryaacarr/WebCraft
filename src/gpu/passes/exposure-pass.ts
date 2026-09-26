import { config } from '../../config';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import type { FrameContext, RenderPass } from './pass';

/** Eye adaptation (exposure.wgsl): updates the exposure buffer from the lit HDR frame. */
export class ExposurePass implements RenderPass {
  readonly name = 'exposure';

  private pipeline!: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private group: GPUBindGroup | null = null;

  constructor(
    private readonly device: GPUDevice,
    private readonly hdr: () => GPUTexture | null,
    private readonly sky: SkySystem,
  ) {
    this.params = device.createBuffer({ label: 'exposure-params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'exposure.wgsl');
    this.pipeline = await this.device.createComputePipelineAsync({ label: this.name, layout: 'auto', compute: { module, entryPoint: 'main' } });
  }

  resize(): void {
    this.group = null;
  }

  execute(ctx: FrameContext): void {
    const hdr = this.hdr();
    if (!hdr) return;
    this.group ??= this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: hdr.createView() },
        { binding: 2, resource: { buffer: this.sky.exposure } },
      ],
    });
    const e = config.exposure;
    this.device.queue.writeBuffer(
      this.params,
      0,
      new Float32Array([Math.min(ctx.dt, 0.25), e.key, e.compensation, e.adaptationSpeed, 2 ** e.minEv, 2 ** e.maxEv, 0, 0]),
    );
    const pass = ctx.encoder.beginComputePass({ label: this.name });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(1);
    pass.end();
  }
}
