import { config } from '../../config';
import { kelvinToLinearRgb } from '../../core/color-temperature';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import type { FrameContext, RenderPass } from './pass';

// ExposureParams in exposure.wgsl.
const PARAMS_SIZE = 80;
const WB_MODES = { off: 0, auto: 1, manual: 2 } as const;

/** Eye adaptation and auto white balance (exposure.wgsl) from the lit HDR frame. */
export class ExposurePass implements RenderPass {
  readonly name = 'exposure';

  private pipeline!: GPUComputePipeline;
  private skyLayout!: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private group: GPUBindGroup | null = null;

  constructor(
    private readonly device: GPUDevice,
    private readonly hdr: () => GPUTexture | null,
    private readonly gbuffer: GBuffer,
    private readonly sky: SkySystem,
  ) {
    this.params = device.createBuffer({ label: 'exposure-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'exposure.wgsl');
    this.pipeline = await this.device.createComputePipelineAsync({ label: this.name, layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.skyLayout = this.pipeline.getBindGroupLayout(3);
  }

  resize(): void {
    this.group = null;
  }

  execute(ctx: FrameContext): void {
    const hdr = this.hdr();
    const g = this.gbuffer.gbuffer0;
    if (!hdr || !g) return;
    this.group ??= this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: hdr.createView() },
        { binding: 2, resource: { buffer: this.sky.exposure } },
        { binding: 3, resource: g.createView() },
      ],
    });
    const e = config.exposure;
    const wb = config.whiteBalance;
    const p = new ArrayBuffer(PARAMS_SIZE);
    const f32 = new Float32Array(p);
    f32.set([
      Math.min(ctx.dt, 0.25),
      e.key,
      e.compensation,
      e.trim,
      2 ** e.minEv,
      2 ** e.maxEv,
      e.adaptDarkerSeconds,
      e.adaptBrighterSeconds,
      e.centerSigma,
      e.skyWeight,
    ]);
    new Uint32Array(p)[10] = WB_MODES[wb.mode];
    f32[11] = wb.strength;
    f32.set(kelvinToLinearRgb(wb.temperature), 12);
    f32[15] = wb.adaptSeconds;
    f32[16] = e.maxDarkAdaptationEv;
    this.device.queue.writeBuffer(this.params, 0, p);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    // Sky params (light direction) and the lighting summary: the scene illuminant.
    pass.setBindGroup(3, this.sky.bindGroup(this.skyLayout, [0, 5]));
    pass.dispatchWorkgroups(1);
    pass.end();
  }
}
