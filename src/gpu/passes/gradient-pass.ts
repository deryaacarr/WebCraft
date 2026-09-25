import { config } from '../../config';
import { createShaderModule } from '../shader';
import type { FrameContext, RenderPass } from './pass';

/** HDR target written by compute passes and consumed by the blit pass. */
export const SCENE_FORMAT: GPUTextureFormat = 'rgba16float';

// Params: size (vec2u), time, speed, wave_strength → 20 bytes, padded to 32.
const PARAMS_SIZE = 32;

/** Test pass: fills the scene texture with an animated gradient from a compute shader. */
export class GradientPass implements RenderPass {
  readonly name = 'gradient';
  output: GPUTexture | null = null;

  private pipeline!: GPUComputePipeline;
  private params!: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private readonly paramData = new ArrayBuffer(PARAMS_SIZE);
  private readonly paramU32 = new Uint32Array(this.paramData);
  private readonly paramF32 = new Float32Array(this.paramData);
  private readonly workgroupSize = config.render.workgroupSize;

  constructor(private readonly device: GPUDevice) {}

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'gradient.wgsl');
    this.pipeline = await this.device.createComputePipelineAsync({
      label: this.name,
      layout: 'auto',
      compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: this.workgroupSize } },
    });
    this.params = this.device.createBuffer({
      label: `${this.name}-params`,
      size: PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  resize(width: number, height: number): void {
    this.output?.destroy();
    this.output = this.device.createTexture({
      label: `${this.name}-output`,
      size: { width, height },
      format: SCENE_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bindGroup = this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: this.output.createView() },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this.output || !this.bindGroup) return;
    const { width, height } = this.output;

    this.paramU32[0] = width;
    this.paramU32[1] = height;
    this.paramF32[2] = ctx.time;
    this.paramF32[3] = config.gradient.speed;
    this.paramF32[4] = config.gradient.waveStrength;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({
      label: this.name,
      ...(timestampWrites && { timestampWrites }),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width / this.workgroupSize),
      Math.ceil(height / this.workgroupSize),
    );
    pass.end();
  }
}
