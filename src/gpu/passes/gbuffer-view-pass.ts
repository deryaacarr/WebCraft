import { config, DEBUG_VIEWS } from '../../config';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import { SCENE_FORMAT } from './gradient-pass';
import type { FrameContext, RenderPass } from './pass';

// ViewParams in gbuffer-view.wgsl: sun_dir vec3f, mode u32, size vec2u, far f32,
// max_steps u32, motion_scale f32, depth_half f32, pad → 48 bytes.
const PARAMS_SIZE = 48;

/** Resolves the G-buffer into the scene texture: lit preview or one debug channel. */
export class GBufferViewPass implements RenderPass {
  readonly name = 'gbuffer-view';
  output: GPUTexture | null = null;

  private pipeline!: GPUComputePipeline;
  private params!: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private readonly paramData = new ArrayBuffer(PARAMS_SIZE);
  private readonly workgroupSize = config.render.workgroupSize;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
  ) {}

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'gbuffer-view.wgsl');
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
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion) return;
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
        { binding: 1, resource: g.gbuffer0.createView() },
        { binding: 2, resource: g.depth.createView() },
        { binding: 3, resource: g.motion.createView() },
        { binding: 4, resource: this.output.createView() },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this.output || !this.bindGroup) return;
    const { width, height } = this.output;
    const f32 = new Float32Array(this.paramData);
    const u32 = new Uint32Array(this.paramData);
    const [sx, sy, sz] = config.debug.sunDirection;
    const len = Math.hypot(sx, sy, sz) || 1;
    f32.set([sx / len, sy / len, sz / len], 0);
    u32[3] = DEBUG_VIEWS.indexOf(config.debug.view);
    u32[4] = width;
    u32[5] = height;
    f32[6] = config.camera.far;
    u32[7] = config.trace.maxSteps;
    f32[8] = config.debug.motionScale;
    f32[9] = config.debug.depthHalf;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / this.workgroupSize), Math.ceil(height / this.workgroupSize));
    pass.end();
  }
}
