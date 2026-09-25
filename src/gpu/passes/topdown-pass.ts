import { config } from '../../config';
import { BLOCKS, BlockId } from '../../world/blocks';
import type { GpuBrickmap } from '../brickmap';
import { createShaderModule } from '../shader';
import { SCENE_FORMAT } from './gradient-pass';
import type { FrameContext, RenderPass } from './pass';

// Params: size vec2u, center vec2f, blocks_per_pixel f32, min_y i32, max_y i32,
// block_count u32, water_id u32 → 36 bytes, padded to 48.
const PARAMS_SIZE = 48;

/** Debug view: straight-down look at the GPU brickmap around the player. */
export class TopdownPass implements RenderPass {
  readonly name = 'topdown';
  output: GPUTexture | null = null;

  private pipeline!: GPUComputePipeline;
  /** Fetched once: getBindGroupLayout() returns a new object per call, defeating bind group caches. */
  private brickmapLayout!: GPUBindGroupLayout;
  private params!: GPUBuffer;
  private colors!: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private readonly paramData = new ArrayBuffer(PARAMS_SIZE);
  private readonly workgroupSize = config.render.workgroupSize;

  constructor(
    private readonly device: GPUDevice,
    private readonly brickmap: GpuBrickmap,
  ) {}

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'topdown.wgsl');
    this.pipeline = await this.device.createComputePipelineAsync({
      label: this.name,
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main',
        constants: { WORKGROUP_SIZE: this.workgroupSize, BRICK_BITS: config.world.brickBits },
      },
    });
    this.brickmapLayout = this.pipeline.getBindGroupLayout(1);
    this.params = this.device.createBuffer({
      label: `${this.name}-params`,
      size: PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const colors = new Float32Array(BLOCKS.length * 4);
    BLOCKS.forEach((b, i) => {
      colors.set([((b.color >> 16) & 0xff) / 255, ((b.color >> 8) & 0xff) / 255, (b.color & 0xff) / 255, 1], i * 4);
    });
    this.colors = this.device.createBuffer({
      label: `${this.name}-colors`,
      size: colors.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.colors, 0, colors);
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
        { binding: 2, resource: { buffer: this.colors } },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this.output || !this.bindGroup) return;
    const { width, height } = this.output;
    const u32 = new Uint32Array(this.paramData);
    const f32 = new Float32Array(this.paramData);
    const i32 = new Int32Array(this.paramData);
    u32[0] = width;
    u32[1] = height;
    f32[2] = ctx.camera.position[0];
    f32[3] = ctx.camera.position[2];
    f32[4] = config.debug.topdownBlocksPerPixel;
    i32[5] = config.world.minY;
    i32[6] = config.world.maxY;
    u32[7] = BLOCKS.length;
    u32[8] = BlockId.water;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(1, this.brickmap.bindGroup(this.brickmapLayout));
    pass.dispatchWorkgroups(Math.ceil(width / this.workgroupSize), Math.ceil(height / this.workgroupSize));
    pass.end();
  }
}
