import { config } from '../../config';
import { BLOCKS, MATERIAL_COLORS, MATERIAL_NAMES } from '../../world/blocks';
import type { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import type { FrameContext, RenderPass } from './pass';

// Camera struct in primary.wgsl: 3 × mat4x4f (192 B) + 5 × 16 B.
const CAMERA_SIZE = 272;

/** sRGB 0xRRGGBB → linear [r, g, b]. */
export function srgbHexToLinear(hex: number): [number, number, number] {
  const c = (v: number) => {
    const s = v / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return [c((hex >> 16) & 0xff), c((hex >> 8) & 0xff), c(hex & 0xff)];
}

/** Traces one ray per pixel through the brickmap and fills the G-buffer. */
export class PrimaryPass implements RenderPass {
  readonly name = 'primary';

  private pipeline!: GPUComputePipeline;
  private brickmapLayout!: GPUBindGroupLayout;
  private camera!: GPUBuffer;
  private faces!: GPUBuffer;
  private albedo!: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private readonly cameraData = new ArrayBuffer(CAMERA_SIZE);
  private readonly workgroupSize = config.render.workgroupSize;

  constructor(
    private readonly device: GPUDevice,
    private readonly brickmap: GpuBrickmap,
    private readonly gbuffer: GBuffer,
  ) {}

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'primary.wgsl');
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
    this.camera = this.device.createBuffer({
      label: `${this.name}-camera`,
      size: CAMERA_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const faces = new Uint32Array(BLOCKS.length * 4);
    BLOCKS.forEach((b, i) => faces.set([b.materials.top, b.materials.side, b.materials.bottom, 0], i * 4));
    this.faces = this.upload(`${this.name}-faces`, faces);
    const albedo = new Float32Array(MATERIAL_NAMES.length * 4);
    MATERIAL_NAMES.forEach((m, i) => albedo.set([...srgbHexToLinear(MATERIAL_COLORS[m]), 1], i * 4));
    this.albedo = this.upload(`${this.name}-albedo`, albedo);
  }

  resize(): void {
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion) return;
    this.bindGroup = this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera } },
        { binding: 1, resource: g.gbuffer0.createView() },
        { binding: 2, resource: g.depth.createView() },
        { binding: 3, resource: g.motion.createView() },
        { binding: 4, resource: { buffer: this.faces } },
        { binding: 5, resource: { buffer: this.albedo } },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this.bindGroup) return;
    const cam = ctx.camera;
    const f32 = new Float32Array(this.cameraData);
    const u32 = new Uint32Array(this.cameraData);
    const i32 = new Int32Array(this.cameraData);
    f32.set(cam.invViewProj, 0);
    f32.set(cam.viewProj, 16);
    f32.set(cam.prevViewProj, 32);
    i32.set(cam.cell, 48);
    u32[51] = config.trace.maxSteps;
    f32.set(cam.frac, 52);
    f32[55] = config.camera.far;
    f32.set(cam.prevDelta, 56);
    f32.set(cam.forward, 60);
    u32[64] = this.gbuffer.width;
    u32[65] = this.gbuffer.height;
    this.device.queue.writeBuffer(this.camera, 0, this.cameraData);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(1, this.brickmap.bindGroup(this.brickmapLayout));
    pass.dispatchWorkgroups(
      Math.ceil(this.gbuffer.width / this.workgroupSize),
      Math.ceil(this.gbuffer.height / this.workgroupSize),
    );
    pass.end();
  }

  private upload(label: string, data: Uint32Array | Float32Array): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
}
