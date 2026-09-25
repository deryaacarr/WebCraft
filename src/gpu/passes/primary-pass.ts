import { config } from '../../config';
import { BLOCKS, MATERIAL_COLORS, MATERIAL_NAMES } from '../../world/blocks';
import type { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import { traceConstants } from '../trace-constants';
import type { FrameContext, RenderPass } from './pass';

// Camera struct in camera.wgsl: 3 × mat4x4f (192 B) + 5 × 16 B.
const CAMERA_SIZE = 272;
const DEG = Math.PI / 180;

/** sRGB 0xRRGGBB → linear [r, g, b]. */
export function srgbHexToLinear(hex: number): [number, number, number] {
  const c = (v: number) => {
    const s = v / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return [c((hex >> 16) & 0xff), c((hex >> 8) & 0xff), c(hex & 0xff)];
}

/**
 * Traces one ray per pixel through the brickmap and fills the G-buffer. A depth prepass
 * at 1/prepassTile resolution first finds, per tile, how far all its rays can safely skip.
 */
export class PrimaryPass implements RenderPass {
  readonly name = 'primary';

  private pipeline!: GPUComputePipeline;
  private prepassPipeline!: GPUComputePipeline;
  private prepassBindGroup: GPUBindGroup | null = null;
  private coarse: GPUTexture | null = null;
  private brickmapLayout!: GPUBindGroupLayout;
  private prepassLayout!: GPUBindGroupLayout;
  private camera!: GPUBuffer;
  private faces!: GPUBuffer;
  private albedo!: GPUBuffer;
  private bindGroup: GPUBindGroup | null = null;
  private readonly cameraData = new ArrayBuffer(CAMERA_SIZE);
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly brickmap: GpuBrickmap,
    private readonly gbuffer: GBuffer,
  ) {}

  async init(): Promise<void> {
    const [module, prepassModule] = await Promise.all([
      createShaderModule(this.device, 'primary.wgsl'),
      createShaderModule(this.device, 'prepass.wgsl'),
    ]);
    const [wx, wy] = this.workgroup;
    const constants = { WORKGROUP_X: wx, WORKGROUP_Y: wy, ...traceConstants() };
    [this.pipeline, this.prepassPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({ label: this.name, layout: 'auto', compute: { module, entryPoint: 'main', constants } }),
      this.device.createComputePipelineAsync({
        label: `${this.name}-prepass`,
        layout: 'auto',
        compute: { module: prepassModule, entryPoint: 'main', constants },
      }),
    ]);
    this.prepassLayout = this.prepassPipeline.getBindGroupLayout(1);
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

  resize(width: number, height: number): void {
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion) return;
    const tile = config.trace.prepassTile;
    this.coarse?.destroy();
    this.coarse = this.device.createTexture({
      label: `${this.name}-coarse`,
      size: { width: Math.ceil(width / tile), height: Math.ceil(height / tile) },
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.prepassBindGroup = this.device.createBindGroup({
      label: `${this.name}-prepass`,
      layout: this.prepassPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera } },
        { binding: 1, resource: this.coarse.createView() },
      ],
    });
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
        { binding: 6, resource: this.coarse.createView() },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    this.encode(ctx, config.trace.prepass);
  }

  /** Records the pass; `prepass` can be forced on/off (the prepass verify tool compares both). */
  encode(ctx: FrameContext, prepass: boolean): void {
    if (!this.bindGroup || !this.prepassBindGroup || !this.coarse) return;
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
    const tile = config.trace.prepassTile;
    // Every ray of a tile passes within half a tile diagonal (in pixels) of its centre ray;
    // at distance t that is at most t · (pixels · tan-space size of one pixel).
    const pixelTan = (2 * Math.tan((config.camera.fovY * DEG) / 2)) / this.gbuffer.height;
    f32[59] = ((tile * Math.SQRT2) / 2) * pixelTan * config.trace.prepassConeMargin;
    f32[63] = config.trace.prepassSafety;
    u32[66] = prepass ? tile : 0;
    u32[67] = config.trace.prepassMaxSteps;
    this.device.queue.writeBuffer(this.camera, 0, this.cameraData);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    if (prepass) {
      pass.setPipeline(this.prepassPipeline);
      pass.setBindGroup(0, this.prepassBindGroup);
      // The prepass reads only pointers (params + grid), never voxels.
      pass.setBindGroup(1, this.brickmap.bindGroup(this.prepassLayout, [0, 1]));
      pass.dispatchWorkgroups(Math.ceil(this.coarse.width / this.workgroup[0]), Math.ceil(this.coarse.height / this.workgroup[1]));
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(1, this.brickmap.bindGroup(this.brickmapLayout));
    pass.dispatchWorkgroups(Math.ceil(this.gbuffer.width / this.workgroup[0]), Math.ceil(this.gbuffer.height / this.workgroup[1]));
    pass.end();
  }

  private upload(label: string, data: Uint32Array | Float32Array): GPUBuffer {
    const buffer = this.device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
}
