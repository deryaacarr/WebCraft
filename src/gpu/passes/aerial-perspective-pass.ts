import { config } from '../../config';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import type { FrameContext, RenderPass } from './pass';

// AerialParams in aerial-perspective.wgsl.
const PARAMS_SIZE = 16;

/**
 * Aerial perspective froxel LUT (aerial-perspective.wgsl), rebuilt every frame from the
 * camera and the sky; sampled by the lighting pass. Runs after the primary pass (camera).
 */
export class AerialPerspectivePass implements RenderPass {
  readonly name = 'aerial';

  private pipeline!: GPUComputePipeline;
  private skyLayout!: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private group: GPUBindGroup | null = null;
  private volume: GPUTexture | null = null;
  private volumeKey = '';

  constructor(
    private readonly device: GPUDevice,
    private readonly camera: () => GPUBuffer,
    private readonly sky: SkySystem,
  ) {
    this.params = device.createBuffer({ label: 'aerial-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** The froxel volume (rgb in-scattering pre-exposed, a transmittance). */
  get output(): GPUTexture {
    const a = config.sky.aerialPerspective;
    const key = `${a.resolution}x${a.slices}`;
    if (!this.volume || key !== this.volumeKey) {
      this.volume?.destroy();
      this.volume = this.device.createTexture({
        label: 'aerial-perspective',
        size: { width: a.resolution, height: a.resolution, depthOrArrayLayers: a.slices },
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.volumeKey = key;
      this.group = null;
      this.onVolumeChange?.();
    }
    return this.volume;
  }

  /** Called when the volume is recreated (bind groups sampling it must be rebuilt). */
  onVolumeChange: (() => void) | null = null;

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'aerial-perspective.wgsl');
    this.pipeline = await this.device.createComputePipelineAsync({
      label: this.name,
      layout: 'auto',
      compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: config.render.workgroupSize } },
    });
    this.skyLayout = this.pipeline.getBindGroupLayout(3);
  }

  resize(): void {}

  execute(ctx: FrameContext): void {
    const a = config.sky.aerialPerspective;
    if (!a.enabled) return;
    const volume = this.output;
    this.group ??= this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera() } },
        { binding: 1, resource: { buffer: this.params } },
        { binding: 2, resource: this.sky.multiscatterView },
        { binding: 3, resource: volume.createView() },
      ],
    });
    const p = new ArrayBuffer(PARAMS_SIZE);
    new Float32Array(p)[0] = a.maxDepthKm;
    new Uint32Array(p)[1] = a.samplesPerSlice;
    this.device.queue.writeBuffer(this.params, 0, p);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.setBindGroup(3, this.sky.bindGroup(this.skyLayout, [0, 1, 2, 4, 6]));
    const w = config.render.workgroupSize;
    pass.dispatchWorkgroups(Math.ceil(volume.width / w), Math.ceil(volume.height / w));
    pass.end();
  }
}
