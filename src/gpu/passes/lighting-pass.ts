import { config } from '../../config';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import { SCENE_FORMAT } from './gradient-pass';
import type { FrameContext, RenderPass } from './pass';
import type { VisibilityPass } from './visibility-pass';

export type LightingMode = 'lit' | 'shadow' | 'skyvis';
const MODES: readonly LightingMode[] = ['lit', 'shadow', 'skyvis'];

/** Shades the G-buffer (lighting.wgsl) into pre-exposed HDR, or shows visibility (debug). */
export class LightingPass implements RenderPass {
  readonly name = 'lighting';
  output: GPUTexture | null = null;
  mode: LightingMode = 'lit';

  private pipeline!: GPUComputePipeline;
  private skyLayout!: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private groups: [GPUBindGroup, GPUBindGroup] | null = null;
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    private readonly visibility: VisibilityPass,
    private readonly sky: SkySystem,
  ) {
    this.params = device.createBuffer({ label: 'lighting-params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'lighting.wgsl');
    const [wx, wy] = this.workgroup;
    this.pipeline = await this.device.createComputePipelineAsync({
      label: this.name,
      layout: 'auto',
      compute: { module, entryPoint: 'main', constants: { WORKGROUP_X: wx, WORKGROUP_Y: wy } },
    });
    this.skyLayout = this.pipeline.getBindGroupLayout(3);
  }

  resize(width: number, height: number): void {
    const g = this.gbuffer;
    if (!g.gbuffer0) return;
    this.output?.destroy();
    this.output = this.device.createTexture({
      label: `${this.name}-output`,
      size: { width, height },
      format: SCENE_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.groups = null; // visibility history textures are (re)created in its resize
  }

  execute(ctx: FrameContext): void {
    const g = this.gbuffer;
    if (!this.output || !g.gbuffer0) return;
    // Bind groups per visibility history texture (it ping-pongs between two).
    if (!this.groups) {
      const make = (index: number) =>
        this.device.createBindGroup({
          label: this.name,
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.camera() } },
            { binding: 1, resource: { buffer: this.params } },
            { binding: 2, resource: g.gbuffer0!.createView() },
            { binding: 3, resource: this.visibility.historyTexture(index).createView() },
            { binding: 4, resource: this.output!.createView() },
          ],
        });
      this.groups = [make(0), make(1)];
    }
    const l = config.lighting;
    const p = new ArrayBuffer(16);
    new Uint32Array(p)[0] = MODES.indexOf(this.mode);
    new Float32Array(p).set([l.emissiveStrength, l.subsurface], 1);
    this.device.queue.writeBuffer(this.params, 0, p);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.groups[this.visibility.outputIndex]!);
    pass.setBindGroup(3, this.sky.bindGroup(this.skyLayout));
    pass.dispatchWorkgroups(Math.ceil(this.output.width / this.workgroup[0]), Math.ceil(this.output.height / this.workgroup[1]));
    pass.end();
  }
}
