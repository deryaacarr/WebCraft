import { config } from '../../config';
import type { GBuffer } from '../gbuffer';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import { SCENE_FORMAT } from './gradient-pass';
import type { FrameContext, RenderPass } from './pass';
import type { AerialPerspectivePass } from './aerial-perspective-pass';
import type { VisibilityPass } from './visibility-pass';

export type LightingMode = 'lit' | 'shadow' | 'skyvis' | 'history';
const MODES: readonly LightingMode[] = ['lit', 'shadow', 'skyvis', 'history'];
// LightingParams in lighting.wgsl.
const PARAMS_SIZE = 64;

/** Shades the G-buffer (lighting.wgsl) into pre-exposed HDR, or shows visibility (debug). */
export class LightingPass implements RenderPass {
  readonly name = 'lighting';
  output: GPUTexture | null = null;
  mode: LightingMode = 'lit';

  private pipeline!: GPUComputePipeline;
  private skyLayout!: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private group: GPUBindGroup | null = null;
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    private readonly visibility: VisibilityPass,
    private readonly sky: SkySystem,
    private readonly aerial: AerialPerspectivePass,
  ) {
    aerial.onVolumeChange = () => {
      this.group = null;
    };
    this.params = device.createBuffer({ label: 'lighting-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    this.group = null; // the visibility texture is (re)created in its resize
  }

  execute(ctx: FrameContext): void {
    const g = this.gbuffer;
    if (!this.output || !g.gbuffer0 || !g.depth) return;
    const vis = this.visibility.output;
    if (!vis) return;
    this.group ??= this.device.createBindGroup({
      label: this.name,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera() } },
        { binding: 1, resource: { buffer: this.params } },
        { binding: 2, resource: g.gbuffer0.createView() },
        { binding: 3, resource: vis.createView() },
        { binding: 4, resource: this.output.createView() },
        { binding: 5, resource: this.aerial.output.createView() },
        { binding: 6, resource: g.depth!.createView() },
      ],
    });
    const l = config.lighting;
    const p = new ArrayBuffer(PARAMS_SIZE);
    new Uint32Array(p)[0] = MODES.indexOf(this.mode);
    // The fill terms are zero unless the debug-fill comparison model is selected.
    const fill = l.model === 'debug-fill' ? l.debugFill : { ambientFloor: 0, bounceAlbedo: [0, 0, 0], bounceIsotropic: 0 };
    new Float32Array(p).set([l.emissiveStrength, l.subsurface, fill.ambientFloor, ...fill.bounceAlbedo, fill.bounceIsotropic, l.historyStill], 1);
    const aerial = config.sky.aerialPerspective;
    new Float32Array(p)[9] = aerial.maxDepthKm;
    new Uint32Array(p)[10] = aerial.enabled ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, p);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.setBindGroup(3, this.sky.bindGroup(this.skyLayout));
    pass.dispatchWorkgroups(Math.ceil(this.output.width / this.workgroup[0]), Math.ceil(this.output.height / this.workgroup[1]));
    pass.end();
  }
}
