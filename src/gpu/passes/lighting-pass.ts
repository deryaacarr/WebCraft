import { config } from '../../config';
import type { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import type { MaterialSystem } from '../materials';
import { traceConstants } from '../trace-constants';
import type { WaterSystem } from '../water';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import { SCENE_FORMAT } from './gradient-pass';
import type { FrameContext, RenderPass } from './pass';
import type { AerialPerspectivePass } from './aerial-perspective-pass';
import type { GiPass } from './gi-pass';
import type { VisibilityPass } from './visibility-pass';
import type { WaterReflectionPass } from './water-reflection-pass';

export type LightingMode = 'lit' | 'shadow' | 'skyvis' | 'history' | 'gi';
const MODES: readonly LightingMode[] = ['lit', 'shadow', 'skyvis', 'history', 'gi'];
// LightingParams in lighting.wgsl.
const PARAMS_SIZE = 80;

/** Shades the G-buffer (lighting.wgsl) into pre-exposed HDR, or shows visibility (debug). */
export class LightingPass implements RenderPass {
  readonly name = 'lighting';
  output: GPUTexture | null = null;
  mode: LightingMode = 'lit';

  /** Pipeline variants by `${underwater}${wet}` (lighting.wgsl UNDERWATER_VIEW,
   *  WET_SURFACES), each with its layouts (auto layouts of different pipelines differ). */
  private readonly variants = new Map<string, { pipeline: GPUComputePipeline; layouts: GPUBindGroupLayout[] }>();
  private readonly params: GPUBuffer;
  /** Bind groups per set of GI textures (they alternate with the frame parity). */
  private readonly groups = new Map<string, GPUBindGroup>();
  /** Stand-ins bound while GI is off. */
  private readonly noGi: { diffuse: GPUTexture; specular: GPUTexture; emissive: GPUTexture; guide: GPUTexture };
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    private readonly visibility: VisibilityPass,
    private readonly sky: SkySystem,
    private readonly aerial: AerialPerspectivePass,
    private readonly gi: GiPass,
    /** Worn edges look up neighbouring blocks and material flags. */
    private readonly brickmap: GpuBrickmap,
    private readonly materials: MaterialSystem,
    private readonly water: WaterSystem,
    private readonly reflection: WaterReflectionPass,
  ) {
    aerial.onVolumeChange = () => {
      this.groups.clear();
    };
    reflection.onResize = () => {
      this.groups.clear();
    };
    gi.onResize = () => {
      this.groups.clear();
    };
    const tiny = (label: string, format: GPUTextureFormat) =>
      device.createTexture({ label, size: { width: 1, height: 1 }, format, usage: GPUTextureUsage.TEXTURE_BINDING });
    this.noGi = {
      diffuse: tiny('no-gi-diffuse', 'rgba16float'),
      specular: tiny('no-gi-specular', 'rgba16float'),
      emissive: tiny('no-gi-emissive', 'rgba16float'),
      guide: tiny('no-gi-guide', 'rgba32uint'),
    };
    this.params = device.createBuffer({ label: 'lighting-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'lighting.wgsl');
    const [wx, wy] = this.workgroup;
    await Promise.all(
      ['00', '01', '10', '11'].map(async (key) => {
        const pipeline = await this.device.createComputePipelineAsync({
          label: `${this.name}-${key}`,
          layout: 'auto',
          compute: {
            module,
            entryPoint: 'main',
            constants: { WORKGROUP_X: wx, WORKGROUP_Y: wy, ...traceConstants(), UNDERWATER_VIEW: Number(key[0]), WET_SURFACES: Number(key[1]) },
          },
        });
        this.variants.set(key, { pipeline, layouts: [0, 1, 2, 3].map((i) => pipeline.getBindGroupLayout(i)) });
      }),
    );
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
    this.groups.clear(); // the visibility and GI textures are (re)created in their resize
  }

  execute(ctx: FrameContext): void {
    const g = this.gbuffer;
    if (!this.output || !g.gbuffer0 || !g.depth || !g.water) return;
    const vis = this.visibility.output;
    if (!vis || !this.reflection.output) return;
    const giOn = config.gi.enabled;
    const gi = (giOn && this.gi.outputs) || this.noGi;
    const w = config.wetness;
    const variantKey = `${this.water.cameraUnder ? 1 : 0}${w.global > 0 || w.zones.length > 0 ? 1 : 0}`;
    const variant = this.variants.get(variantKey)!;
    const key = `${variantKey}|${vis.label}|${gi.diffuse.label}|${gi.specular.label}|${gi.emissive.label}|${gi.guide.label}|${this.reflection.output.label}`;
    let group = this.groups.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        label: this.name,
        layout: variant.layouts[0]!,
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 2, resource: g.gbuffer0.createView() },
          { binding: 3, resource: vis.createView() },
          { binding: 4, resource: this.output.createView() },
          { binding: 5, resource: this.aerial.output.createView() },
          { binding: 6, resource: g.depth!.createView() },
          { binding: 7, resource: gi.diffuse.createView() },
          { binding: 8, resource: gi.specular.createView() },
          { binding: 9, resource: gi.guide.createView() },
          { binding: 10, resource: gi.emissive.createView() },
          { binding: 11, resource: { buffer: this.gi.emitterRadiance } },
          { binding: 12, resource: g.water!.createView() },
          { binding: 13, resource: { buffer: this.water.params } },
          { binding: 14, resource: this.reflection.output.createView() },
        ],
      });
      this.groups.set(key, group);
    }
    const l = config.lighting;
    const p = new ArrayBuffer(PARAMS_SIZE);
    new Uint32Array(p)[0] = MODES.indexOf(this.mode);
    // The fill terms are zero unless the debug-fill comparison model is selected.
    const fill = l.model === 'debug-fill' ? l.debugFill : { ambientFloor: 0, bounceAlbedo: [0, 0, 0], bounceIsotropic: 0 };
    new Float32Array(p).set([ctx.time, l.subsurface, fill.ambientFloor, ...fill.bounceAlbedo, fill.bounceIsotropic, l.historyStill], 1);
    const aerial = config.sky.aerialPerspective;
    new Float32Array(p)[9] = aerial.maxDepthKm;
    new Uint32Array(p)[10] = aerial.enabled ? 1 : 0;
    new Uint32Array(p)[11] = giOn && this.gi.outputs ? 1 : 0;
    new Float32Array(p)[12] = config.gi.specularThreshold;
    new Float32Array(p)[13] = config.gi.planeTolerance;
    new Float32Array(p).set([l.flicker.amount, l.flicker.speed, l.flicker.scale, config.gi.resolutionDivisor], 14);
    this.device.queue.writeBuffer(this.params, 0, p);

    const timestampWrites = ctx.profiler.timestampWrites(this.name);
    const pass = ctx.encoder.beginComputePass({ label: this.name, ...(timestampWrites && { timestampWrites }) });
    pass.setPipeline(variant.pipeline);
    pass.setBindGroup(0, group);
    pass.setBindGroup(3, this.sky.bindGroup(variant.layouts[3]!));
    pass.setBindGroup(1, this.brickmap.bindGroup(variant.layouts[1]!, [0, 1, 2]));
    // Worn edges: params, material table, block table, noise volume, sampler.
    pass.setBindGroup(2, this.materials.bindGroup(variant.layouts[2]!, [0, 1, 2, 6, 10]));
    pass.dispatchWorkgroups(Math.ceil(this.output.width / this.workgroup[0]), Math.ceil(this.output.height / this.workgroup[1]));
    pass.end();
  }
}
