import { config } from '../../config';
import { kelvinToLinearRgb } from '../../core/color-temperature';
import { BLOCKS } from '../../world/blocks';
import type { EmitterRegistry } from '../../world/emitters';
import type { GpuBrickmap } from '../brickmap';
import type { GBuffer } from '../gbuffer';
import type { MaterialSystem } from '../materials';
import { createShaderModule } from '../shader';
import type { SkySystem } from '../sky';
import { traceConstants } from '../trace-constants';
import type { FrameContext, RenderPass } from './pass';

// GiParams in gi-common.wgsl: 32 scalars.
const PARAMS_SIZE = 128;
const BLUE_NOISE_URL = `${import.meta.env.BASE_URL}blue-noise.bin`;
const BLUE_NOISE = { size: 64, slices: 16 };

type Stage = 'all' | 'trace' | 'denoise';

/** Textures the shading pass reads: GI irradiance / radiance and the half-res guide. */
export interface GiOutputs {
  diffuse: GPUTexture;
  specular: GPUTexture;
  /** Emissive direct light (ReSTIR DI), without flicker. */
  emissive: GPUTexture;
  guide: GPUTexture;
}

/**
 * Global illumination at half resolution (see restir.wgsl, gi-trace.wgsl, gi-denoise.wgsl):
 * guide + ReSTIR DI reservoirs → spatial reuse → path samples → temporal accumulation →
 * variance → à-trous iterations. Ping-pong textures alternate every frame (`cur`).
 */
export class GiPass implements RenderPass {
  readonly name = 'gi';
  /** Benchmarks run the stages separately. */
  stage: Stage = 'all';

  private pipelines!: Record<'prepare' | 'spatial' | 'trace' | 'temporal' | 'variance' | 'atrous', GPUComputePipeline>;
  private readonly params: GPUBuffer;
  private readonly historyExposure: GPUBuffer;
  private readonly blockRadiance: GPUBuffer;
  private lights: GPUBuffer;
  private lightCapacity = 0;
  private lightCount = 0;
  private lightVersion = -1;
  private lightCenter: [number, number, number] = [Infinity, Infinity, Infinity];
  private atrousParams: GPUBuffer[] = [];
  private blueNoise: GPUTexture;
  private tex: {
    guide: GPUTexture[];
    reservoirsInitial: GPUTexture;
    reservoirs: GPUTexture[];
    rawDiffuse: GPUTexture;
    rawSpecular: GPUTexture;
    rawEmissive: GPUTexture;
    accDiffuse: GPUTexture[];
    accSpecular: GPUTexture[];
    accEmissive: GPUTexture[];
    moments: GPUTexture[];
    filtDiffuse: GPUTexture[];
    filtSpecular: GPUTexture[];
    filtEmissive: GPUTexture[];
  } | null = null;
  private groups: Record<'prepare' | 'spatial' | 'trace' | 'temporal' | 'variance', GPUBindGroup[]> | null = null;
  private halfSize: [number, number] = [0, 0];
  private divisor = 0;
  /** Called after the GI textures were re-created outside a renderer resize. */
  onResize: (() => void) | null = null;
  private cur = 0;
  private readonly workgroup = config.trace.workgroup;

  constructor(
    private readonly device: GPUDevice,
    private readonly gbuffer: GBuffer,
    private readonly camera: () => GPUBuffer,
    private readonly brickmap: GpuBrickmap,
    private readonly materials: MaterialSystem,
    private readonly sky: SkySystem,
    private readonly emitters: EmitterRegistry,
    /** Last frame's lit image (screen-space reuse of bounce hits). */
    private readonly prevLit: () => GPUTexture | null,
  ) {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: 'gi-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.historyExposure = device.createBuffer({ label: 'gi-history-exposure', size: 16, usage: storage });
    device.queue.writeBuffer(this.historyExposure, 0, new Float32Array([1, 0, 0, 0]));
    this.blockRadiance = device.createBuffer({ label: 'gi-block-radiance', size: BLOCKS.length * 16, usage: storage });
    this.lights = this.createLightBuffer(1);
    this.fallbackLit = device.createTexture({ label: 'gi-no-lit', size: { width: 1, height: 1 }, format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING });
    // White noise until the blue noise file has loaded (or if it is missing).
    this.blueNoise = device.createTexture({
      label: 'blue-noise',
      size: { width: BLUE_NOISE.size, height: BLUE_NOISE.size, depthOrArrayLayers: BLUE_NOISE.slices },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const white = new Uint8Array(BLUE_NOISE.size * BLUE_NOISE.size * BLUE_NOISE.slices * 4);
    for (let i = 0; i < white.length; i++) white[i] = (Math.random() * 256) | 0;
    this.writeBlueNoise(white);
  }

  async init(): Promise<void> {
    const [restir, trace, denoise] = await Promise.all([
      createShaderModule(this.device, 'restir.wgsl'),
      createShaderModule(this.device, 'gi-trace.wgsl'),
      createShaderModule(this.device, 'gi-denoise.wgsl'),
    ]);
    const [wx, wy] = this.workgroup;
    const wg = { WORKGROUP_X: wx, WORKGROUP_Y: wy };
    const make = (module: GPUShaderModule, entryPoint: string, trace = false) =>
      this.device.createComputePipelineAsync({
        label: `gi-${entryPoint}`,
        layout: 'auto',
        compute: { module, entryPoint, constants: { ...wg, ...(trace && traceConstants()) } },
      });
    const [prepare, spatial, tracePipeline, temporal, variance, atrous] = await Promise.all([
      make(restir, 'prepare', true),
      make(restir, 'spatial', true),
      make(trace, 'main', true),
      make(denoise, 'temporal'),
      make(denoise, 'variance'),
      make(denoise, 'atrousStep'),
    ]);
    this.pipelines = { prepare, spatial, trace: tracePipeline, temporal, variance, atrous };
    void this.loadBlueNoise();
  }

  /** Emitted radiance per block id (rgb mean over a face, a = emissive coverage). */
  get emitterRadiance(): GPUBuffer {
    return this.blockRadiance;
  }

  /** Current GI result for shading (depends on the debug signal and frame parity). */
  get outputs(): GiOutputs | null {
    const t = this.tex;
    if (!t) return null;
    const signal = config.gi.debugSignal;
    const n = config.gi.atrousIterations;
    const diffuse = signal === 'raw' ? t.rawDiffuse : signal === 'accumulated' ? t.accDiffuse[this.cur]! : t.filtDiffuse[n % 2]!;
    const specular = signal === 'raw' ? t.rawSpecular : signal === 'accumulated' ? t.accSpecular[this.cur]! : t.filtSpecular[n % 2]!;
    const emissive = signal === 'raw' ? t.rawEmissive : signal === 'accumulated' ? t.accEmissive[this.cur]! : t.filtEmissive[n % 2]!;
    return { diffuse, specular, emissive, guide: t.guide[this.cur]! };
  }

  resize(width: number, height: number): void {
    const g = this.gbuffer;
    if (!g.gbuffer0 || !g.depth || !g.motion) return;
    if (this.tex) for (const v of Object.values(this.tex)) for (const t of [v].flat()) t.destroy();
    const div = config.gi.resolutionDivisor;
    this.divisor = div;
    const hw = Math.ceil(width / div);
    const hh = Math.ceil(height / div);
    this.halfSize = [hw, hh];
    const make = (label: string, format: GPUTextureFormat) =>
      this.device.createTexture({
        label: `gi-${label}`,
        size: { width: hw, height: hh },
        format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    const pair = (label: string, format: GPUTextureFormat) => [make(`${label}-a`, format), make(`${label}-b`, format)];
    const t = {
      guide: pair('guide', 'rgba32uint'),
      reservoirsInitial: make('reservoirs-initial', 'rgba32uint'),
      reservoirs: pair('reservoirs', 'rgba32uint'),
      rawDiffuse: make('raw-diffuse', 'rgba16float'),
      rawSpecular: make('raw-specular', 'rgba16float'),
      rawEmissive: make('raw-emissive', 'rgba16float'),
      accDiffuse: pair('acc-diffuse', 'rgba16float'),
      accSpecular: pair('acc-specular', 'rgba16float'),
      accEmissive: pair('acc-emissive', 'rgba16float'),
      moments: pair('moments', 'rgba32float'),
      filtDiffuse: pair('filt-diffuse', 'rgba16float'),
      filtSpecular: pair('filt-specular', 'rgba16float'),
      filtEmissive: pair('filt-emissive', 'rgba16float'),
    };
    this.tex = t;
    this.buildGroups();
  }

  execute(ctx: FrameContext): void {
    const t = this.tex;
    if (!t || !this.groups || !config.gi.enabled) return;
    if (config.gi.resolutionDivisor !== this.divisor) {
      // GI grid changed: rebuild the textures (the lighting pass follows via labels).
      this.resize(this.gbuffer.width, this.gbuffer.height);
      this.onResize?.();
    }
    this.updateLights(ctx);
    this.writeParams(ctx);
    this.cur = 1 - this.cur;
    const c = this.cur;
    const atrousGroups = this.atrousGroupsByParity[c]!;
    const [hw, hh] = this.halfSize;
    const x = Math.ceil(hw / this.workgroup[0]);
    const y = Math.ceil(hh / this.workgroup[1]);
    const sky = (p: GPUComputePipeline, bindings: number[]) => this.sky.bindGroup(p.getBindGroupLayout(3), bindings);

    if (this.stage !== 'denoise') {
      const tw = ctx.profiler.timestampWrites('gi-trace');
      const pass = ctx.encoder.beginComputePass({ label: 'gi-trace', ...(tw && { timestampWrites: tw }) });
      pass.setPipeline(this.pipelines.prepare);
      pass.setBindGroup(0, this.groups.prepare[c]!);
      pass.setBindGroup(1, this.brickmap.bindGroup(this.pipelines.prepare.getBindGroupLayout(1)));
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.pipelines.spatial);
      pass.setBindGroup(0, this.groups.spatial[c]!);
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.pipelines.trace);
      pass.setBindGroup(0, this.traceGroups()![c]!);
      // Compact dispatch over the traced pixels only (tracedPixel in gi-common.wgsl).
      pass.setBindGroup(1, this.brickmap.bindGroup(this.pipelines.trace.getBindGroupLayout(1)));
      pass.setBindGroup(2, this.materials.bindGroup(this.pipelines.trace.getBindGroupLayout(2), [1, 2, 3, 6]));
      pass.setBindGroup(3, sky(this.pipelines.trace, [0, 1, 3, 4, 5, 6]));
      const pattern = config.gi.tracePattern;
      const tw2 = pattern === 'all' ? hw : Math.ceil(hw / 2);
      const th2 = pattern === 'quarter' ? Math.ceil(hh / 2) : hh;
      pass.dispatchWorkgroups(Math.ceil(tw2 / this.workgroup[0]), Math.ceil(th2 / this.workgroup[1]));
      pass.end();
    }
    if (this.stage !== 'trace') {
      const tw = ctx.profiler.timestampWrites('gi-denoise');
      const pass = ctx.encoder.beginComputePass({ label: 'gi-denoise', ...(tw && { timestampWrites: tw }) });
      pass.setPipeline(this.pipelines.temporal);
      pass.setBindGroup(0, this.groups.temporal[c]!);
      pass.setBindGroup(3, sky(this.pipelines.temporal, [6]));
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.pipelines.variance);
      pass.setBindGroup(0, this.groups.variance[c]!);
      pass.setBindGroup(3, sky(this.pipelines.variance, [6]));
      pass.dispatchWorkgroups(x, y);
      pass.setPipeline(this.pipelines.atrous);
      for (let i = 0; i < config.gi.atrousIterations; i++) {
        pass.setBindGroup(0, atrousGroups[i]!);
        pass.dispatchWorkgroups(x, y);
      }
      pass.end();
    }
  }

  // ------------------------------------------------------------------ internals

  private buildGroups(): void {
    const t = this.tex!;
    const g = this.gbuffer;
    const p = this.pipelines;
    const group = (pipeline: GPUComputePipeline, label: string, entries: [number, GPUBindingResource][]) =>
      this.device.createBindGroup({
        label,
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map(([binding, resource]) => ({ binding, resource })),
      });
    const cam: [number, GPUBindingResource] = [0, { buffer: this.camera() }];
    const params: [number, GPUBindingResource] = [1, { buffer: this.params }];
    const v = (tex: GPUTexture) => tex.createView();
    const byParity = (build: (c: number, prev: number) => GPUBindGroup) => [build(0, 1), build(1, 0)];
    this.groups = {
      prepare: byParity((c, prev) =>
        group(p.prepare, 'gi-prepare', [
          cam, params,
          [2, v(g.gbuffer0!)], [3, v(g.depth!)], [4, v(g.motion!)],
          [5, v(t.guide[c]!)], [6, v(t.guide[prev]!)],
          [7, v(t.reservoirs[prev]!)], [8, v(t.reservoirsInitial)],
          [9, { buffer: this.lights }], [10, { buffer: this.blockRadiance }],
        ]),
      ),
      spatial: byParity((c) =>
        group(p.spatial, 'gi-spatial', [
          cam, params,
          [8, v(t.reservoirs[c]!)], [11, v(t.guide[c]!)], [12, v(t.reservoirsInitial)],
          [10, { buffer: this.blockRadiance }],
        ]),
      ),
      trace: [],
      temporal: byParity((c, prev) =>
        group(p.temporal, 'gi-temporal', [
          cam, params,
          [2, v(t.rawDiffuse)], [3, v(t.rawSpecular)],
          [4, v(t.guide[c]!)], [5, v(t.guide[prev]!)], [6, v(g.motion!)],
          [7, v(t.accDiffuse[prev]!)], [8, v(t.accSpecular[prev]!)], [9, v(t.moments[prev]!)],
          [10, v(t.accDiffuse[c]!)], [11, v(t.accSpecular[c]!)], [12, v(t.moments[c]!)],
          [13, { buffer: this.historyExposure }],
          [25, v(t.rawEmissive)], [26, v(t.accEmissive[prev]!)], [27, v(t.accEmissive[c]!)],
        ]),
      ),
      variance: byParity((c) =>
        group(p.variance, 'gi-variance', [
          params,
          [4, v(t.guide[c]!)],
          [14, v(t.accDiffuse[c]!)], [15, v(t.accSpecular[c]!)], [16, v(t.moments[c]!)],
          [17, v(t.filtDiffuse[0]!)], [18, v(t.filtSpecular[0]!)],
          [19, { buffer: this.historyExposure }],
          [28, v(t.accEmissive[c]!)], [29, v(t.filtEmissive[0]!)],
        ]),
      ),
    };
    this.buildAtrousGroups();
    this.traceLit = null;
  }

  /** Trace groups also bind last frame's lit image, which the lighting pass re-creates on
   *  resize after this pass: built on first use. */
  private traceGroups(): GPUBindGroup[] | null {
    const lit = this.prevLit() ?? this.fallbackLit;
    if (lit === this.traceLit && this.groups!.trace.length) return this.groups!.trace;
    const t = this.tex!;
    const g = this.gbuffer;
    const p = this.pipelines.trace;
    this.traceLit = lit;
    this.groups!.trace = [0, 1].map((c) =>
      this.device.createBindGroup({
        label: 'gi-trace',
        layout: p.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.camera() } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 2, resource: t.guide[c]!.createView() },
          { binding: 3, resource: t.reservoirs[c]!.createView() },
          { binding: 4, resource: t.rawDiffuse.createView() },
          { binding: 5, resource: t.rawSpecular.createView() },
          { binding: 6, resource: this.blueNoise.createView({ dimension: '2d-array' }) },
          { binding: 7, resource: { buffer: this.blockRadiance } },
          { binding: 8, resource: g.depth!.createView() },
          { binding: 9, resource: g.motion!.createView() },
          { binding: 10, resource: lit.createView() },
          { binding: 11, resource: { buffer: this.historyExposure } },
          { binding: 12, resource: t.rawEmissive.createView() },
        ],
      }),
    );
    return this.groups!.trace;
  }

  private traceLit: GPUTexture | null = null;
  /** Bound before the lighting pass has an output (screen reuse is off then). */
  private readonly fallbackLit: GPUTexture;

  /** À-trous bind groups per frame parity (guide) and iteration (ping-pong, step). */
  private buildAtrousGroups(): void {
    const t = this.tex!;
    const n = config.gi.atrousIterations;
    while (this.atrousParams.length < n) {
      this.atrousParams.push(this.device.createBuffer({ label: 'gi-atrous-step', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
    this.atrousGroupsByParity = [0, 1].map((c) =>
      Array.from({ length: n }, (_, i) =>
        this.device.createBindGroup({
          label: `gi-atrous-${i}`,
          layout: this.pipelines.atrous.getBindGroupLayout(0),
          entries: [
            { binding: 1, resource: { buffer: this.params } },
            { binding: 4, resource: t.guide[c]!.createView() },
            { binding: 20, resource: { buffer: this.atrousParams[i]! } },
            { binding: 21, resource: t.filtDiffuse[i % 2]!.createView() },
            { binding: 22, resource: t.filtSpecular[i % 2]!.createView() },
            { binding: 23, resource: t.filtDiffuse[(i + 1) % 2]!.createView() },
            { binding: 24, resource: t.filtSpecular[(i + 1) % 2]!.createView() },
            { binding: 30, resource: t.filtEmissive[i % 2]!.createView() },
            { binding: 31, resource: t.filtEmissive[(i + 1) % 2]!.createView() },
          ],
        }),
      ),
    );
    this.atrousIterations = n;
  }

  private atrousGroupsByParity: GPUBindGroup[][] = [];
  private atrousIterations = 0;

  private writeParams(ctx: FrameContext): void {
    const gi = config.gi;
    const r = gi.restir;
    const l = config.lighting;
    if (gi.atrousIterations !== this.atrousIterations) this.buildAtrousGroups();
    for (let i = 0; i < gi.atrousIterations; i++) {
      this.device.queue.writeBuffer(this.atrousParams[i]!, 0, new Uint32Array([1 << i, i < gi.specularIterations ? 1 : 0, 0, 0]));
    }
    const c = ctx.camera;
    const turn = Math.acos(Math.min(1, c.forward[0] * c.prevForward[0] + c.forward[1] * c.prevForward[1] + c.forward[2] * c.prevForward[2]));
    const cameraMotion = Math.min(1, Math.hypot(...c.prevDelta) / l.historyCameraSpeed + (turn * 180) / Math.PI / l.historyCameraTurn);
    const res = this.materials.stats.resolution;
    const buf = new ArrayBuffer(PARAMS_SIZE);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u.set([this.halfSize[0], this.halfSize[1], this.gbuffer.width, this.gbuffer.height, { all: 0, half: 1, quarter: 2 }[gi.tracePattern]]);
    f[5] = gi.range;
    u[6] = gi.maxSteps;
    f[7] = gi.shadowDistance;
    u[8] = gi.shadowMaxSteps;
    f[9] = gi.skyDistance;
    u[10] = gi.skyMaxSteps;
    f[11] = l.leafTransmission;
    f[12] = gi.specularThreshold;
    f[13] = Math.max(0, Math.log2(Math.max(res, 1) / gi.hitTexels));
    f[14] = gi.fireflyClamp;
    u[15] = this.lightCount;
    u[16] = r.enabled ? 1 : 0;
    u[17] = r.candidates;
    f[18] = r.candidates * r.temporalMaxMFactor;
    u[19] = r.spatialSamples;
    f[20] = r.spatialRadius;
    f[21] = gi.historyStill;
    f[22] = gi.historyMoving;
    f[23] = cameraMotion;
    f[24] = l.historyMotionPixels;
    f[25] = l.temporalDepthTolerance;
    f[26] = gi.sigmaLuminance;
    f[27] = gi.planeTolerance;
    f[28] = gi.sigmaLuminanceSpecular;
    f[29] = gi.specularRoughnessSigma;
    u[30] = gi.screenReuse && this.prevLit() ? 1 : 0;
    u[31] = this.divisor;
    this.device.queue.writeBuffer(this.params, 0, buf);

    // Emitted radiance per block: black-body colour × configured mean radiance (rgb) and
    // the emissive coverage of its texture (a; the lighting pass concentrates the light on
    // the glowing texels).
    const radiance = new Float32Array(BLOCKS.length * 4);
    const mats = this.materials.emissiveRadiance;
    for (const b of BLOCKS) {
      const e = l.emitters[b.name];
      if (b.emissive <= 0 || !e) continue;
      const rgb = kelvinToLinearRgb(e.temperature);
      radiance.set([rgb[0] * e.radiance, rgb[1] * e.radiance, rgb[2] * e.radiance, mats[b.materials.side * 4 + 3] ?? 0], b.id * 4);
    }
    this.device.queue.writeBuffer(this.blockRadiance, 0, radiance);
  }

  /** Rebuilds the light list when emitters changed or the camera moved far enough. */
  private updateLights(ctx: FrameContext): void {
    const r = config.gi.restir;
    const [x, y, z] = ctx.camera.position;
    const [cx, cy, cz] = this.lightCenter;
    const moved = Math.hypot(x - cx, y - cy, z - cz) > r.rebuildDistance;
    if (!moved && this.emitters.version === this.lightVersion) return;
    this.lightVersion = this.emitters.version;
    this.lightCenter = [x, y, z];
    const list = this.emitters.nearest(x, y, z, r.lightRadius, r.maxLights);
    this.lightCount = list.length;
    if (list.length > this.lightCapacity) {
      this.lights.destroy();
      this.lights = this.createLightBuffer(Math.max(list.length, r.maxLights));
      this.buildGroups();
    }
    if (!list.length) return;
    const data = new Int32Array(list.length * 4);
    list.forEach((e, i) => data.set([e.x, e.y, e.z, e.id], i * 4));
    this.device.queue.writeBuffer(this.lights, 0, data);
  }

  private createLightBuffer(count: number): GPUBuffer {
    this.lightCapacity = count;
    return this.device.createBuffer({ label: 'gi-lights', size: Math.max(1, count) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  private writeBlueNoise(data: Uint8Array): void {
    const { size, slices } = BLUE_NOISE;
    this.device.queue.writeTexture(
      { texture: this.blueNoise },
      data,
      { bytesPerRow: size * 4, rowsPerImage: size },
      { width: size, height: size, depthOrArrayLayers: slices },
    );
  }

  private async loadBlueNoise(): Promise<void> {
    const { size, slices } = BLUE_NOISE;
    try {
      const res = await fetch(BLUE_NOISE_URL);
      if (!res.ok) throw new Error(`${res.status}`);
      const data = new Uint8Array(await res.arrayBuffer());
      if (data.length !== size * size * slices * 4) throw new Error(`unexpected size ${data.length}`);
      this.writeBlueNoise(data);
    } catch (err) {
      console.warn('[gi] blue noise not loaded (npm run bluenoise); using white noise:', err);
    }
  }
}
