import { config, type DebugView } from '../config';
import type { CameraFrame, FlyCamera } from '../player/camera';
import type { GpuBrickmap } from './brickmap';
import type { MaterialSystem } from './materials';
import { AerialPerspectivePass } from './passes/aerial-perspective-pass';
import { ExposurePass } from './passes/exposure-pass';
import { LightingPass } from './passes/lighting-pass';
import { VisibilityPass } from './passes/visibility-pass';
import type { SkySystem } from './sky';
import { CanvasSize } from './canvas-size';
import type { GpuContext } from './device';
import { GB_STEPS_BITS, GB_STEPS_MAX, GB_STEPS_SHIFT, GBuffer } from './gbuffer';
import { BlitPass } from './passes/blit-pass';
import { GBufferViewPass } from './passes/gbuffer-view-pass';
import { GradientPass } from './passes/gradient-pass';
import type { FrameContext, RenderPass } from './passes/pass';
import { PrimaryPass } from './passes/primary-pass';
import { TopdownPass } from './passes/topdown-pass';
import { GpuProfiler } from './profiler';

/** A chain of passes whose last one writes `output` (the scene texture for the blit). */
interface SceneChain {
  passes: RenderPass[];
  output: () => GPUTexture | null;
  /** Output is pre-exposed HDR: tone map it (otherwise it is shown as is). */
  tonemap: boolean;
  /** Needs this frame's sky LUTs / lighting summary. */
  sky: boolean;
}

type ChainName = 'lit' | 'visibility' | 'gbuffer' | 'topdown' | 'gradient';

/** Owns the pass chains: scene passes at internal resolution → blit to canvas. */
export class Renderer {
  readonly profiler: GpuProfiler;
  readonly size: CanvasSize;
  readonly gbuffer: GBuffer;
  private readonly chains: Record<ChainName, SceneChain>;
  private readonly lighting: LightingPass;
  private readonly visibility: VisibilityPass;
  private lastTime: number | null = null;
  private readonly allPasses: RenderPass[];
  private readonly blit: BlitPass;
  private resolutionDirty = true;
  private boundOutput: GPUTexture | null = null;
  private readonly primary: PrimaryPass;
  private lastCamera: CameraFrame | null = null;

  constructor(
    private readonly gpu: GpuContext,
    canvas: HTMLCanvasElement,
    brickmap: GpuBrickmap,
    materials: MaterialSystem,
    private readonly sky: SkySystem,
  ) {
    const { device } = gpu;
    this.profiler = new GpuProfiler(device, gpu.timestampQuery);
    this.size = new CanvasSize(canvas, device.limits.maxTextureDimension2D);
    this.gbuffer = new GBuffer(device);
    const primary = new PrimaryPass(device, brickmap, this.gbuffer, materials);
    this.primary = primary;
    const camera = () => primary.camera;
    const visibility = new VisibilityPass(device, this.gbuffer, camera, brickmap, materials, sky);
    const aerial = new AerialPerspectivePass(device, camera, sky);
    const lighting = new LightingPass(device, this.gbuffer, camera, visibility, sky, aerial);
    this.visibility = visibility;
    this.lighting = lighting;
    const exposure = new ExposurePass(device, () => lighting.output, this.gbuffer, sky);
    const view = new GBufferViewPass(device, this.gbuffer);
    const topdown = new TopdownPass(device, brickmap);
    const gradient = new GradientPass(device);
    const out = (p: { output: GPUTexture | null }) => () => p.output;
    this.chains = {
      lit: { passes: [primary, visibility, aerial, lighting, exposure], output: out(lighting), tonemap: true, sky: true },
      visibility: { passes: [primary, visibility, aerial, lighting], output: out(lighting), tonemap: false, sky: true },
      gbuffer: { passes: [primary, view], output: out(view), tonemap: false, sky: false },
      topdown: { passes: [topdown], output: out(topdown), tonemap: false, sky: false },
      gradient: { passes: [gradient], output: out(gradient), tonemap: false, sky: false },
    };
    // Resize order matters: the G-buffer consumers after the passes they read from.
    this.allPasses = [primary, visibility, aerial, lighting, exposure, view, topdown, gradient];
    this.blit = new BlitPass(device, gpu.context, gpu.format, sky.exposure);
  }

  async init(): Promise<void> {
    await Promise.all([...this.allPasses.map((p) => p.init()), this.blit.init()]);
  }

  /** Forces render targets to be rebuilt on the next frame (e.g. renderScale changed). */
  invalidateResolution(): void {
    this.size.remeasure();
    this.resolutionDirty = true;
  }

  /** Internal (traced) resolution. */
  get internalSize(): { width: number; height: number } {
    return { width: this.gbuffer.width, height: this.gbuffer.height };
  }

  render(time: number, camera: FlyCamera, alpha: number): void {
    if (this.size.apply() || this.resolutionDirty) this.resizeTargets();
    const view = config.debug.view;
    const chain = this.chainFor(view);
    this.lighting.mode = view === 'shadow' || view === 'skyvis' || view === 'history' ? view : 'lit';
    this.blit.setTonemap(chain.tonemap ? config.render.tonemapper : 'none');
    const output = chain.output();
    if (output !== this.boundOutput && output) {
      this.blit.setSource(output);
      this.boundOutput = output;
    }

    const { device } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'frame' });
    const frame = camera.frame(alpha, this.gbuffer.width, this.gbuffer.height);
    this.lastCamera = frame;
    const dt = this.lastTime === null ? 0 : Math.max(0, time - this.lastTime);
    this.lastTime = time;
    const ctx: FrameContext = { encoder, profiler: this.profiler, time, dt, camera: frame };

    this.profiler.beginFrame();
    if (chain.sky) this.sky.encode(encoder);
    for (const pass of chain.passes) pass.execute(ctx);
    this.blit.execute(ctx);
    this.profiler.resolve(encoder);

    device.queue.submit([encoder.finish()]);
    this.profiler.afterSubmit();
  }

  /**
   * Runs only the primary pass `iterations` times back to back (same camera as the last
   * frame) and reports the time per pass: from timestamps around the whole batch when
   * available, and from wall-clock submit → completion. Unlike the per-pass profiler this
   * does not pick up waits on neighbouring passes.
   */
  async benchmarkPrimary(iterations: number): Promise<{ gpuMs: number | null; wallMs: number; avgSteps: number; p95Steps: number }> {
    const timing = await this.benchmarkPasses([this.primary], iterations);
    return { ...timing, ...(await this.stepStats()) };
  }

  /** Times the lighting passes after primary (visibility rays + accumulation, shading). */
  async benchmarkLighting(iterations: number): Promise<{ visibilityMs: number | null; lightingMs: number | null }> {
    const [vis, light] = [this.visibility, this.lighting];
    return {
      visibilityMs: (await this.benchmarkPasses([vis], iterations)).gpuMs,
      lightingMs: (await this.benchmarkPasses([light], iterations)).gpuMs,
    };
  }

  /**
   * Runs `passes` `iterations` times back to back (same camera as the last frame) and
   * reports the time per iteration, from timestamps around the batch and wall clock.
   */
  private async benchmarkPasses(passes: RenderPass[], iterations: number): Promise<{ gpuMs: number | null; wallMs: number }> {
    const camera = this.lastCamera;
    if (!camera) throw new Error('no frame rendered yet');
    const { device } = this.gpu;
    await device.queue.onSubmittedWorkDone();

    const querySet = this.gpu.timestampQuery ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
    let pass = 0;
    const profiler = {
      timestampWrites: () => {
        const i = pass++;
        const last = iterations * passes.length - 1;
        if (!querySet || (i !== 0 && i !== last)) return undefined;
        return {
          querySet,
          ...(i === 0 && { beginningOfPassWriteIndex: 0 }),
          ...(i === last && { endOfPassWriteIndex: 1 }),
        };
      },
    };
    const encoder = device.createCommandEncoder({ label: 'benchmark-primary' });
    const ctx: FrameContext = { encoder, profiler, time: 0, dt: 0, camera };
    for (let i = 0; i < iterations; i++) for (const pass of passes) pass.execute(ctx);

    let resolve: GPUBuffer | null = null;
    let read: GPUBuffer | null = null;
    if (querySet) {
      resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      read = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 16);
    }
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wallMs = (performance.now() - start) / iterations;

    let gpuMs: number | null = null;
    if (read) {
      await read.mapAsync(GPUMapMode.READ);
      const [t0, t1] = new BigInt64Array(read.getMappedRange());
      gpuMs = Number(t1! - t0!) / 1e6 / iterations;
      read.unmap();
    }
    for (const b of [resolve, read]) b?.destroy();
    querySet?.destroy();
    return { gpuMs, wallMs };
  }

  /** Mean and 95th percentile of the DDA step count stored in the G-buffer. */
  private async stepStats(): Promise<{ avgSteps: number; p95Steps: number }> {
    const g = this.gbuffer.gbuffer0;
    if (!g) return { avgSteps: 0, p95Steps: 0 };
    const { device } = this.gpu;
    const row = Math.ceil((g.width * 16) / 256) * 256;
    const buf = device.createBuffer({ size: row * g.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: g }, { buffer: buf, bytesPerRow: row }, { width: g.width, height: g.height });
    device.queue.submit([encoder.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(buf.getMappedRange());
    const histogram = new Uint32Array(GB_STEPS_MAX + 1);
    let sum = 0;
    for (let y = 0; y < g.height; y++) {
      for (let x = 0; x < g.width; x++) {
        const steps = (words[(y * row) / 4 + x * 4 + 3]! >>> GB_STEPS_SHIFT) & GB_STEPS_MAX;
        histogram[steps]!++;
        sum += steps;
      }
    }
    buf.unmap();
    buf.destroy();
    const n = g.width * g.height;
    let acc = 0;
    let p95 = 0;
    while (p95 < histogram.length && (acc += histogram[p95]!) < 0.95 * n) p95++;
    return { avgSteps: sum / n, p95Steps: p95 };
  }

  /**
   * Checks the depth prepass: renders the G-buffer without and with it (same camera as the
   * last frame) and compares every pixel. Everything except the DDA step count must match.
   */
  async verifyPrepass(): Promise<{ pixels: number; mismatches: number; first?: string }> {
    const camera = this.lastCamera;
    const g = this.gbuffer;
    if (!camera || !g.gbuffer0 || !g.depth) throw new Error('no frame rendered yet');
    const { device } = this.gpu;
    const { width, height } = g;
    const rowG = Math.ceil((width * 16) / 256) * 256;
    const rowD = Math.ceil((width * 4) / 256) * 256;
    const readback = () => ({
      g: device.createBuffer({ size: rowG * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      d: device.createBuffer({ size: rowD * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    });
    const runs = [readback(), readback()];
    const noTimestamps = { timestampWrites: () => undefined };
    [false, true].forEach((prepass, i) => {
      const encoder = device.createCommandEncoder({ label: `verify-prepass-${prepass}` });
      this.primary.encode({ encoder, profiler: noTimestamps, time: 0, dt: 0, camera }, prepass);
      encoder.copyTextureToBuffer({ texture: g.gbuffer0! }, { buffer: runs[i]!.g, bytesPerRow: rowG }, { width, height });
      encoder.copyTextureToBuffer({ texture: g.depth! }, { buffer: runs[i]!.d, bytesPerRow: rowD }, { width, height });
      device.queue.submit([encoder.finish()]);
    });
    await Promise.all(runs.flatMap((r) => [r.g.mapAsync(GPUMapMode.READ), r.d.mapAsync(GPUMapMode.READ)]));
    const [a, b] = runs.map((r) => ({ g: new Uint32Array(r.g.getMappedRange()), d: new Float32Array(r.d.getMappedRange()) }));
    let mismatches = 0;
    let first: string | undefined;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const gi = (y * rowG) / 4 + x * 4;
        const di = (y * rowD) / 4 + x;
        const same =
          a!.g[gi] === b!.g[gi] &&
          a!.g[gi + 1] === b!.g[gi + 1] &&
          a!.g[gi + 2] === b!.g[gi + 2] &&
          (a!.g[gi + 3]! & ~GB_STEPS_BITS) === (b!.g[gi + 3]! & ~GB_STEPS_BITS) &&
          a!.d[di] === b!.d[di];
        if (!same) {
          mismatches++;
          first ??= `(${x}, ${y}) depth ${a!.d[di]} vs ${b!.d[di]}, block ${a!.g[gi + 3]! & 0xff} vs ${b!.g[gi + 3]! & 0xff}`;
        }
      }
    }
    for (const r of runs) {
      r.g.destroy();
      r.d.destroy();
    }
    return { pixels: width * height, mismatches, ...(first !== undefined && { first }) };
  }

  private probePending = false;

  /** G-buffer albedo (linear luminance) and material at the screen centre (debug panel). */
  async probeCenter(): Promise<{ albedo: number; material: number; sky: boolean } | null> {
    const g = this.gbuffer.gbuffer0;
    if (!g || this.probePending) return null;
    this.probePending = true;
    const { device } = this.gpu;
    const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture: g, origin: { x: g.width >> 1, y: g.height >> 1 } },
      { buffer: buf, bytesPerRow: 256 },
      { width: 1, height: 1 },
    );
    device.queue.submit([encoder.finish()]);
    try {
      await buf.mapAsync(GPUMapMode.READ);
      const [x = 0, , , w = 0] = new Uint32Array(buf.getMappedRange().slice(0));
      // gbuffer.wgsl: x = pack4x8unorm(linear albedo, AO); w bits 8-15 material, 31 sky.
      const rgb = [x & 0xff, (x >> 8) & 0xff, (x >> 16) & 0xff].map((c) => c / 255);
      return {
        albedo: 0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!,
        material: (w >>> 8) & 0xff,
        sky: (w >>> 31) === 1,
      };
    } finally {
      buf.destroy();
      this.probePending = false;
    }
  }

  private chainFor(view: DebugView): SceneChain {
    if (view === 'lit') return this.chains.lit;
    if (view === 'shadow' || view === 'skyvis' || view === 'history') return this.chains.visibility;
    if (view === 'topdown') return this.chains.topdown;
    if (view === 'gradient') return this.chains.gradient;
    return this.chains.gbuffer;
  }

  private resizeTargets(): void {
    this.resolutionDirty = false;
    const maxDim = this.gpu.device.limits.maxTextureDimension2D;
    const scale = config.render.renderScale;
    const w = Math.max(1, Math.min(Math.round(this.size.width * scale), maxDim));
    const h = Math.max(1, Math.min(Math.round(this.size.height * scale), maxDim));

    this.gbuffer.resize(w, h); // before the passes that bind it
    for (const pass of this.allPasses) pass.resize(w, h);
    this.blit.resize();
    this.boundOutput = null;
  }
}
