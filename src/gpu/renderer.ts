import { config, type DebugView } from '../config';
import type { FlyCamera } from '../player/camera';
import type { GpuBrickmap } from './brickmap';
import { CanvasSize } from './canvas-size';
import type { GpuContext } from './device';
import { GBuffer } from './gbuffer';
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
}

/** Owns the pass chains: scene passes at internal resolution → blit to canvas. */
export class Renderer {
  readonly profiler: GpuProfiler;
  readonly size: CanvasSize;
  readonly gbuffer: GBuffer;
  private readonly chains: { gbuffer: SceneChain; topdown: SceneChain; gradient: SceneChain };
  private readonly allPasses: RenderPass[];
  private readonly blit: BlitPass;
  private resolutionDirty = true;
  private boundOutput: GPUTexture | null = null;

  constructor(
    private readonly gpu: GpuContext,
    canvas: HTMLCanvasElement,
    brickmap: GpuBrickmap,
  ) {
    const { device } = gpu;
    this.profiler = new GpuProfiler(device, gpu.timestampQuery);
    this.size = new CanvasSize(canvas, device.limits.maxTextureDimension2D);
    this.gbuffer = new GBuffer(device);
    const primary = new PrimaryPass(device, brickmap, this.gbuffer);
    const view = new GBufferViewPass(device, this.gbuffer);
    const topdown = new TopdownPass(device, brickmap);
    const gradient = new GradientPass(device);
    this.chains = {
      gbuffer: { passes: [primary, view], output: () => view.output },
      topdown: { passes: [topdown], output: () => topdown.output },
      gradient: { passes: [gradient], output: () => gradient.output },
    };
    this.allPasses = [primary, view, topdown, gradient];
    this.blit = new BlitPass(device, gpu.context, gpu.format);
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
    const chain = this.chainFor(config.debug.view);
    const output = chain.output();
    if (output !== this.boundOutput && output) {
      this.blit.setSource(output);
      this.boundOutput = output;
    }

    const { device } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'frame' });
    const frame = camera.frame(alpha, this.gbuffer.width, this.gbuffer.height);
    const ctx: FrameContext = { encoder, profiler: this.profiler, time, camera: frame };

    this.profiler.beginFrame();
    for (const pass of chain.passes) pass.execute(ctx);
    this.blit.execute(ctx);
    this.profiler.resolve(encoder);

    device.queue.submit([encoder.finish()]);
    this.profiler.afterSubmit();
  }

  private chainFor(view: DebugView): SceneChain {
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
