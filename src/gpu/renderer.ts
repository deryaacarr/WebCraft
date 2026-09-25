import { config } from '../config';
import { CanvasSize } from './canvas-size';
import type { GpuContext } from './device';
import { BlitPass } from './passes/blit-pass';
import { GradientPass } from './passes/gradient-pass';
import { GpuProfiler } from './profiler';

/** Owns the pass chain: compute passes at internal resolution → blit to canvas. */
export class Renderer {
  readonly profiler: GpuProfiler;
  readonly size: CanvasSize;
  private readonly gradient: GradientPass;
  private readonly blit: BlitPass;
  private resolutionDirty = true;

  constructor(
    private readonly gpu: GpuContext,
    canvas: HTMLCanvasElement,
  ) {
    this.profiler = new GpuProfiler(gpu.device, gpu.timestampQuery);
    this.size = new CanvasSize(canvas, gpu.device.limits.maxTextureDimension2D);
    this.gradient = new GradientPass(gpu.device);
    this.blit = new BlitPass(gpu.device, gpu.context, gpu.format);
  }

  async init(): Promise<void> {
    await Promise.all([this.gradient.init(), this.blit.init()]);
  }

  /** Forces render targets to be rebuilt on the next frame (e.g. renderScale changed). */
  invalidateResolution(): void {
    this.size.remeasure();
    this.resolutionDirty = true;
  }

  render(time: number): void {
    if (this.size.apply() || this.resolutionDirty) this.resizeTargets();

    const { device } = this.gpu;
    const encoder = device.createCommandEncoder({ label: 'frame' });
    const ctx = { encoder, profiler: this.profiler, time };

    this.profiler.beginFrame();
    this.gradient.execute(ctx);
    this.blit.execute(ctx);
    this.profiler.resolve(encoder);

    device.queue.submit([encoder.finish()]);
    this.profiler.afterSubmit();
  }

  private resizeTargets(): void {
    this.resolutionDirty = false;
    const maxDim = this.gpu.device.limits.maxTextureDimension2D;
    const scale = config.render.renderScale;
    const w = Math.max(1, Math.min(Math.round(this.size.width * scale), maxDim));
    const h = Math.max(1, Math.min(Math.round(this.size.height * scale), maxDim));

    this.gradient.resize(w, h);
    this.blit.resize();
    if (this.gradient.output) this.blit.setSource(this.gradient.output);
  }
}
