import type { CameraFrame } from '../../player/camera';
import type { GpuProfiler } from '../profiler';

/** Per-frame data shared by every pass. */
export interface FrameContext {
  encoder: GPUCommandEncoder;
  profiler: GpuProfiler;
  /** Seconds since start, interpolated to the render moment. */
  time: number;
  camera: CameraFrame;
}

/** Every render pass is its own class with this lifecycle. */
export interface RenderPass {
  readonly name: string;
  /** Creates pipelines and long-lived resources. Called once. */
  init(): Promise<void>;
  /** (Re)creates size-dependent resources. Called before the first execute and on every resize. */
  resize(width: number, height: number): void;
  /** Records the pass into `ctx.encoder`. */
  execute(ctx: FrameContext): void;
}
