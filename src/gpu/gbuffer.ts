/** Bit layout of gbuffer0.w (mirrors gbuffer.wgsl). */
export const GB_STEPS_SHIFT = 19;
export const GB_STEPS_MAX = 0x7ff;
export const GB_STEPS_BITS = GB_STEPS_MAX << GB_STEPS_SHIFT;

/**
 * G-buffer written by the primary ray pass (layout documented in gbuffer.wgsl), plus the
 * water samples of pixels seen through a water surface (water.wgsl).
 */
export class GBuffer {
  gbuffer0: GPUTexture | null = null;
  depth: GPUTexture | null = null;
  motion: GPUTexture | null = null;
  /** Water samples (water.wgsl), cleared every frame by the primary pass. */
  water: GPUTexture | null = null;

  constructor(private readonly device: GPUDevice) {}

  get width(): number {
    return this.gbuffer0?.width ?? 0;
  }

  get height(): number {
    return this.gbuffer0?.height ?? 0;
  }

  /** Bytes per pixel across all targets (for memory reporting). */
  static readonly BYTES_PER_PIXEL = 16 + 4 + 8 + 16;

  resize(width: number, height: number): void {
    for (const t of [this.gbuffer0, this.depth, this.motion, this.water]) t?.destroy();
    const make = (label: string, format: GPUTextureFormat, extra = 0) =>
      this.device.createTexture({
        label,
        size: { width, height },
        format,
        // COPY_SRC: debug tools read the G-buffer back.
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | extra,
      });
    this.gbuffer0 = make('gbuffer0', 'rgba32uint');
    this.depth = make('gbuffer-depth', 'r32float');
    this.motion = make('gbuffer-motion', 'rg32float');
    // RENDER_ATTACHMENT: cleared with a render pass (only water pixels are written).
    this.water = make('gbuffer-water', 'rgba32float', GPUTextureUsage.RENDER_ATTACHMENT);
  }
}
