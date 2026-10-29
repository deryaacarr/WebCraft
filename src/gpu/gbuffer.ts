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

  /** Active region (dynamic resolution): passes render into the top-left `width` ×
   *  `height` of the allocated targets; last frame's for history lookups. */
  private active = { width: 0, height: 0 };
  private prevActive = { width: 0, height: 0 };

  constructor(private readonly device: GPUDevice) {}

  /** Active (rendered) size this frame. */
  get width(): number {
    return this.active.width;
  }

  get height(): number {
    return this.active.height;
  }

  /** Last frame's active size (where the histories' pixels are). */
  get prevWidth(): number {
    return this.prevActive.width;
  }

  get prevHeight(): number {
    return this.prevActive.height;
  }

  /** Allocated size (the largest active size). */
  get allocWidth(): number {
    return this.gbuffer0?.width ?? 0;
  }

  get allocHeight(): number {
    return this.gbuffer0?.height ?? 0;
  }

  /** Sets this frame's active region (clamped to the allocation); the old one becomes the
   *  previous frame's. */
  setActive(width: number, height: number): void {
    this.prevActive = this.active;
    this.active = { width: Math.max(1, Math.min(width, this.allocWidth)), height: Math.max(1, Math.min(height, this.allocHeight)) };
    if (this.prevActive.width === 0) this.prevActive = this.active;
  }

  /** Bytes per pixel across all targets (for memory reporting). */
  static readonly BYTES_PER_PIXEL = 16 + 4 + 8 + 16;

  resize(width: number, height: number): void {
    this.active = { width, height };
    this.prevActive = this.active;
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
