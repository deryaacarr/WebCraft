/**
 * G-buffer written by the primary ray pass (layout documented in gbuffer.wgsl).
 * Three storage textures stay within the default limit of 4 per shader stage.
 */
export class GBuffer {
  gbuffer0: GPUTexture | null = null;
  depth: GPUTexture | null = null;
  motion: GPUTexture | null = null;

  constructor(private readonly device: GPUDevice) {}

  get width(): number {
    return this.gbuffer0?.width ?? 0;
  }

  get height(): number {
    return this.gbuffer0?.height ?? 0;
  }

  /** Bytes per pixel across all targets (for memory reporting). */
  static readonly BYTES_PER_PIXEL = 16 + 4 + 8;

  resize(width: number, height: number): void {
    for (const t of [this.gbuffer0, this.depth, this.motion]) t?.destroy();
    const make = (label: string, format: GPUTextureFormat) =>
      this.device.createTexture({
        label,
        size: { width, height },
        format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.gbuffer0 = make('gbuffer0', 'rgba32uint');
    this.depth = make('gbuffer-depth', 'r32float');
    this.motion = make('gbuffer-motion', 'rg32float');
  }
}
