import { config } from '../config';

/**
 * Tracks the canvas' size in physical pixels. The backing store is only resized
 * when `apply()` is called at the start of a frame, so it never changes mid-frame.
 */
export class CanvasSize {
  width = 1;
  height = 1;
  private pendingW = 1;
  private pendingH = 1;
  private readonly observer: ResizeObserver;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly maxDimension: number,
  ) {
    this.observer = new ResizeObserver((entries) => {
      for (const entry of entries) this.measure(entry);
    });
    try {
      this.observer.observe(canvas, { box: 'device-pixel-content-box' });
    } catch {
      // Safari: device-pixel-content-box is unsupported.
      this.observer.observe(canvas, { box: 'content-box' });
    }
    this.remeasure();
  }

  /** Re-reads the CSS size, e.g. after `maxPixelRatio` changes (no ResizeObserver event fires). */
  remeasure(): void {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = this.pixelRatio();
    this.setPending(rect.width * ratio, rect.height * ratio);
  }

  /** Applies a pending resize. Returns true if the backing size changed. */
  apply(): boolean {
    if (this.pendingW === this.width && this.pendingH === this.height) return false;
    this.width = this.pendingW;
    this.height = this.pendingH;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    return true;
  }

  dispose(): void {
    this.observer.disconnect();
  }

  private measure(entry: ResizeObserverEntry): void {
    const box = entry.contentBoxSize[0];
    if (!box) return;
    const dp = entry.devicePixelContentBoxSize?.[0];
    const ratio = this.pixelRatio();
    const dpr = window.devicePixelRatio;
    // Exact device pixels are preferred (no blur from rounding), but only when uncapped
    // and consistent with css * dpr — DevTools DPR emulation reports CSS pixels here.
    const dpConsistent =
      dp !== undefined &&
      Math.abs(dp.inlineSize - box.inlineSize * dpr) <= 1 &&
      Math.abs(dp.blockSize - box.blockSize * dpr) <= 1;
    if (dp && dpConsistent && ratio === dpr) {
      this.setPending(dp.inlineSize, dp.blockSize);
    } else {
      this.setPending(box.inlineSize * ratio, box.blockSize * ratio);
    }
  }

  private pixelRatio(): number {
    return Math.min(window.devicePixelRatio || 1, config.render.maxPixelRatio);
  }

  private setPending(w: number, h: number): void {
    this.pendingW = clampDim(w, this.maxDimension);
    this.pendingH = clampDim(h, this.maxDimension);
  }
}

function clampDim(v: number, max: number): number {
  return Math.max(1, Math.min(Math.round(v), max));
}
