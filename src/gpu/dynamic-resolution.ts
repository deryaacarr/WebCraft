/**
 * Dynamic resolution: picks the internal render scale from the measured GPU frame time.
 * Pure logic (times in ms and seconds), driven by the renderer and by tests.
 *
 *   down  when the frame time over a short window (its upper percentile, so single-frame
 *         spikes — a chunk upload, a first shader compile — do not count) exceeds the
 *         target: at once, by the step the cost model (∝ scale²) predicts
 *   up    only after staying well under the target for a while, one quantum at a time
 *   hold  after every change, until the window has refilled with frames at the new scale
 *         (the measurements arrive a few frames late) — no oscillation
 *
 * Scales are quantised so the active region changes rarely and in clean steps.
 */
export interface DynamicResolutionConfig {
  enabled: boolean;
  targetMs: number;
  minScale: number;
  maxScale: number;
}

/** Quantum of scale changes. */
export const SCALE_STEP = 0.025;
/** Frames in the decision window, and the percentile looked at (0.8 of 10: the third
 *  highest — two spikes in the window are ignored). */
const WINDOW = 10;
const PERCENTILE = 0.8;
/** Go down above target × DOWN_MARGIN; go up only below target × UP_MARGIN, sustained for
 *  UP_DELAY seconds (one step costs ~+12 % at 0.4: from below 0.85 it stays under 1). */
const DOWN_MARGIN = 1.05;
const UP_MARGIN = 0.85;
const UP_DELAY = 1;
/** Measurements ignored after a change: they still describe frames at the old scale (the
 *  timestamps arrive a few frames late; counting them caused a second drop, then a rise). */
const SETTLE_FRAMES = 6;

export class DynamicResolution {
  private samples: number[] = [];
  private underSince = -1;
  private settle = 0;
  scale: number;

  constructor(initial: number) {
    this.scale = initial;
  }

  /** Quantised and clamped to the configured range. */
  private clampScale(s: number, c: DynamicResolutionConfig): number {
    const q = Math.round(s / SCALE_STEP) * SCALE_STEP;
    return Math.min(Math.max(q, c.minScale), c.maxScale);
  }

  private change(to: number): void {
    this.scale = to;
    this.samples = [];
    this.underSince = -1;
    this.settle = SETTLE_FRAMES;
  }

  /** Feeds one frame's GPU time (ms) measured at `now` (seconds); returns the scale. */
  update(gpuMs: number, now: number, c: DynamicResolutionConfig): number {
    const clamped = this.clampScale(this.scale, c);
    if (clamped !== this.scale) this.change(clamped);
    if (!c.enabled || !(gpuMs > 0)) return this.scale;
    if (this.settle > 0) {
      this.settle--;
      return this.scale;
    }
    this.samples.push(gpuMs);
    if (this.samples.length > WINDOW) this.samples.shift();
    if (this.samples.length < WINDOW) return this.scale;
    const sorted = [...this.samples].sort((a, b) => a - b);
    const ms = sorted[Math.floor(PERCENTILE * (WINDOW - 1))]!;
    if (ms > c.targetMs * DOWN_MARGIN) {
      // Cost ∝ scale²: the scale that would meet the target, at least one step down.
      const ideal = this.scale * Math.sqrt(c.targetMs / ms);
      const next = Math.min(this.clampScale(Math.floor(ideal / SCALE_STEP) * SCALE_STEP, c), this.scale - SCALE_STEP);
      if (next >= c.minScale - 1e-9 && next < this.scale) this.change(Math.max(next, c.minScale));
      return this.scale;
    }
    if (ms < c.targetMs * UP_MARGIN && this.scale < c.maxScale) {
      if (this.underSince < 0) this.underSince = now;
      if (now - this.underSince >= UP_DELAY) this.change(this.clampScale(this.scale + SCALE_STEP, c));
      return this.scale;
    }
    this.underSince = -1;
    return this.scale;
  }
}
