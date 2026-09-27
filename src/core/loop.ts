import { config } from '../config';
import { onPowerSavingResume, powerSavingSuspended } from './power';

export interface LoopCallbacks {
  /** Advances the simulation by exactly `dt` seconds. */
  update(dt: number): void;
  /**
   * Draws a frame. `alpha` ∈ [0, 1) is how far the render moment lies between the
   * previous and the current simulation state; interpolate with it.
   * `time` is the simulation time at the render moment (simTime + alpha * step).
   */
  render(alpha: number, time: number, frameDt: number): void;
}

/**
 * Fixed-timestep accumulator ("Fix Your Timestep!"). Pure logic, no timers,
 * so it can be driven from tests as well as from requestAnimationFrame.
 */
export class FixedStepClock {
  /** Total simulated time in seconds (advances in whole steps). */
  simTime = 0;
  private accumulator = 0;

  constructor(
    private readonly getStep: () => number = () => 1 / config.sim.tickRate,
    private readonly getMaxFrame: () => number = () => config.sim.maxFrameTime,
  ) {}

  /** Consumes `frameDt` and returns how many fixed steps to run plus the blend factor. */
  advance(frameDt: number): { steps: number; alpha: number; step: number } {
    const step = this.getStep();
    this.accumulator += Math.min(Math.max(frameDt, 0), this.getMaxFrame());
    let steps = 0;
    while (this.accumulator >= step) {
      this.accumulator -= step;
      this.simTime += step;
      steps++;
    }
    return { steps, alpha: this.accumulator / step, step };
  }
}

/**
 * Frame pacing for energy saving: a frame rate cap, a lower rate once idle. Pure logic
 * (times in ms), driven by requestAnimationFrame callbacks and by tests.
 */
export class FramePacer {
  private next = -Infinity;
  private lastActive = -Infinity;

  constructor(
    private readonly getConfig: () => { enabled: boolean; maxFps: number; idleFps: number; idleAfterSeconds: number } = () => config.power,
    private readonly suspended: () => boolean = powerSavingSuspended,
  ) {}

  /** Something changed (input, camera motion, world): full rate, starting with the next
   *  display refresh (no waiting out an idle interval). */
  markActive(now: number): void {
    if (this.isIdle(now)) this.next = -Infinity;
    this.lastActive = now;
  }

  private isIdle(now: number): boolean {
    return now - this.lastActive > this.getConfig().idleAfterSeconds * 1000;
  }

  /** Target frame rate at `now`: 0 = uncapped, −1 = stopped (idle with idleFps 0). */
  targetFps(now: number): number {
    const c = this.getConfig();
    if (!c.enabled || this.suspended()) return 0;
    if (!this.isIdle(now)) return c.maxFps;
    if (c.idleFps <= 0) return -1;
    return c.maxFps > 0 ? Math.min(c.idleFps, c.maxFps) : c.idleFps;
  }

  /** Whether to run a frame for the display refresh at `now`. */
  shouldRun(now: number): boolean {
    const fps = this.targetFps(now);
    if (fps < 0) return false;
    if (fps === 0) return true;
    const interval = 1000 / fps;
    // Half a display frame of slack, so a 60 Hz display runs a 30 fps cap every 2nd frame.
    if (now < this.next - 4) return false;
    // Step the schedule without drift; after a gap (or at start) restart it from now.
    this.next = (this.next < now - interval ? now : this.next) + interval;
    return true;
  }

  reset(): void {
    this.next = -Infinity;
  }
}

/** Drives `FixedStepClock` from requestAnimationFrame. */
export class GameLoop {
  readonly clock = new FixedStepClock();
  readonly pacer = new FramePacer();
  private rafId = 0;
  private lastTime = -1;

  constructor(private readonly callbacks: LoopCallbacks) {
    // Hidden tab or unfocused window: stop entirely; back: restart without a huge first
    // frame delta. Re-evaluated on every change and when a benchmark ends.
    const update = () => this.updatePaused();
    document.addEventListener('visibilitychange', update);
    window.addEventListener('blur', update);
    window.addEventListener('focus', update);
    onPowerSavingResume(update);
  }

  /** Whether energy saving wants the loop stopped right now. */
  private shouldPause(): boolean {
    const p = config.power;
    if (!p.enabled || powerSavingSuspended()) return false;
    return (p.pauseHidden && document.hidden) || (p.pauseUnfocused && !document.hasFocus());
  }

  private updatePaused(): void {
    if (!this.running) return;
    if (this.shouldPause()) this.pause();
    else this.resume();
  }

  private running = false;

  start(): void {
    this.running = true;
    this.updatePaused();
  }

  stop(): void {
    this.running = false;
    this.pause();
  }

  /** Marks activity (see FramePacer). */
  markActive(): void {
    this.pacer.markActive(performance.now());
  }

  private resume(): void {
    if (this.rafId) return;
    this.lastTime = -1;
    this.pacer.reset();
    this.markActive();
    this.rafId = requestAnimationFrame(this.frame);
  }

  private pause(): void {
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
  }

  private readonly frame = (now: number): void => {
    this.rafId = requestAnimationFrame(this.frame);
    if (!this.pacer.shouldRun(now)) return;
    const frameDt = this.lastTime < 0 ? 0 : (now - this.lastTime) / 1000;
    this.lastTime = now;

    const { steps, alpha, step } = this.clock.advance(frameDt);
    for (let i = 0; i < steps; i++) this.callbacks.update(step);
    this.callbacks.render(alpha, this.clock.simTime + alpha * step, frameDt);
  };
}
