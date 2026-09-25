import { config } from '../config';

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

/** Drives `FixedStepClock` from requestAnimationFrame. */
export class GameLoop {
  readonly clock = new FixedStepClock();
  private rafId = 0;
  private lastTime = -1;

  constructor(private readonly callbacks: LoopCallbacks) {}

  start(): void {
    if (this.rafId) return;
    this.lastTime = -1;
    this.rafId = requestAnimationFrame(this.frame);
  }

  stop(): void {
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
  }

  private readonly frame = (now: number): void => {
    this.rafId = requestAnimationFrame(this.frame);
    const frameDt = this.lastTime < 0 ? 0 : (now - this.lastTime) / 1000;
    this.lastTime = now;

    const { steps, alpha, step } = this.clock.advance(frameDt);
    for (let i = 0; i < steps; i++) this.callbacks.update(step);
    this.callbacks.render(alpha, this.clock.simTime + alpha * step, frameDt);
  };
}
