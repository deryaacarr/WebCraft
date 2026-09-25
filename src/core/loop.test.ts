import { describe, expect, it } from 'vitest';
import { FixedStepClock } from './loop';

const STEP = 1 / 60;
const MAX_FRAME = 0.25;
const makeClock = () => new FixedStepClock(() => STEP, () => MAX_FRAME);

describe('FixedStepClock', () => {
  it('runs no step and accumulates alpha for a sub-step frame', () => {
    const clock = makeClock();
    const r = clock.advance(STEP / 2);
    expect(r.steps).toBe(0);
    expect(r.alpha).toBeCloseTo(0.5);
    expect(clock.simTime).toBe(0);
  });

  it('carries the remainder across frames', () => {
    const clock = makeClock();
    clock.advance(STEP * 0.75);
    const r = clock.advance(STEP * 0.75);
    expect(r.steps).toBe(1);
    expect(r.alpha).toBeCloseTo(0.5);
    expect(clock.simTime).toBeCloseTo(STEP);
  });

  it('runs several steps on a long frame', () => {
    const r = makeClock().advance(STEP * 3.25);
    expect(r.steps).toBe(3);
    expect(r.alpha).toBeCloseTo(0.25);
  });

  it('clamps huge frames to avoid the spiral of death', () => {
    const clock = makeClock();
    const r = clock.advance(10);
    expect(r.steps).toBe(Math.floor(MAX_FRAME / STEP));
    expect(clock.simTime).toBeLessThanOrEqual(MAX_FRAME);
  });

  it('ignores negative deltas', () => {
    const r = makeClock().advance(-1);
    expect(r.steps).toBe(0);
    expect(r.alpha).toBe(0);
  });

  it('keeps alpha in [0, 1) over many irregular frames', () => {
    const clock = makeClock();
    let total = 0;
    for (let i = 0; i < 1000; i++) {
      const dt = 0.001 + ((i * 7919) % 37) / 1000;
      total += dt;
      const { alpha } = clock.advance(dt);
      expect(alpha).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeLessThan(1);
    }
    // No frame exceeds MAX_FRAME, so no time is dropped.
    expect(clock.simTime).toBeLessThanOrEqual(total);
    expect(total - clock.simTime).toBeLessThan(STEP);
  });
});
