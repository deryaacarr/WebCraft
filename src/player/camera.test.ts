import { afterEach, describe, expect, it } from 'vitest';
import { config } from '../config';
import type { Input } from '../core/input';
import { FlyCamera } from './camera';

function fakeInput(keys: string[] = [], mouse = { x: 0, y: 0 }): Input {
  return { isDown: (k: string) => keys.includes(k), mouseDelta: mouse } as unknown as Input;
}

const saved = { ...config.camera };
afterEach(() => Object.assign(config.camera, saved));

describe('FlyCamera', () => {
  it('moves along the view direction with W and strafes with D', () => {
    const cam = new FlyCamera([0, 0, 0], 0);
    cam.update(1, fakeInput(['KeyW']));
    expect(cam.position[2]).toBeCloseTo(-config.camera.speed); // yaw 0 looks down −Z
    cam.update(1, fakeInput(['KeyD']));
    expect(cam.position[0]).toBeCloseTo(config.camera.speed);
  });

  it('keeps diagonal speed constant and boosts with ControlLeft', () => {
    const cam = new FlyCamera([0, 0, 0], 0);
    cam.update(1, fakeInput(['KeyW', 'KeyD']));
    expect(Math.hypot(...cam.position)).toBeCloseTo(config.camera.speed);
    const fast = new FlyCamera([0, 0, 0], 0);
    fast.update(1, fakeInput(['Space', 'ControlLeft']));
    expect(fast.position[1]).toBeCloseTo(config.camera.speed * config.camera.boostMultiplier);
  });

  it('clamps pitch', () => {
    const cam = new FlyCamera([0, 0, 0], 0);
    cam.update(1 / 60, fakeInput([], { x: 0, y: -1e6 }));
    expect(cam.pitch).toBeCloseTo((config.camera.maxPitch * Math.PI) / 180);
  });

  it('splits the position into cell + fraction and keeps the previous view-projection', () => {
    const cam = new FlyCamera([-1.25, 10.5, 3.75], 0);
    const first = cam.frame(1, 64, 32);
    expect(first.cell).toEqual([-2, 10, 3]);
    first.frac.forEach((f) => expect(f).toBeGreaterThanOrEqual(0));
    expect(first.frac[0]).toBeCloseTo(0.75);
    expect(first.prevViewProj).toEqual(first.viewProj); // no history yet
    cam.update(1 / 60, fakeInput([], { x: 100, y: 0 }));
    const second = cam.frame(1, 64, 32);
    expect(second.prevViewProj).toEqual(first.viewProj);
    expect(second.viewProj).not.toEqual(first.viewProj);
  });

  it('interpolates between simulation steps', () => {
    const cam = new FlyCamera([0, 0, 0], 0);
    cam.update(1, fakeInput(['KeyW']));
    expect(cam.frame(0.5, 16, 16).position[2]).toBeCloseTo(-config.camera.speed / 2);
  });

  it('jitters with Halton(2,3) only when enabled', () => {
    const cam = new FlyCamera([0, 0, 0], 0);
    expect(cam.frame(1, 8, 8).jitter).toEqual([0, 0]);
    config.camera.jitter = true;
    expect(cam.frame(1, 8, 8).jitter).toEqual([0, 1 / 3 - 0.5]);
    const j2 = cam.frame(1, 8, 8).jitter;
    expect(j2[0]).toBeCloseTo(-0.25);
    expect(j2[1]).toBeCloseTo(2 / 3 - 0.5);
  });
});
