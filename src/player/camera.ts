import { config } from '../config';
import type { Input } from '../core/input';
import {
  forwardFromYawPitch,
  halton,
  invert,
  jitterProjection,
  multiply,
  perspective,
  viewRotation,
  type Mat4,
  type Vec3,
} from '../core/math';

const DEG = Math.PI / 180;

/** Everything a frame needs from the camera. Matrices are rotation-only (camera-relative). */
export interface CameraFrame {
  /** Camera position split for float32 precision: integer block + fraction in [0, 1). */
  cell: Vec3;
  frac: Vec3;
  position: Vec3;
  forward: Vec3;
  /** Jittered inverse view-projection: ray generation. */
  invViewProj: Mat4;
  /** Unjittered view-projection of this and the previous frame: motion vectors. */
  viewProj: Mat4;
  prevViewProj: Mat4;
  /** position − previous frame's position (world units). */
  prevDelta: Vec3;
  /** Current sub-pixel jitter in pixels, each in [−0.5, 0.5). */
  jitter: [number, number];
  /** Previous frame's forward axis. */
  prevForward: Vec3;
  /** Frame counter (noise seeds). */
  frameIndex: number;
}

/**
 * Free-flying debug camera. Movement and look run in the fixed simulation step and
 * are interpolated for rendering. WASD move, Space/ShiftLeft up/down, ControlLeft boost,
 * mouse look while the pointer is locked (click the canvas).
 */
export class FlyCamera {
  position: Vec3;
  yaw = 0;
  pitch: number;
  private prevPosition: Vec3;
  private prevYaw = 0;
  private prevPitch: number;
  private frameIndex = 0;
  private lastFrame: { viewProj: Mat4; position: Vec3; forward: Vec3 } | null = null;
  private frames = 0;

  constructor(position: Vec3, pitchDeg = -20, yawDeg = 0) {
    this.position = [...position];
    this.prevPosition = [...position];
    this.pitch = pitchDeg * DEG;
    this.prevPitch = this.pitch;
    this.yaw = yawDeg * DEG;
    this.prevYaw = this.yaw;
  }

  /** Fixed-step update. */
  update(dt: number, input: Input): void {
    this.prevPosition = [...this.position];
    this.prevYaw = this.yaw;
    this.prevPitch = this.pitch;

    // Scripted motion (debug): constant turn and forward flight.
    this.yaw += config.debug.cameraSpin * DEG * dt;
    if (config.debug.cameraFly) {
      const f = forwardFromYawPitch(this.yaw, this.pitch);
      for (let a = 0; a < 3; a++) this.position[a]! += f[a]! * config.debug.cameraFly * dt;
    }

    const { x: mx, y: my } = input.mouseDelta;
    const sens = config.input.mouseSensitivity;
    const maxPitch = config.camera.maxPitch * DEG;
    this.yaw += mx * sens;
    this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch - my * sens));

    const f = forwardFromYawPitch(this.yaw, 0);
    const right: Vec3 = [-f[2], 0, f[0]];
    let dx = 0;
    let dy = 0;
    let dz = 0;
    const add = (v: Vec3, s: number) => {
      dx += v[0] * s;
      dy += v[1] * s;
      dz += v[2] * s;
    };
    if (input.isDown('KeyW')) add(forwardFromYawPitch(this.yaw, this.pitch), 1);
    if (input.isDown('KeyS')) add(forwardFromYawPitch(this.yaw, this.pitch), -1);
    if (input.isDown('KeyD')) add(right, 1);
    if (input.isDown('KeyA')) add(right, -1);
    if (input.isDown('Space')) dy += 1;
    if (input.isDown('ShiftLeft')) dy -= 1;
    const len = Math.hypot(dx, dy, dz);
    if (len > 0) {
      const boost = input.isDown('ControlLeft') ? config.camera.boostMultiplier : 1;
      const s = (config.camera.speed * boost * dt) / len;
      this.position[0] += dx * s;
      this.position[1] += dy * s;
      this.position[2] += dz * s;
    }
  }

  /** Builds this frame's matrices; `alpha` interpolates between the last two sim steps. */
  frame(alpha: number, width: number, height: number): CameraFrame {
    const lerp = (a: number, b: number) => a + (b - a) * alpha;
    const position: Vec3 = [
      lerp(this.prevPosition[0], this.position[0]),
      lerp(this.prevPosition[1], this.position[1]),
      lerp(this.prevPosition[2], this.position[2]),
    ];
    const yaw = lerp(this.prevYaw, this.yaw);
    const pitch = lerp(this.prevPitch, this.pitch);

    const { fovY, near, far } = config.camera;
    const proj = perspective(fovY * DEG, width / height, near, far);
    const view = viewRotation(yaw, pitch);
    const viewProj = multiply(proj, view);

    const jitter = this.nextJitter();
    const jittered = multiply(jitterProjection(proj, jitter[0], jitter[1], width, height), view);
    const invViewProj = invert(jittered);

    const forward = forwardFromYawPitch(yaw, pitch);
    const prev = this.lastFrame ?? { viewProj, position, forward };
    const prevDelta: Vec3 = [position[0] - prev.position[0], position[1] - prev.position[1], position[2] - prev.position[2]];
    this.lastFrame = { viewProj, position, forward };

    const cell: Vec3 = [Math.floor(position[0]), Math.floor(position[1]), Math.floor(position[2])];
    return {
      cell,
      frac: [position[0] - cell[0], position[1] - cell[1], position[2] - cell[2]],
      position,
      forward,
      invViewProj,
      viewProj,
      prevViewProj: prev.viewProj,
      prevDelta,
      jitter,
      prevForward: prev.forward,
      frameIndex: this.frames++,
    };
  }

  private nextJitter(): [number, number] {
    if (!config.camera.jitter) return [0, 0];
    // Index from 1: halton(0) = 0 for every base.
    const i = (this.frameIndex++ % config.camera.jitterSequenceLength) + 1;
    return [halton(i, 2) - 0.5, halton(i, 3) - 0.5];
  }
}
