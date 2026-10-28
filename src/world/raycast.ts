import type { Vec3 } from '../core/math';
import { BlockId } from './blocks';
import { isShaped, MICRO, microSolid } from './shapes';
import type { World } from './world';

export interface RayHit {
  /** Block coordinates of the hit voxel. */
  cell: Vec3;
  /** Face normal (axis unit vector pointing out of the hit face); zeros if the ray starts inside. */
  normal: Vec3;
  /** Distance along the (normalised) direction to the hit face. */
  t: number;
  id: number;
}

/**
 * Voxel DDA (Amanatides & Woo) against the CPU world, one voxel at a time. Reference for
 * the GPU tracer and the basis for block picking. Any non-air block counts as a hit;
 * shaped blocks (shapes.ts) only where the ray meets one of their micro voxels.
 * Axis ties resolve in the same order as trace.wgsl (x, then y, then z).
 */
export function raycast(world: World, origin: Vec3, dir: Vec3, maxT: number): RayHit | null {
  const len = Math.hypot(dir[0], dir[1], dir[2]);
  if (len === 0) return null;
  const d: Vec3 = [dir[0] / len, dir[1] / len, dir[2] / len];
  const cell: Vec3 = [Math.floor(origin[0]), Math.floor(origin[1]), Math.floor(origin[2])];
  const step: Vec3 = [d[0] >= 0 ? 1 : -1, d[1] >= 0 ? 1 : -1, d[2] >= 0 ? 1 : -1];
  // Boundary distances are recomputed from the cell (not accumulated), like trace.wgsl.
  const boundaryT = (a: number) =>
    d[a] === 0 ? Infinity : (cell[a]! + (step[a]! > 0 ? 1 : 0) - origin[a]!) / d[a]!;
  const tMax: Vec3 = [boundaryT(0), boundaryT(1), boundaryT(2)];

  let t = 0;
  const normal: Vec3 = [0, 0, 0];
  for (;;) {
    const id = world.getBlock(cell[0], cell[1], cell[2]);
    if (id !== BlockId.air) {
      if (!isShaped(id)) return { cell: [...cell], normal: [...normal], t, id };
      const hit = traceMicro(id, cell, origin, d, step, t, normal);
      if (hit && hit.t <= maxT) return hit;
    }

    const a = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : tMax[1] < tMax[2] ? 1 : 2;
    t = tMax[a]!;
    if (t > maxT) return null;
    cell[a]! += step[a]!;
    tMax[a] = boundaryT(a);
    normal[0] = normal[1] = normal[2] = 0;
    normal[a] = -step[a]!;
  }
}

/** DDA through the micro grid of the shaped block at `cell`, entered at `t` through `normal`. */
function traceMicro(id: number, cell: Vec3, origin: Vec3, d: Vec3, step: Vec3, t: number, normal: Vec3): RayHit | null {
  const n: Vec3 = [...normal];
  const m: Vec3 = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    m[a] = Math.min(Math.max(Math.floor((origin[a]! + d[a]! * t - cell[a]!) * MICRO), 0), MICRO - 1);
  }
  const boundaryT = (a: number) =>
    d[a] === 0 ? Infinity : (cell[a]! + (m[a]! + (step[a]! > 0 ? 1 : 0)) / MICRO - origin[a]!) / d[a]!;
  for (;;) {
    if (microSolid(id, cell, m[0], m[1], m[2])) return { cell: [...cell], normal: n, t, id };
    const tb: Vec3 = [boundaryT(0), boundaryT(1), boundaryT(2)];
    const a = tb[0] < tb[1] ? (tb[0] < tb[2] ? 0 : 2) : tb[1] < tb[2] ? 1 : 2;
    t = tb[a]!;
    m[a]! += step[a]!;
    if (m[a]! < 0 || m[a]! >= MICRO) return null;
    n[0] = n[1] = n[2] = 0;
    n[a] = -step[a]!;
  }
}
