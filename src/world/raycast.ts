import type { Vec3 } from '../core/math';
import { BlockId } from './blocks';
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
 * the GPU tracer and the basis for block picking. Any non-air block counts as a hit.
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
    if (id !== BlockId.air) return { cell: [...cell], normal: [...normal], t, id };

    const a = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : tMax[1] < tMax[2] ? 1 : 2;
    t = tMax[a]!;
    if (t > maxT) return null;
    cell[a]! += step[a]!;
    tMax[a] = boundaryT(a);
    normal[0] = normal[1] = normal[2] = 0;
    normal[a] = -step[a]!;
  }
}
