import { config } from '../config';
import { shapedIdMask } from '../world/shapes';

/** Pipeline-override constants every pipeline including trace.wgsl must set. */
export function traceConstants(): Record<string, number> {
  // Sub-voxel shapes off at startup (&micro=0) compiles the micro walk out entirely; the
  // panel toggle later only clears the shape table (shaped blocks become cubes).
  return { BRICK_BITS: config.world.brickBits, SHAPED_IDS: config.detail.microVoxels ? shapedIdMask() : 0 };
}
