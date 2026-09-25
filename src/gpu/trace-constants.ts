import { config } from '../config';

/** Pipeline-override constants every pipeline including trace.wgsl must set. */
export function traceConstants(): Record<string, number> {
  return { BRICK_BITS: config.world.brickBits };
}
