/**
 * Energy-saving suspension: while any `withoutPowerSaving` call is in flight (benchmarks,
 * verification tools), frame pacing and pausing are off, so measurements see a normal
 * frame loop. Nesting is counted.
 */
let suspended = 0;
const listeners = new Set<() => void>();

export function powerSavingSuspended(): boolean {
  return suspended > 0;
}

/** Called when the suspension ends (the loop re-evaluates pausing). */
export function onPowerSavingResume(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export async function withoutPowerSaving<T>(fn: () => Promise<T>): Promise<T> {
  suspended++;
  try {
    return await fn();
  } finally {
    suspended--;
    if (suspended === 0) for (const l of listeners) l();
  }
}

/**
 * Animated content on screen (water waves now; rain, plants in the wind later): while any
 * source reports some, the idle frame rate is `config.power.liveIdleFps` instead of
 * `idleFps`, so the animation keeps running smoothly with the camera still. Sources report
 * by name and clear their own flag.
 */
const live = new Set<string>();

export function setLiveContent(source: string, on: boolean): void {
  if (on) live.add(source);
  else live.delete(source);
}

export function hasLiveContent(): boolean {
  return live.size > 0;
}
