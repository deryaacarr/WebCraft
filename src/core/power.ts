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
