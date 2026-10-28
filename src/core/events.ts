/**
 * Game events for loosely coupled systems (sound, UI, later gameplay). Typed wrapper over
 * an EventTarget: `on` returns an unsubscribe function.
 */
export interface GameEventMap {
  /** The camera went under water or came out (water surface with its waves). */
  underwater: { underwater: boolean };
}

const target = new EventTarget();

export function emit<K extends keyof GameEventMap>(type: K, detail: GameEventMap[K]): void {
  target.dispatchEvent(new CustomEvent(type, { detail }));
}

export function on<K extends keyof GameEventMap>(type: K, listener: (detail: GameEventMap[K]) => void): () => void {
  const handler = (e: Event) => listener((e as CustomEvent<GameEventMap[K]>).detail);
  target.addEventListener(type, handler);
  return () => target.removeEventListener(type, handler);
}
