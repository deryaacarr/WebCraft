/**
 * Keyboard and mouse state. Keys are identified by `KeyboardEvent.code`
 * (layout-independent: 'KeyW' is the same physical key on QWERTY and AZERTY).
 *
 * "Pressed"/"released" edges and mouse deltas accumulate until `endTick()`,
 * so they are seen by exactly one fixed simulation step.
 */
export class Input {
  private readonly down = new Set<string>();
  private readonly pressed = new Set<string>();
  private readonly released = new Set<string>();
  private readonly buttons = new Set<number>();
  private readonly pressedButtons = new Set<number>();
  private dx = 0;
  private dy = 0;
  private wheel = 0;
  private readonly abort = new AbortController();

  constructor(private readonly target: HTMLElement) {
    const opts = { signal: this.abort.signal };
    window.addEventListener('keydown', this.onKeyDown, opts);
    window.addEventListener('keyup', this.onKeyUp, opts);
    window.addEventListener('blur', this.reset, opts);
    target.addEventListener('mousedown', this.onMouseDown, opts);
    window.addEventListener('mouseup', this.onMouseUp, opts);
    window.addEventListener('mousemove', this.onMouseMove, opts);
    target.addEventListener('wheel', this.onWheel, { ...opts, passive: true });
    target.addEventListener('contextmenu', (e) => e.preventDefault(), opts);
    document.addEventListener('pointerlockchange', this.onLockChange, opts);
  }

  get pointerLocked(): boolean {
    return document.pointerLockElement === this.target;
  }

  isDown(code: string): boolean {
    return this.down.has(code);
  }

  wasPressed(code: string): boolean {
    return this.pressed.has(code);
  }

  wasReleased(code: string): boolean {
    return this.released.has(code);
  }

  isButtonDown(button: number): boolean {
    return this.buttons.has(button);
  }

  wasButtonPressed(button: number): boolean {
    return this.pressedButtons.has(button);
  }

  /** Mouse movement in pixels since the last `endTick()`; only while pointer is locked. */
  get mouseDelta(): { x: number; y: number } {
    return { x: this.dx, y: this.dy };
  }

  get wheelDelta(): number {
    return this.wheel;
  }

  requestPointerLock(): void {
    if (!this.pointerLocked) {
      // Chromium returns a promise that rejects if the user re-locks too quickly.
      Promise.resolve(this.target.requestPointerLock()).catch(() => {});
    }
  }

  exitPointerLock(): void {
    if (this.pointerLocked) document.exitPointerLock();
  }

  /** Clears per-tick edges and deltas. Call after each simulation step. */
  endTick(): void {
    this.pressed.clear();
    this.released.clear();
    this.pressedButtons.clear();
    this.dx = 0;
    this.dy = 0;
    this.wheel = 0;
  }

  dispose(): void {
    this.abort.abort();
  }

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    // Keep browser shortcuts (F5, F12, Ctrl+R…) working unless we own the pointer.
    if (this.pointerLocked && !e.ctrlKey && !e.metaKey) e.preventDefault();
    if (e.repeat) return;
    this.down.add(e.code);
    this.pressed.add(e.code);
  };

  private readonly onKeyUp = (e: KeyboardEvent): void => {
    this.down.delete(e.code);
    this.released.add(e.code);
  };

  private readonly onMouseDown = (e: MouseEvent): void => {
    if (!this.pointerLocked) {
      // The first click only captures the pointer; it is not a game action.
      this.requestPointerLock();
      return;
    }
    this.buttons.add(e.button);
    this.pressedButtons.add(e.button);
  };

  private readonly onMouseUp = (e: MouseEvent): void => {
    this.buttons.delete(e.button);
  };

  private readonly onMouseMove = (e: MouseEvent): void => {
    if (!this.pointerLocked) return;
    this.dx += e.movementX;
    this.dy += e.movementY;
  };

  private readonly onWheel = (e: WheelEvent): void => {
    this.wheel += Math.sign(e.deltaY);
  };

  private readonly onLockChange = (): void => {
    if (!this.pointerLocked) this.buttons.clear();
  };

  /** Drops all held state, e.g. when the window loses focus mid-keypress. */
  private readonly reset = (): void => {
    this.down.clear();
    this.buttons.clear();
    this.endTick();
  };
}
