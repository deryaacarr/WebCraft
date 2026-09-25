import GUI from 'lil-gui';
import Stats from 'stats-gl';
import { config } from '../config';
import type { GpuProfiler } from '../gpu/profiler';

export interface DebugOverlayHooks {
  /** Called when a setting that affects render target sizes changes. */
  onResolutionChange(): void;
}

/**
 * F3 overlay: stats-gl (FPS / CPU / total GPU) and a lil-gui panel that edits
 * `config` live, plus a read-only per-pass GPU timing folder.
 */
export class DebugOverlay {
  readonly gui: GUI;
  private readonly stats: Stats;
  private readonly gpuPanel: InstanceType<typeof Stats.Panel> | null;
  private gpuPanelMax = 0;
  private lastGpuPanelUpdate = 0;
  private readonly timingFolder: GUI;
  private readonly timingValues: Record<string, string> = {};
  private visible = config.debug.visibleOnStart;

  constructor(
    private readonly profiler: GpuProfiler,
    hooks: DebugOverlayHooks,
  ) {
    // GPU time comes from our multi-pass profiler, not stats-gl's single-pass tracker.
    this.stats = new Stats({ trackGPU: false, horizontal: true });
    this.gpuPanel = profiler.enabled
      ? this.stats.addPanel(new Stats.Panel('GPU', '#ff0', '#220'))
      : null;
    document.body.appendChild(this.stats.dom);

    this.gui = new GUI({ title: 'WebCraft debug' });
    this.buildConfigControls(hooks);
    this.timingFolder = this.gui.addFolder('GPU timings (ms)');
    if (!profiler.enabled) {
      this.timingFolder.add({ status: 'timestamp-query yok' }, 'status').name('durum').disable();
    }

    window.addEventListener('keydown', (e) => {
      if (e.code !== config.debug.toggleKey) return;
      e.preventDefault(); // F3 is "find" in most browsers.
      if (!e.repeat) this.setVisible(!this.visible);
    });
    this.setVisible(this.visible);
  }

  beginFrame(): void {
    this.stats.begin();
  }

  endFrame(): void {
    this.stats.end();
    this.stats.update();
    if (!this.visible) return;
    this.updateGpuPanel();
    this.updateTimings();
  }

  private setVisible(visible: boolean): void {
    this.visible = visible;
    this.stats.dom.style.display = visible ? '' : 'none';
    this.gui.show(visible);
  }

  private buildConfigControls(hooks: DebugOverlayHooks): void {
    const sim = this.gui.addFolder('Simulation');
    sim.add(config.sim, 'tickRate', 10, 240, 1).name('tick rate (Hz)');
    sim.add(config.sim, 'maxFrameTime', 0.05, 1, 0.01).name('max frame (s)');

    const render = this.gui.addFolder('Render');
    render
      .add(config.render, 'renderScale', 0.25, 2, 0.05)
      .name('render scale')
      .onFinishChange(() => hooks.onResolutionChange());
    render
      .add(config.render, 'maxPixelRatio', 0.5, 4, 0.25)
      .name('max pixel ratio')
      .onFinishChange(() => hooks.onResolutionChange());

    const gradient = this.gui.addFolder('Test gradient');
    gradient.add(config.gradient, 'speed', 0, 2, 0.01);
    gradient.add(config.gradient, 'waveStrength', 0, 1, 0.01).name('wave strength');

    const input = this.gui.addFolder('Input');
    input.add(config.input, 'mouseSensitivity', 0.0001, 0.01, 0.0001).name('mouse sensitivity');

    const debug = this.gui.addFolder('Debug');
    debug.add(config.debug, 'profilerSmoothing', 1, 240, 1).name('GPU smoothing (frames)');
    debug.close();
  }

  // Drives the panel directly: stats-gl's public updatePanel() passes arguments to
  // Panel.update() in the wrong order (v4.2), which garbles the label.
  private updateGpuPanel(): void {
    const now = performance.now();
    if (!this.gpuPanel || now - this.lastGpuPanelUpdate < config.debug.gpuPanelIntervalMs) return;
    this.lastGpuPanelUpdate = now;
    const ms = this.profiler.totalMs;
    this.gpuPanelMax = Math.max(this.gpuPanelMax * config.debug.gpuPanelMaxDecay, ms);
    this.gpuPanel.update(ms, this.gpuPanelMax, 2);
    this.gpuPanel.updateGraph(ms, this.gpuPanelMax);
  }

  private updateTimings(): void {
    for (const [name, ms] of this.profiler.timings) {
      const text = ms.toFixed(3);
      if (!(name in this.timingValues)) {
        this.timingValues[name] = text;
        this.timingFolder.add(this.timingValues, name).disable().listen();
      }
      this.timingValues[name] = text;
    }
  }
}
