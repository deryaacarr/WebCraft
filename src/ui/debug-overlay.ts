import GUI from 'lil-gui';
import Stats from 'stats-gl';
import { config, DEBUG_VIEWS } from '../config';
import type { Vec3 } from '../core/math';
import type { BrickmapMemory } from '../gpu/brickmap';
import type { VerifyResult } from '../gpu/brickmap-verify';
import type { BrickStoreStats } from '../gpu/brick-store';
import type { GpuProfiler } from '../gpu/profiler';
import type { TraceVerifyResult } from '../gpu/trace-verify';

const MB = 2 ** 20;

export interface WorldDebugStats {
  loaded: number;
  inFlight: number;
  missing: number;
  /** Chunks holding a dense block array. */
  dense: number;
  /** Chunks stored as a single block id (no array). */
  uniform: number;
  /** Bytes held by all dense chunk arrays. */
  chunkBytes: number;
  workers: number;
  /** Requests waiting for a free terrain worker. */
  queued: number;
  avgChunkMs: number;
}

export interface DebugOverlayHooks {
  /** Called when a setting that affects render target sizes changes. */
  onResolutionChange(): void;
  /** Current world streaming counters; only polled while the overlay is visible. */
  worldStats(): WorldDebugStats;
  /** Discards the world and regenerates it from `config.terrain`. */
  onRegenerate(): void;
  /** GPU brickmap memory and brick counters; polled while the overlay is visible. */
  brickmapStats(): BrickmapMemory & BrickStoreStats;
  /** Compares GPU getVoxel() with the CPU world. */
  onVerifyBrickmap(): Promise<VerifyResult>;
  /** Compares GPU traceRay() with the CPU voxel raycast. */
  onVerifyTrace(): Promise<TraceVerifyResult>;
  cameraInfo(): { position: Vec3; internal: { width: number; height: number } };
  /** Times the primary pass alone, back to back. */
  onBenchmarkPrimary(): Promise<{ gpuMs: number | null; wallMs: number; avgSteps: number; p95Steps: number }>;
  /** Compares the G-buffer rendered with and without the depth prepass. */
  onVerifyPrepass(): Promise<{ pixels: number; mismatches: number; first?: string }>;
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
  private readonly worldValues = {
    loaded: 0,
    inFlight: 0,
    missing: 0,
    dense: 0,
    uniform: 0,
    chunkMB: '0',
    heapMB: 'n/a',
    workers: 0,
    queued: 0,
    avgChunkMs: '0',
  };
  private readonly brickValues = {
    gridMB: '0',
    poolMB: '0',
    poolUsedMB: '0',
    mixed: 0,
    uniform: 0,
    pending: 0,
    uploadedMB: '0',
    dropped: 0,
    verify: '–',
  };
  private lastWorldStatsUpdate = 0;

  constructor(
    private readonly profiler: GpuProfiler,
    private readonly hooks: DebugOverlayHooks,
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
    this.updateWorldStats();
  }

  private setVisible(visible: boolean): void {
    this.visible = visible;
    this.stats.dom.style.display = visible ? '' : 'none';
    this.gui.show(visible);
  }

  private buildConfigControls(hooks: DebugOverlayHooks): void {
    this.gui.add(config.debug, 'view', [...DEBUG_VIEWS]).name('view');

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

    const world = this.gui.addFolder('World streaming');
    world.add(config.streaming, 'horizontalRadius', 0, 32, 1).name('horizontal radius');
    world.add(config.streaming, 'unloadMargin', 0, 4, 1).name('unload margin');
    world.add(config.streaming, 'maxInFlight', 1, 64, 1).name('max in flight');
    world.add(this.worldValues, 'loaded').name('loaded chunks').disable().listen();
    world.add(this.worldValues, 'inFlight').name('in flight').disable().listen();
    world.add(this.worldValues, 'missing').name('missing').disable().listen();
    world.add(this.worldValues, 'dense').name('dense chunks').disable().listen();
    world.add(this.worldValues, 'uniform').name('uniform chunks').disable().listen();
    world.add(this.worldValues, 'chunkMB').name('chunk data (MB)').disable().listen();
    world.add(this.worldValues, 'heapMB').name('total memory (MB)').disable().listen();
    world.add(this.worldValues, 'workers').name('terrain workers').disable().listen();
    world.add(this.worldValues, 'queued').name('queued').disable().listen();
    world.add(this.worldValues, 'avgChunkMs').name('gen ms/chunk').disable().listen();

    this.buildTerrainControls(hooks);
    this.buildBrickmapControls(hooks);
    this.buildCameraControls(hooks);

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

  private updateWorldStats(): void {
    const now = performance.now();
    if (now - this.lastWorldStatsUpdate < config.debug.worldStatsIntervalMs) return;
    this.lastWorldStatsUpdate = now;
    const s = this.hooks.worldStats();
    this.worldValues.loaded = s.loaded;
    this.worldValues.inFlight = s.inFlight;
    this.worldValues.missing = s.missing;
    this.worldValues.dense = s.dense;
    this.worldValues.uniform = s.uniform;
    this.worldValues.chunkMB = (s.chunkBytes / MB).toFixed(1);
    // Chromium-only, main-thread JS heap (worker heaps are separate isolates).
    const heap = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
    this.worldValues.heapMB = heap ? (heap.usedJSHeapSize / MB).toFixed(1) : 'n/a';
    this.worldValues.workers = s.workers;
    this.worldValues.queued = s.queued;
    this.worldValues.avgChunkMs = s.avgChunkMs.toFixed(2);

    const cam = this.hooks.cameraInfo();
    this.cameraValues.position = cam.position.map((c) => c.toFixed(1)).join(', ');
    this.cameraValues.internal = `${cam.internal.width}×${cam.internal.height}`;

    const b = this.hooks.brickmapStats();
    this.brickValues.gridMB = (b.gridBytes / MB).toFixed(1);
    this.brickValues.poolMB = (b.poolBytes / MB).toFixed(1);
    this.brickValues.poolUsedMB = (b.poolUsedBytes / MB).toFixed(1);
    this.brickValues.mixed = b.mixedBricks;
    this.brickValues.uniform = b.uniformBricks;
    this.brickValues.pending = b.pendingChunks;
    this.brickValues.uploadedMB = (b.uploadedBytes / MB).toFixed(1);
    this.brickValues.dropped = b.droppedBricks;
  }

  private readonly cameraValues = { position: '', internal: '', verify: '–', benchmark: '–', prepass: '–' };

  private buildCameraControls(hooks: DebugOverlayHooks): void {
    const v = this.cameraValues;
    const f = this.gui.addFolder('Camera & rays');
    f.add(config.camera, 'fovY', 30, 120, 1).name('FOV (°)');
    f.add(config.camera, 'speed', 1, 200, 1).name('speed (blocks/s)');
    f.add(config.camera, 'far', 32, 2048, 16).name('far (blocks)');
    f.add(config.camera, 'jitter').name('jitter (Halton 2,3)');
    f.add(config.trace, 'maxSteps', 16, 4096, 1).name('max DDA steps');
    f.add(config.trace, 'prepass').name('depth prepass');
    f.add(v, 'position').name('position').disable().listen();
    f.add(v, 'internal').name('traced resolution').disable().listen();
    f.add(v, 'verify').name('verify result').disable().listen();
    const actions = {
      verify: () => {
        v.verify = 'running…';
        hooks.onVerifyTrace().then(
          (r) => {
            v.verify = `${r.agree}/${r.rays} agree (${r.hits} hits), ${r.ms.toFixed(0)} ms`;
            (r.agree === r.rays ? console.info : console.warn)('[trace] verify:', r);
          },
          (err: unknown) => {
            v.verify = `error: ${err instanceof Error ? err.message : String(err)}`;
          },
        );
      },
    };
    f.add(actions, 'verify').name('Verify rays (GPU vs CPU)');
    f.add(v, 'benchmark').name('primary benchmark').disable().listen();
    const bench = {
      run: () => {
        v.benchmark = 'running…';
        hooks.onBenchmarkPrimary().then(
          (r) => {
            v.benchmark =
              `${r.gpuMs?.toFixed(2) ?? 'n/a'} ms gpu, ${r.wallMs.toFixed(2)} ms wall, ` +
              `steps avg ${r.avgSteps.toFixed(1)} p95 ${r.p95Steps}`;
            console.info('[benchmark] primary pass:', r);
          },
          (err: unknown) => {
            v.benchmark = `error: ${err instanceof Error ? err.message : String(err)}`;
          },
        );
      },
    };
    f.add(bench, 'run').name(`Benchmark primary (×${config.debug.benchmarkIterations})`);
    f.add(v, 'prepass').name('prepass verify').disable().listen();
    const prepass = {
      run: () => {
        v.prepass = 'running…';
        hooks.onVerifyPrepass().then(
          (r) => {
            v.prepass = `${r.pixels - r.mismatches}/${r.pixels} identical`;
            (r.mismatches === 0 ? console.info : console.warn)('[prepass] verify:', r);
          },
          (err: unknown) => {
            v.prepass = `error: ${err instanceof Error ? err.message : String(err)}`;
          },
        );
      },
    };
    f.add(prepass, 'run').name('Verify prepass (with vs without)');
  }

  private buildBrickmapControls(hooks: DebugOverlayHooks): void {
    const v = this.brickValues;
    const f = this.gui.addFolder('GPU brickmap');
    f.add(config.debug, 'topdownBlocksPerPixel', 0.25, 8, 0.25).name('top-down zoom (blocks/px)');
    f.add(v, 'gridMB').name('grid (MB)').disable().listen();
    f.add(v, 'poolMB').name('pool allocated (MB)').disable().listen();
    f.add(v, 'poolUsedMB').name('pool used (MB)').disable().listen();
    f.add(v, 'mixed').name('mixed bricks').disable().listen();
    f.add(v, 'uniform').name('uniform bricks').disable().listen();
    f.add(v, 'pending').name('upload queue (chunks)').disable().listen();
    f.add(v, 'uploadedMB').name('uploaded total (MB)').disable().listen();
    f.add(v, 'dropped').name('dropped (pool full)').disable().listen();
    f.add(v, 'verify').name('verify result').disable().listen();
    const actions = {
      verify: () => {
        v.verify = 'running…';
        hooks.onVerifyBrickmap().then(
          (r) => {
            v.verify = `${r.mismatches}/${r.samples} wrong (${r.solid} solid), ${r.ms.toFixed(0)} ms`;
            const log = r.mismatches === 0 ? console.info : console.error;
            log('[brickmap] verify:', r);
          },
          (err: unknown) => {
            v.verify = `error: ${err instanceof Error ? err.message : String(err)}`;
          },
        );
      },
    };
    f.add(actions, 'verify').name('Verify brickmap (GPU vs CPU)');
  }

  /** Terrain params only take effect on "Regenerate" (the workers hold their own copy). */
  private buildTerrainControls(hooks: DebugOverlayHooks): void {
    const t = config.terrain;
    const f = this.gui.addFolder('Terrain (regenerate to apply)');
    const seed = f.add(t, 'seed', 0, 2 ** 31 - 1, 1);
    f.add(t, 'seaLevel', 0, 200, 1).name('sea level');
    f.add(t.continentalness, 'frequency', 0.0001, 0.005, 0.0001).name('continent freq');
    f.add(t.erosion, 'frequency', 0.0001, 0.005, 0.0001).name('erosion freq');
    f.add(t.peaksValleys, 'frequency', 0.0005, 0.01, 0.0001).name('peaks freq');
    f.add(t.warp, 'amplitude', 0, 150, 1).name('warp amplitude');
    f.add(t.detail, 'amplitude', 0, 10, 0.1).name('detail amplitude');
    f.add(t.caves, 'cheeseThreshold', 0.2, 1, 0.01).name('cheese threshold');
    f.add(t.caves, 'spaghettiWidth', 0, 0.2, 0.005).name('spaghetti width');
    f.add(t.trees, 'cellSize', 3, 16, 1).name('tree cell size');
    f.add(t.trees, 'treeline', 64, 300, 1);
    const actions = {
      regenerate: () => hooks.onRegenerate(),
      randomSeed: () => {
        t.seed = Math.floor(Math.random() * 2 ** 31);
        seed.updateDisplay();
        hooks.onRegenerate();
      },
    };
    f.add(actions, 'regenerate').name('Regenerate');
    f.add(actions, 'randomSeed').name('Random seed + regenerate');
    f.close();
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
