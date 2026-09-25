/**
 * Single source of truth for every tunable parameter.
 * Anything exposed here can be edited live from the debug panel (F3).
 */
export const config = {
  sim: {
    /** Fixed simulation rate in Hz. */
    tickRate: 60,
    /** Longest frame delta (s) fed into the accumulator; prevents spiral of death after stalls. */
    maxFrameTime: 0.25,
  },
  render: {
    /** Internal resolution relative to the canvas backing size (upscaled in the blit pass). */
    renderScale: 1.0,
    /** Upper bound on devicePixelRatio to keep 4K/Retina displays affordable. */
    maxPixelRatio: 2.0,
    /** Workgroup edge length for 2D compute passes; must match @workgroup_size in WGSL. */
    workgroupSize: 8,
  },
  gradient: {
    /** Animation speed of the test gradient (cycles per second). */
    speed: 0.1,
    /** Blend strength of the animated wave on top of the UV gradient. */
    waveStrength: 0.25,
  },
  input: {
    /** Radians per pixel of mouse movement. */
    mouseSensitivity: 0.002,
  },
  debug: {
    /** Key that toggles the debug overlay. */
    toggleKey: 'F3',
    /** Show the overlay on start. */
    visibleOnStart: false,
    /** Number of frames the GPU profiler averages over. */
    profilerSmoothing: 30,
    /** Max number of profiled passes per frame (2 timestamps each). */
    profilerMaxPasses: 16,
    /** Readback buffers in flight; frames with none free are skipped. */
    profilerReadbackBuffers: 4,
    /** Refresh interval of the stats-gl GPU panel in milliseconds. */
    gpuPanelIntervalMs: 100,
    /** Per-update decay of the GPU panel graph scale, so it shrinks back after spikes. */
    gpuPanelMaxDecay: 0.99,
  },
};

export type Config = typeof config;
