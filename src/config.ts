/**
 * Single source of truth for every tunable parameter.
 * Anything exposed here can be edited live from the debug panel (F3).
 */
export const DEBUG_VIEWS = ['lit', 'albedo', 'normal', 'depth', 'steps', 'motion', 'uv', 'material', 'topdown', 'gradient'] as const;
export type DebugView = (typeof DEBUG_VIEWS)[number];

/** Spline control points; a helper so config stays plain mutable data with a precise type. */
function points(...p: [number, number][]): [number, number][] {
  return p;
}

export const config = {
  sim: {
    /** Fixed simulation rate in Hz. */
    tickRate: 60,
    /** Longest frame delta (s) fed into the accumulator; prevents spiral of death after stalls. */
    maxFrameTime: 0.25,
  },
  render: {
    /** Internal (traced) resolution relative to the canvas backing size, upscaled in the
     *  blit pass. 0.5 on a Retina canvas ≈ 1440×800 primary rays. */
    renderScale: 0.5,
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
  world: {
    /** log2 of the chunk edge length (5 → 32³). Structural: read once at startup. */
    chunkBits: 5,
    /** Lowest block Y that exists (inclusive). Chunks entirely below are never loaded. */
    minY: -64,
    /** Top of the world (exclusive). Chunks entirely at or above are never loaded. */
    maxY: 320,
    /** log2 of the GPU brick edge length (3 → 8³ voxels). Structural. */
    brickBits: 3,
  },
  brickmap: {
    /** Toroidal brick grid width on X and Z, in chunks (power of two). Must exceed the
     *  streamed diameter 2·(horizontalRadius + unloadMargin) + 1. Vertically the grid
     *  spans world.minY..world.maxY. */
    gridChunksXZ: 32,
    /** Initial brick pool size (mixed bricks); grows on demand up to the device limit. */
    initialPoolBricks: 81920,
    /** Pool growth factor when it runs out of slots. */
    poolGrowth: 1.5,
    /** Main-thread time per frame spent packing and uploading changed chunks (ms). */
    uploadBudgetMs: 3,
    /** Positions sampled by the "verify brickmap" debug check. */
    verifySamples: 65536,
  },
  generation: {
    /** Terrain worker count; 0 = navigator.hardwareConcurrency - 1. */
    workers: 0,
    /** Column height grids cached per worker (one per chunk column, shared by its vertical stack). */
    heightCacheColumns: 512,
  },
  /**
   * Terrain shape. Plain data: sent to the workers as-is. Changing it needs "Regenerate".
   * Heights are world Y in blocks. Spline points are [noise value, output].
   */
  terrain: {
    seed: 1337,
    seaLevel: 64,
    /** Large-scale land height: lowlands ↔ highlands. */
    continentalness: { frequency: 0.0006, octaves: 4 },
    /** Low erosion = rugged mountains, high erosion = flattened terrain. */
    erosion: { frequency: 0.0011, octaves: 3 },
    /** Peaks & valleys: ridged fractal noise, connected crests near +1, valleys between. */
    peaksValleys: { frequency: 0.0028, octaves: 4 },
    /** Small-scale roughness added to the final height. */
    detail: { frequency: 0.03, octaves: 3, amplitude: 2.5 },
    /** Domain warp applied to all 2D terrain noise (bends ridges and valleys). */
    warp: { frequency: 0.0015, amplitude: 35 },
    // Spline inputs span the practical noise range (~±0.7; the tails beyond are rare).
    /** continentalness → base height. */
    continentalnessSpline: points([-0.7, 44], [-0.4, 60], [-0.15, 70], [0.15, 80], [0.45, 96], [0.7, 112]),
    /** erosion → how strongly peaks/valleys apply (0..1). */
    erosionSpline: points([-0.6, 1], [-0.2, 0.9], [0.1, 0.55], [0.35, 0.22], [0.7, 0.08]),
    /** peaks/valleys → height offset (scaled by the erosion factor). */
    peaksValleysSpline: points([-0.7, -32], [-0.35, -16], [0, 6], [0.25, 45], [0.5, 115], [0.75, 190]),
    caves: {
      /** Distance between cave-noise samples; values are trilinearly interpolated between them. */
      latticeStep: 4,
      cheeseFrequency: 0.016,
      /** Vertical squash of cheese caves (>1 = flatter, wider caverns). */
      cheeseVerticalScale: 1.6,
      /** Noise above this is carved; higher = fewer, smaller caverns. */
      cheeseThreshold: 0.6,
      spaghettiFrequency: 0.011,
      /** Tunnel half-width in noise units; the intersection of two thin shells forms a tube. */
      spaghettiWidth: 0.055,
      /** Cheese caverns stay at least this deep; spaghetti tunnels may break the surface. */
      cheeseMinDepth: 10,
      /** Under water, nothing is carved within this depth of the lake/sea floor. */
      underwaterMinDepth: 8,
    },
    surface: {
      dirtDepthMin: 3,
      dirtDepthMax: 4,
      /** Max height difference to a neighbour column before gravel shows. */
      gravelSlope: 3,
      /** Max height difference before bare stone shows. */
      stoneSlope: 5,
      /** Gravel layer thickness on steep slopes. */
      gravelDepth: 2,
      /** Columns up to seaLevel + beachHeight are sand. */
      beachHeight: 1,
      sandDepth: 3,
      /** Above this Y the ground is bare rock. */
      rockAltitude: 200,
    },
    trees: {
      /** One candidate per cell (jittered); smaller = denser possible forest. */
      cellSize: 5,
      forestFrequency: 0.006,
      /** Forest noise → acceptance probability. */
      densitySpline: points([-1, 0], [-0.3, 0.05], [0, 0.4], [0.35, 0.9], [1, 0.95]),
      /** No trees above this Y (treeline). */
      treeline: 175,
      /** Density fades to zero over this many blocks below the treeline. */
      treelineFade: 30,
      /** Max neighbour height difference at the trunk. */
      maxSlope: 2,
      trunkMin: 4,
      trunkMax: 7,
      /** Canopy radius (blocks) around the trunk. */
      leafRadius: 2,
    },
  },
  streaming: {
    /** Horizontal load radius in chunks (circular, measured on XZ). Vertically the whole
     *  world column between world.minY and world.maxY is loaded. */
    horizontalRadius: 12,
    /** Extra chunks beyond the load radius before a chunk is unloaded (hysteresis). */
    unloadMargin: 1,
    /** Max chunk requests in flight at once (should exceed the worker count to keep all busy). */
    maxInFlight: 32,
  },
  camera: {
    /** Vertical field of view in degrees. */
    fovY: 70,
    /** Near plane (blocks); used for projection / motion vectors. */
    near: 0.05,
    /** Far plane (blocks); rays stop here. */
    far: 1024,
    /** Fly speed in blocks per second. */
    speed: 20,
    /** Speed multiplier while ControlLeft is held. */
    boostMultiplier: 5,
    /** Pitch limit in degrees (avoids flipping over the poles). */
    maxPitch: 89,
    /** Sub-pixel Halton(2,3) jitter for TAA. Off until TAA exists (it shimmers without it). */
    jitter: false,
    /** Jitter sequence length before it repeats. */
    jitterSequenceLength: 8,
    /** Spawn height above the ground. */
    spawnHeight: 30,
  },
  trace: {
    /** Upper bound on DDA iterations (brick steps + voxel steps) per primary ray. */
    maxSteps: 512,
    /** Rays checked by the "verify rays" debug tool. */
    verifySamples: 4096,
    /** Ray length for the verify tool (inside the resident window). */
    verifyMaxDistance: 200,
    /** Step cap for the verify tool: high enough that no ray is cut short. */
    verifyMaxSteps: 4096,
  },
  input: {
    /** Radians per pixel of mouse movement. */
    mouseSensitivity: 0.002,
  },
  debug: {
    /** What the screen shows. G-buffer views come from the primary ray pass. */
    view: 'lit' as DebugView,
    /** Top-down brickmap view zoom: world blocks per screen pixel. */
    topdownBlocksPerPixel: 1,
    /** Direction towards the sun for the placeholder "lit" view. */
    sunDirection: [0.45, 0.8, 0.35] as [number, number, number],
    /** Motion view: UV delta multiplier before display. */
    motionScale: 20,
    /** Depth view: depth (blocks) at which the grey ramp reaches 50 %. */
    depthHalf: 64,
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
    /** Refresh interval of the world/streaming/memory counters in milliseconds. */
    worldStatsIntervalMs: 250,
  },
};

export type Config = typeof config;
export type TerrainParams = Config['terrain'];
