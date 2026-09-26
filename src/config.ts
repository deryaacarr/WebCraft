/**
 * Single source of truth for every tunable parameter.
 * Anything exposed here can be edited live from the debug panel (F3).
 */
export const TEXTURE_RESOLUTIONS = [16, 32, 64, 128, 256] as const;
export type TextureResolution = (typeof TEXTURE_RESOLUTIONS)[number];

export const DEBUG_VIEWS = [
  'lit',
  'albedo',
  'normal',
  'depth',
  'steps',
  'motion',
  'roughness',
  'ao',
  'material',
  'shadow',
  'skyvis',
  'history',
  'gi',
  'topdown',
  'gradient',
] as const;
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
    /** Tone mapper for the lit view (ACES fit = the stage 7 reference look; AgX options). */
    tonemapper: 'aces' as 'agx' | 'agx-punchy' | 'aces',
    /** AgX punchy look, applied in AgX log space: contrast power and saturation. */
    agxPunchyContrast: 1.15,
    agxPunchySaturation: 1.1,
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
    /** Workgroup [x, y] of the primary ray and prepass shaders. Structural (reload). */
    workgroup: [8, 8] as [number, number],
    /** Depth prepass: trace one conservative cone per tile first; full rays start there. */
    prepass: true,
    /** Depth prepass tile edge in pixels (the prepass runs at 1/tile resolution). */
    prepassTile: 8,
    /** Distance (blocks) subtracted from the prepass result before the full rays use it. */
    prepassSafety: 0.5,
    /** Safety factor on the tile's cone radius. */
    prepassConeMargin: 1.1,
    /** Iteration cap for a prepass cone. */
    prepassMaxSteps: 256,
    /** Distance field: largest stored distance in bricks (cost per update ∝ 2·max + 1). */
    distanceMax: 8,
    /** Distance field passes: workgroup edge (cubic). */
    distanceWorkgroup: 4,
    /** Rays checked by the "verify rays" debug tool. */
    verifySamples: 4096,
    /** Ray length for the verify tool (inside the resident window). */
    verifyMaxDistance: 200,
    /** Step cap for the verify tool: high enough that no ray is cut short. */
    verifyMaxSteps: 4096,
  },
  textures: {
    /** Texels per block face: 16, 32, 64, 128 or 256 ("Ultra"). Needs `npm run textures`. */
    resolution: 128 as TextureResolution,
    /** Added to the ray-cone texture LOD (positive = blurrier, less shimmer). */
    lodBias: 0,
    /** Sampler anisotropy (1 = off). */
    maxAnisotropy: 8,
    /** Parallax occlusion mapping for materials that enable it (cobblestone, gravel, dirt). */
    pom: true,
    /** POM depth in blocks. */
    pomDepth: 0.06,
    pomSteps: 12,
    /** Beyond this distance (blocks) parallax is sub-pixel and skipped. */
    pomMaxDistance: 32,
    /** Opacity below which alpha-tested texels (leaves) let rays through. */
    alphaCutoff: 0.5,
    /** Texture variants are chosen per block from a low-frequency world-space noise, so
     *  equal-toned variants form regions and veins instead of a checkerboard. Feature
     *  size in blocks (0 = independent random choice per block and face). */
    variantRegionScale: 24,
    /** Domain warp of the region noise (in feature sizes): bends regions into veins. */
    variantWarp: 0.6,
  },
  /** Natural material detail (stage 8.5). Each feature can be toggled in the "Detail" panel. */
  detail: {
    /** Natural materials (stone, dirt, gravel, sand) sample world-space layers covering
     *  2 × 2 m, so walls read as one surface instead of a block grid. */
    worldTextures: true,
  },
  sky: {
    /** Time of day at start (hours, 12 = solar noon). */
    timeOfDay: 9.5,
    /** Day of the lunar cycle at start (≈ 7 = first quarter, 14.8 = full moon). */
    startDay: 10,
    /** Real-time minutes for one full in-game day (0 = time stands still). */
    dayLengthMinutes: 20,
    paused: false,
    /** Observer latitude (degrees) and season (sun declination, degrees). */
    latitude: 45,
    sunDeclination: 15,
    /** Tilt of the moon's path against the sun's (degrees). */
    moonInclination: 5.1,
    /** Sun illuminance at the top of the atmosphere; the unit of all lighting. */
    sunIlluminance: 1,
    /** Visible (and soft-shadow) angular radii, degrees. The real sun is 0.27°. */
    sunAngularRadius: 0.3,
    moonAngularRadius: 0.26,
    moonAlbedo: 0.12,
    /** Moonlight is ~400 000× dimmer than sunlight; this lifts nights to a playable level. */
    nightBoost: 60,
    /** Star brightness relative to the sun (after the night boost). */
    starBrightness: 0.004,
    /** Viewer altitude above the planet surface at sea level (km); +1 m per block above. */
    seaLevelAltitudeKm: 0.2,
    /** Aerial perspective LUT (Hillaire 2020 §5.5): camera-aligned froxels holding the
     *  in-scattered light and transmittance between the camera and each depth slice. */
    aerialPerspective: {
      /** Off by default: the stage 7 reference look (docs/sky-*.png at 4b24741) has none. */
      enabled: false,
      /** Froxel grid: screen resolution (x = y) and depth slices. */
      resolution: 32,
      slices: 32,
      /** View depth (km) of the last slice; slices are spaced quadratically (denser near). */
      maxDepthKm: 4,
      /** Ray-march samples per slice. */
      samplesPerSlice: 2,
    },
    /** Earth-like atmosphere (km⁻¹, km), Hillaire 2020 / Bruneton defaults. */
    atmosphere: {
      bottomRadius: 6360,
      topRadius: 6460,
      rayleighScattering: [5.802e-3, 13.558e-3, 33.1e-3] as [number, number, number],
      rayleighScaleHeight: 8,
      mieScattering: 3.996e-3,
      /** Stage 7 reference value (Hillaire's own split would be 0.444e-3: extinction
       *  4.44e-3 = scattering 3.996e-3 + absorption); changing it changes the sky's look. */
      mieAbsorption: 4.4e-3,
      mieScaleHeight: 1.2,
      mieG: 0.8,
      ozoneAbsorption: [0.65e-3, 1.881e-3, 0.085e-3] as [number, number, number],
      ozoneCenter: 25,
      ozoneWidth: 30,
      /** Low haze layer (boundary-layer aerosols over humid mountain forest): meteorological
       *  visibility at the planet surface (km, 0 = none), scale height (km) and
       *  single-scattering albedo. Clear-air aerosols above alone give ~100 km+. */
      hazeVisibilityKm: 0,
      hazeScaleHeight: 0.5,
      hazeAlbedo: 0.9,
      /** Planet surface beyond the loaded world (forest-like, linear RGB). */
      groundAlbedo: [0.07, 0.09, 0.05] as [number, number, number],
    },
  },
  lighting: {
    /** Visibility history cap (frames) for a still camera / a fast-moving one. */
    historyStill: 16,
    historyMoving: 4,
    /** Screen motion (pixels per frame) at which a pixel counts as fully moving. */
    historyMotionPixels: 8,
    /** Camera translation (blocks/frame) and rotation (°/frame) that count as fully moving. */
    historyCameraSpeed: 0.5,
    historyCameraTurn: 1.5,
    /** Neighbourhood clipping: history is clipped to mean ± k·σ of the current samples. */
    historyClipK: 1.25,
    /** Reject history when the reprojected depth differs by more than this fraction. */
    temporalDepthTolerance: 0.05,
    /** Trace visibility for one pixel of each 2×2 block per frame (rotating); the temporal
     *  pass fills in the rest. ~4× cheaper; slightly slower to converge. */
    visibilityCheckerboard: true,
    /** Sky-visibility rays: max distance (blocks). Short = local occlusion only. */
    skyVisibilityDistance: 12,
    /** Step cap for sky-visibility rays (they only need local occlusion). */
    skyVisibilitySteps: 32,
    /** Shadow rays stop here (blocks). */
    shadowDistance: 512,
    /** Light-emitting blocks: black-body colour temperature (K) and mean radiance of a
     *  block face (relative units: sun illuminance = 1). Falls off with 1 / d² (ReSTIR DI).
     *  The glowing texels are brighter: the mean is spread over the emissive part only. */
    emitters: {
      torch: { temperature: 1950, radiance: 0.2 },
      lava: { temperature: 1700, radiance: 0.2 },
    } as Record<string, { temperature: number; radiance: number }>,
    /** Flicker of emitted light: relative amplitude, speed (Hz-ish) and the spatial size
     *  (blocks) of the smooth noise field, so each torch's surroundings flicker on their own. */
    flicker: { amount: 0.15, speed: 3, scale: 6 },
    /** Light passing through leaves (subsurface scattering approximation). */
    subsurface: 0.5,
    /** Fraction of light a leaf voxel lets through for shadow / sky rays. */
    leafTransmission: 0.3,
    /** Shadow rays carry light through at most this many opaque leaf texels. */
    maxLeafLayers: 6,
    /** 'physical': sun + atmosphere only. 'debug-fill': adds the non-physical fill lights
     *  below, only for comparison (not a fix — GI will supply bounce light). */
    model: 'physical' as 'physical' | 'debug-fill',
    debugFill: {
      /** Sky ambient raised (in the sun's colour) until direct : sky is at most this. */
      maxDirectToSky: 8,
      /** Minimum sky visibility. */
      ambientFloor: 0.12,
      /** Bounce fill: surroundings albedo (linear RGB) and share arriving from all directions. */
      bounceAlbedo: [0.18, 0.2, 0.14] as [number, number, number],
      bounceIsotropic: 0.1,
    },
  },
  /** Global illumination: half-resolution path tracing (1 bounce), ReSTIR DI for emissive
   *  blocks, SVGF denoising. Replaces the approximate sky-ambient term when enabled. */
  gi: {
    enabled: true,
    /** GI resolution divisor: the GI grid is (width / d) × (height / d); 3 keeps GI +
     *  denoising within ~5 ms in dense scenes at 1440×800 (2 = sharper, costlier). */
    resolutionDivisor: 3,
    /** Half-res pixels traced per frame: all, half (alternating checkerboard) or a
     *  quarter (one per 2×2 block, rotating); temporal accumulation fills the rest. */
    tracePattern: 'quarter' as 'all' | 'half' | 'quarter',
    /** Bounce hits visible on screen reuse last frame's lit radiance (cheap, adds further
     *  bounces); off = always shade them with shadow and sky rays. */
    screenReuse: true,
    /** Bounce ray length (blocks) and DDA step budget; beyond it the sky LUT is used. */
    range: 48,
    maxSteps: 96,
    /** Shadow rays from bounce hits (and to emitters). */
    shadowDistance: 96,
    shadowMaxSteps: 96,
    /** Short sky ray from bounce hits (enclosed hits, e.g. in caves, get no sky). */
    skyDistance: 12,
    skyMaxSteps: 32,
    /** Albedo at bounce hits from a coarse mip: this many texels across a face (1 = the
     *  material's mean colour). */
    hitTexels: 1,
    /** Surfaces with (1 − roughness)² below this get no specular rays (rough specular
     *  comes from the diffuse irradiance). */
    specularThreshold: 0.25,
    /** Per-sample luminance ceiling (pre-exposed; mid-grey ≈ 0.18) against fireflies. */
    fireflyClamp: 16,
    /** Temporal accumulation cap (frames) for a still / fast-moving camera. */
    historyStill: 32,
    historyMoving: 8,
    /** À-trous iterations for diffuse / specular, luminance edge-stopping strength (× σ),
     *  specular roughness edge-stopping. */
    atrousIterations: 5,
    specularIterations: 3,
    sigmaLuminance: 4,
    sigmaLuminanceSpecular: 2,
    /** "Same surface" tolerance (blocks) between face planes: absolute, since parallel
     *  voxel faces lie whole blocks apart. */
    planeTolerance: 0.25,
    specularRoughnessSigma: 10,
    /** Which GI signal shading uses (debug): denoised / temporally accumulated / raw 1 spp. */
    debugSignal: 'denoised' as 'denoised' | 'accumulated' | 'raw',
    /** ReSTIR DI for emissive blocks (torch, lava). */
    restir: {
      enabled: true,
      /** Initial RIS candidates per pixel and frame. */
      candidates: 8,
      /** Temporal reuse: history capped at this × candidates samples. */
      temporalMaxMFactor: 20,
      /** Spatial reuse: neighbours and radius (half-res pixels). */
      spatialSamples: 3,
      spatialRadius: 12,
      /** Light list: emitters within this distance of the camera, nearest first. */
      lightRadius: 64,
      maxLights: 1024,
      /** Rebuild the light list when the camera moved this far (blocks). */
      rebuildDistance: 4,
    },
  },
  exposure: {
    /** Target mid-grey after exposure. */
    key: 0.18,
    /** Manual correction in EV (stops). */
    compensation: 0,
    /** Adaptation time constants (s): scene getting darker (eyes open up) / brighter. */
    adaptDarkerSeconds: 3.5,
    adaptBrighterSeconds: 0.5,
    /** Metering: drop this fraction of the weight at each end of the histogram. */
    trim: 0.05,
    /** Centre weighting: Gaussian sigma as a fraction of the half screen (0 = uniform,
     *  like the stage 7 reference). */
    centerSigma: 0,
    /** Sky pixels count this much relative to ground (1 = like the stage 7 reference). */
    skyWeight: 1,
    /** Exposure limits as log2 multipliers of the lighting unit. */
    minEv: -2,
    maxEv: 16,
    /** Dark adaptation limit: at most this many EV brighter than for a mid-grey surface in
     *  the open (sun + sky there). Keeps caves and deep shade dark by day; nights (dark
     *  outside too) still get the full range. */
    maxDarkAdaptationEv: 8,
  },
  whiteBalance: {
    /** 'auto': neutralise the scene light like a camera; 'manual': fixed temperature. */
    mode: 'off' as 'auto' | 'manual' | 'off',
    /** Manual white point (K). 6500 = neutral daylight. */
    temperature: 6500,
    /** Fraction of the cast removed (1 = fully neutral). < 1 keeps sunsets warm, nights blue. */
    strength: 0.4,
    /** Auto adaptation time constant (s). */
    adaptSeconds: 4,
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
    /** Scripted camera motion for reproducible motion tests: yaw rate (°/s) and forward
     *  flight (blocks/s). URL: &spin=90&fly=20. */
    cameraSpin: 0,
    cameraFly: 0,
    /** "Place test torches" button / &torches=N: count and radius around the camera. */
    testTorches: 12,
    testTorchRadius: 16,
    /** Back-to-back primary passes timed by "Benchmark primary". */
    benchmarkIterations: 50,
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
