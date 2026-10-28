import { config } from '../config';
import { BlockId } from '../world/blocks';

/** WaterParams in water.wgsl: 16 16-byte rows. */
const PARAMS_SIZE = 256;
/** Local wet zones the shader takes (wetness.wgsl). */
export const MAX_WET_ZONES = 4;

type Rgb = [number, number, number];

/** Absorption and scattering coefficients per metre (water.wgsl) from the config. */
export function waterCoefficients(w = config.water): { absorption: Rgb; scattering: Rgb } {
  const absorption = w.absorptionColor.map((c) => -Math.log(Math.min(Math.max(c, 1e-4), 1)) / Math.max(w.absorptionDistance, 1e-3)) as Rgb;
  const s = w.maxScattering + (w.minScattering - w.maxScattering) * Math.min(Math.max(w.clarity, 0), 1);
  return { absorption, scattering: [s, s, s] };
}

/** Light left after a refracted ray's full length: beyond it the bottom cannot be seen. */
const TRACE_CUTOFF = 0.01;

/** Refracted rays' length: `maxDistance`, or less where even the clearest channel has
 *  dropped to TRACE_CUTOFF (ln 100 / σ_t ≈ 24 m in the default lake) — near-horizontal
 *  refracted rays under a wavy surface walked the voxels just below it up to the full
 *  48 m. */
export function waterTraceDistance(w = config.water): number {
  const { absorption, scattering } = waterCoefficients(w);
  const clearest = Math.min(...absorption.map((a, i) => a + scattering[i]!));
  return Math.min(w.maxDistance, Math.log(1 / TRACE_CUTOFF) / Math.max(clearest, 1e-4));
}

/** Swell height scale (metres) for the current wind (water-waves.wgsl `wave_amplitude`). */
export function waveAmplitude(w = config.water, wind = config.wind): number {
  // Wind waves grow with the square of the wind speed (calm lake ≈ 2 m/s).
  return w.waveStrength * w.waveHeight * (wind.speed / 2) ** 2;
}

/** Unit direction the waves travel (x, z): towards `wind.directionDeg` (0 = north, −Z). */
export function windDirection(wind = config.wind): [number, number] {
  const a = (wind.directionDeg * Math.PI) / 180;
  return [Math.sin(a), -Math.cos(a)];
}

/** Swell height above the rest level at world (x, z) and `time` seconds: mirrors
 *  waterWaveHeight in water-waves.wgsl (constants and all, footprint 0 — no fading), so
 *  the CPU knows where the visible surface is (camera under water or not). */
export function waveHeight(x: number, z: number, time: number, w = config.water, wind = config.wind): number {
  const amplitude = waveAmplitude(w, wind);
  if (amplitude <= 0) return 0;
  const [dx, dz] = windDirection(wind);
  const t = time % 3600;
  const spread = [0, 0.55, -0.4, 1.0];
  let h = 0;
  for (let i = 0; i < spread.length; i++) {
    const lambda = w.wavelength * 0.57 ** i;
    const c = Math.cos(spread[i]!);
    const s = Math.sin(spread[i]!);
    const d = [c * dx - s * dz, s * dx + c * dz];
    const k = (2 * Math.PI) / lambda;
    h += amplitude * (lambda / w.wavelength) * Math.sin(k * (d[0]! * x + d[1]! * z) - Math.sqrt(9.81 * k) * t + i * 1.7);
  }
  return h;
}

/** Whether a point is under the water surface (waves included), and the surface's rest
 *  level above it (absolute y). `getBlock`: world lookup. */
export function waterAt(getBlock: (x: number, y: number, z: number) => number, x: number, y: number, z: number, time: number): { under: boolean; surfaceY: number } {
  const cx = Math.floor(x);
  const cz = Math.floor(z);
  let top = Math.floor(y);
  if (!config.water.enabled || getBlock(cx, top, cz) !== BlockId.water) return { under: false, surfaceY: 0 };
  // Up to the top water voxel (the surface lies inside it).
  for (let i = 0; i < 256 && getBlock(cx, top + 1, cz) === BlockId.water; i++) top++;
  const surfaceY = top + config.water.level;
  return { under: y < surfaceY + waveHeight(x, z, time), surfaceY };
}

/** Water parameters uniform, shared by the passes that trace or shade water. */
export class WaterSystem {
  readonly params: GPUBuffer;
  private readonly data = new ArrayBuffer(PARAMS_SIZE);
  /** Camera under water (primary UNDERWATER pipeline) and the surface's rest level above it
   *  relative to the camera's cell (set by the renderer every frame). */
  cameraUnder = false;
  cameraSurface = 0;

  constructor(private readonly device: GPUDevice) {
    this.params = device.createBuffer({ label: 'water-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Uploads this frame's parameters (called by the primary pass, the first reader);
   *  `time` in seconds drives the waves. */
  update(time: number): void {
    const w = config.water;
    const { absorption, scattering } = waterCoefficients(w);
    const f32 = new Float32Array(this.data);
    f32.set([...absorption, w.ior, ...scattering, waterTraceDistance(w), ...w.scatterColor, w.shadowBlur, w.detailBlur], 0);
    new Uint32Array(this.data)[13] = w.enabled ? 1 : 0;
    f32[14] = w.sunRoughness;
    const { wind, wetness } = config;
    f32.set([...windDirection(wind), waveAmplitude(w, wind), time % 3600, w.wavelength, w.ripples * Math.min(wind.speed / 2, 2), w.level], 16);
    f32.set([w.turbidity, w.turbidityDepth, w.foam, w.foamWidth, ...w.sedimentColor, w.caustics, w.wetShore, w.wetMaxDistance], 24);
    const u32 = new Uint32Array(this.data);
    u32[36] = this.cameraUnder ? 1 : 0;
    f32[37] = this.cameraSurface;
    f32.set([wetness.global, wetness.puddles, wetness.puddleScale], 38);
    const zones = wetness.zones.slice(0, MAX_WET_ZONES);
    u32[41] = zones.length;
    f32[42] = w.lightShafts;
    zones.forEach((z, i) => {
      f32.set([z.x, z.y, z.z, z.radius], 44 + i * 4);
      f32[60 + i] = z.amount;
    });
    this.device.queue.writeBuffer(this.params, 0, this.data);
  }
}
