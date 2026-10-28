import { config } from '../config';

/** WaterParams in water.wgsl: six 16-byte rows. */
const PARAMS_SIZE = 96;

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

/** Water parameters uniform, shared by the passes that trace or shade water. */
export class WaterSystem {
  readonly params: GPUBuffer;
  private readonly data = new ArrayBuffer(PARAMS_SIZE);

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
    const { wind } = config;
    const a = (wind.directionDeg * Math.PI) / 180;
    // Wind waves grow with the square of the wind speed (calm lake ≈ 2 m/s).
    const amplitude = w.waveStrength * w.waveHeight * (wind.speed / 2) ** 2;
    f32.set([Math.sin(a), -Math.cos(a), amplitude, time % 3600, w.wavelength, w.ripples * Math.min(wind.speed / 2, 2), w.level], 16);
    this.device.queue.writeBuffer(this.params, 0, this.data);
  }
}
