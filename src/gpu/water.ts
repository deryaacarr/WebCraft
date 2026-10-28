import { config } from '../config';

/** WaterParams in water.wgsl: four 16-byte rows. */
const PARAMS_SIZE = 64;

type Rgb = [number, number, number];

/** Absorption and scattering coefficients per metre (water.wgsl) from the config. */
export function waterCoefficients(w = config.water): { absorption: Rgb; scattering: Rgb } {
  const absorption = w.absorptionColor.map((c) => -Math.log(Math.min(Math.max(c, 1e-4), 1)) / Math.max(w.absorptionDistance, 1e-3)) as Rgb;
  const s = w.maxScattering + (w.minScattering - w.maxScattering) * Math.min(Math.max(w.clarity, 0), 1);
  return { absorption, scattering: [s, s, s] };
}

/** Water parameters uniform, shared by the passes that trace or shade water. */
export class WaterSystem {
  readonly params: GPUBuffer;
  private readonly data = new ArrayBuffer(PARAMS_SIZE);

  constructor(private readonly device: GPUDevice) {
    this.params = device.createBuffer({ label: 'water-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Uploads this frame's parameters (called by the primary pass, the first reader). */
  update(): void {
    const w = config.water;
    const { absorption, scattering } = waterCoefficients(w);
    const f32 = new Float32Array(this.data);
    f32.set([...absorption, w.ior, ...scattering, w.maxDistance, ...w.scatterColor, w.shadowBlur, w.detailBlur], 0);
    new Uint32Array(this.data)[13] = w.enabled ? 1 : 0;
    f32[14] = w.sunRoughness;
    this.device.queue.writeBuffer(this.params, 0, this.data);
  }
}
