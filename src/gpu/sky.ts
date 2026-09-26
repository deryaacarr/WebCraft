import { config } from '../config';
import type { Vec3 } from '../core/math';
import type { CelestialState } from '../core/time-of-day';
import { createShaderModule } from './shader';

const DEG = Math.PI / 180;
const TRANSMITTANCE_SIZE: [number, number] = [256, 64];
const MULTISCATTER_SIZE: [number, number] = [32, 32];
const SKYVIEW_SIZE: [number, number] = [192, 108];
// Struct sizes (see atmosphere.wgsl / atmosphere-luts.wgsl / sky.wgsl).
const ATMOSPHERE_SIZE = 112;
/** Koschmieder: visibility (km) = 3.912 / extinction (2 % contrast threshold). */
const KOSCHMIEDER = 3.912;
const LUT_PARAMS_SIZE = 32;
const SKY_PARAMS_SIZE = 112;
const LIGHTING_SIZE = 32;
/** ExposureState in exposure.wgsl: exposure + pad, wb gains, 3 matrix columns. */
export const EXPOSURE_STATE_SIZE = 80;

export interface LightingStats {
  exposureEv: number;
  rawDirectToSky: number;
  directToSky: number;
  wbGains: Vec3;
}
/** Sun elevation (degrees) below which the moon becomes the dominant light. */
const MOON_TAKEOVER_ELEVATION = -4;

export interface SkyState {
  sunDir: Vec3;
  moonDir: Vec3;
  lightDir: Vec3;
  lightIsMoon: boolean;
}

/**
 * Atmosphere LUTs (Hillaire 2020), per-frame sky parameters, the lighting summary buffer
 * and the exposure buffer. Everything is bound at @group(3) (sky.wgsl).
 */
export class SkySystem {
  readonly exposure: GPUBuffer;
  private readonly atmosphere: GPUBuffer;
  private readonly lutParams: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly lighting: GPUBuffer;
  private readonly transmittance: GPUTexture;
  private readonly multiscatter: GPUTexture;
  private readonly skyview: GPUTexture;
  private readonly sampler: GPUSampler;
  private pipelines!: Record<'transmittance' | 'multiscatter' | 'skyview' | 'ambient', GPUComputePipeline>;
  private groups!: Record<'transmittance' | 'multiscatter' | 'skyview' | 'ambientSky' | 'ambientOut', GPUBindGroup>;
  private atmosphereKey = '';
  private readonly bindGroups = new Map<GPUBindGroupLayout, GPUBindGroup>();
  private readonly skyData = new ArrayBuffer(SKY_PARAMS_SIZE);
  private readonly lutData = new ArrayBuffer(LUT_PARAMS_SIZE);
  state: SkyState = { sunDir: [0, 1, 0], moonDir: [0, -1, 0], lightDir: [0, 1, 0], lightIsMoon: false };

  constructor(private readonly device: GPUDevice) {
    const uniform = (label: string, size: number) =>
      device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.atmosphere = uniform('atmosphere', ATMOSPHERE_SIZE);
    this.lutParams = uniform('sky-lut-params', LUT_PARAMS_SIZE);
    this.params = uniform('sky-params', SKY_PARAMS_SIZE);
    this.lighting = device.createBuffer({ label: 'sky-lighting', size: LIGHTING_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.exposure = device.createBuffer({
      label: 'exposure',
      size: EXPOSURE_STATE_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // ExposureState (exposure.wgsl): a daylight exposure, uninitialised white-balance
    // gains (w = 0 → the first frame snaps) and an identity white-balance matrix.
    device.queue.writeBuffer(
      this.exposure,
      0,
      new Float32Array([2, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
    );
    const lut = (label: string, [w, h]: [number, number], layers = 1) =>
      device.createTexture({
        label,
        size: { width: w, height: h, depthOrArrayLayers: layers },
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    this.transmittance = lut('transmittance-lut', TRANSMITTANCE_SIZE);
    this.multiscatter = lut('multiscatter-lut', MULTISCATTER_SIZE);
    this.skyview = lut('skyview-lut', SKYVIEW_SIZE, 2);
    this.sampler = device.createSampler({ label: 'sky', magFilter: 'linear', minFilter: 'linear' });
  }

  async init(): Promise<void> {
    const [luts, ambient] = await Promise.all([
      createShaderModule(this.device, 'atmosphere-luts.wgsl'),
      createShaderModule(this.device, 'sky-ambient.wgsl'),
    ]);
    const constants = { WORKGROUP_SIZE: config.render.workgroupSize };
    const make = (module: GPUShaderModule, entryPoint: string, withConstants = true) =>
      this.device.createComputePipelineAsync({
        label: entryPoint,
        layout: 'auto',
        compute: { module, entryPoint, ...(withConstants && { constants }) },
      });
    const [transmittance, multiscatter, skyview, ambientPipeline] = await Promise.all([
      make(luts, 'transmittance_lut'),
      make(luts, 'multiscatter_lut'),
      make(luts, 'skyview_lut'),
      make(ambient, 'main', false),
    ]);
    this.pipelines = { transmittance, multiscatter, skyview, ambient: ambientPipeline };

    const g = (pipeline: GPUComputePipeline, group: number, entries: [number, GPUBindingResource][]) =>
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(group),
        entries: entries.map(([binding, resource]) => ({ binding, resource })),
      });
    this.groups = {
      transmittance: g(transmittance, 0, [
        [0, { buffer: this.atmosphere }],
        [5, this.transmittance.createView()],
      ]),
      multiscatter: g(multiscatter, 0, [
        [0, { buffer: this.atmosphere }],
        [2, this.transmittance.createView()],
        [4, this.sampler],
        [6, this.multiscatter.createView()],
      ]),
      skyview: g(skyview, 0, [
        [0, { buffer: this.atmosphere }],
        [1, { buffer: this.lutParams }],
        [2, this.transmittance.createView()],
        [3, this.multiscatter.createView()],
        [4, this.sampler],
        [7, this.skyview.createView({ dimension: '2d-array' })],
      ]),
      ambientOut: g(ambientPipeline, 0, [[0, { buffer: this.lighting }]]),
      ambientSky: this.bindGroup(ambientPipeline.getBindGroupLayout(3), [0, 1, 2, 3, 4]),
    };
  }

  /** Updates light directions and parameters from the clock; call once per frame. */
  update(celestial: CelestialState, cameraY: number): void {
    const s = config.sky;
    const sunElevation = Math.asin(celestial.sunDir[1]) / DEG;
    const lightIsMoon = sunElevation < MOON_TAKEOVER_ELEVATION;
    const lightDir = lightIsMoon ? celestial.moonDir : celestial.sunDir;
    this.state = { sunDir: celestial.sunDir, moonDir: celestial.moonDir, lightDir, lightIsMoon };

    const a = s.atmosphere;
    const viewerR = a.bottomRadius + s.seaLevelAltitudeKm + (cameraY - config.terrain.seaLevel) / 1000;
    // Moon illuminance ≈ sun × albedo × disk solid angle / π, scaled by phase and the boost.
    const moonSolidAngle = Math.PI * (s.moonAngularRadius * DEG) ** 2;
    const moonIlluminance = s.sunIlluminance * s.moonAlbedo * (moonSolidAngle / Math.PI) * celestial.moonPhase * s.nightBoost;

    const f32 = new Float32Array(this.skyData);
    f32.set(celestial.sunDir, 0);
    f32[3] = s.sunIlluminance;
    f32.set(celestial.moonDir, 4);
    f32[7] = moonIlluminance;
    f32.set(lightDir, 8);
    f32[11] = (lightIsMoon ? s.moonAngularRadius : s.sunAngularRadius) * DEG;
    // Celestial pole: north (−Z) raised by the latitude.
    f32.set([0, Math.sin(s.latitude * DEG), -Math.cos(s.latitude * DEG)], 12);
    f32[15] = celestial.starRotation;
    f32[16] = s.sunAngularRadius * DEG;
    f32[17] = s.moonAngularRadius * DEG;
    f32[18] = viewerR;
    f32[19] = s.starBrightness;
    f32[20] = s.moonAlbedo;
    f32[21] = lightIsMoon ? 1 : 0;
    // Debug fill only: raise the sky ambient to a direct : sky ceiling (0 = atmosphere as is).
    f32[22] = config.lighting.model === 'debug-fill' ? config.lighting.debugFill.maxDirectToSky : 0;
    this.device.queue.writeBuffer(this.params, 0, this.skyData);

    const lut = new Float32Array(this.lutData);
    lut.set(celestial.sunDir, 0);
    lut[3] = viewerR;
    lut.set(celestial.moonDir, 4);
    this.device.queue.writeBuffer(this.lutParams, 0, this.lutData);
  }

  /** Records LUT updates: transmittance / multi-scattering when the atmosphere changed,
   *  the sky view and the lighting summary every frame. */
  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: 'sky' });
    const w = config.render.workgroupSize;
    const dispatch2d = ([x, y]: [number, number], layers = 1) => pass.dispatchWorkgroups(Math.ceil(x / w), Math.ceil(y / w), layers);
    if (this.writeAtmosphere()) {
      pass.setPipeline(this.pipelines.transmittance);
      pass.setBindGroup(0, this.groups.transmittance);
      dispatch2d(TRANSMITTANCE_SIZE);
      pass.setPipeline(this.pipelines.multiscatter);
      pass.setBindGroup(0, this.groups.multiscatter);
      dispatch2d(MULTISCATTER_SIZE);
    }
    pass.setPipeline(this.pipelines.skyview);
    pass.setBindGroup(0, this.groups.skyview);
    dispatch2d(SKYVIEW_SIZE, 2);
    pass.setPipeline(this.pipelines.ambient);
    pass.setBindGroup(0, this.groups.ambientOut);
    pass.setBindGroup(3, this.groups.ambientSky);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  private statsPending = false;

  /** Reads back exposure and lighting summaries (debug panel); null while a read is in flight. */
  async readStats(): Promise<LightingStats | null> {
    if (this.statsPending) return null;
    this.statsPending = true;
    const read = this.device.createBuffer({ size: LIGHTING_SIZE + EXPOSURE_STATE_SIZE, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.lighting, 0, read, 0, LIGHTING_SIZE);
    encoder.copyBufferToBuffer(this.exposure, 0, read, LIGHTING_SIZE, EXPOSURE_STATE_SIZE);
    this.device.queue.submit([encoder.finish()]);
    try {
      await read.mapAsync(GPUMapMode.READ);
      const f = new Float32Array(read.getMappedRange().slice(0));
      return { rawDirectToSky: f[3]!, directToSky: f[7]!, exposureEv: Math.log2(f[8]!), wbGains: [f[12]!, f[13]!, f[14]!] };
    } finally {
      read.destroy();
      this.statsPending = false;
    }
  }

  /** Multiple-scattering LUT (aerial perspective samples it as well). */
  get multiscatterView(): GPUTextureView {
    return this.multiscatter.createView();
  }

  /** Bind group for @group(3); `bindings` = the sky.wgsl bindings the pipeline uses. */
  bindGroup(layout: GPUBindGroupLayout, bindings: readonly number[] = [0, 1, 2, 3, 4, 5, 6]): GPUBindGroup {
    const cached = this.bindGroups.get(layout);
    if (cached) return cached;
    const resources: GPUBindingResource[] = [
      { buffer: this.params },
      { buffer: this.atmosphere },
      this.transmittance.createView(),
      this.skyview.createView({ dimension: '2d-array' }),
      this.sampler,
      { buffer: this.lighting },
      { buffer: this.exposure },
    ];
    const group = this.device.createBindGroup({
      label: 'sky',
      layout,
      entries: bindings.map((binding) => ({ binding, resource: resources[binding]! })),
    });
    this.bindGroups.set(layout, group);
    return group;
  }

  /** Uploads the atmosphere if its config changed; true when the static LUTs must be rebuilt. */
  private writeAtmosphere(): boolean {
    const a = config.sky.atmosphere;
    const key = JSON.stringify(a);
    if (key === this.atmosphereKey) return false;
    this.atmosphereKey = key;
    const f = new Float32Array(ATMOSPHERE_SIZE / 4);
    f.set(a.rayleighScattering, 0);
    f[3] = a.bottomRadius;
    f.set([a.mieScattering, a.mieScattering, a.mieScattering], 4);
    f[7] = a.topRadius;
    const mieExt = a.mieScattering + a.mieAbsorption;
    f.set([mieExt, mieExt, mieExt], 8);
    f[11] = a.mieG;
    f.set(a.ozoneAbsorption, 12);
    f[15] = a.rayleighScaleHeight;
    f.set(a.groundAlbedo, 16);
    f[19] = a.mieScaleHeight;
    f[20] = a.ozoneCenter;
    f[21] = a.ozoneWidth;
    // Haze layer from the meteorological visibility at the surface (0 = no haze).
    const hazeExtinction = a.hazeVisibilityKm > 0 ? KOSCHMIEDER / a.hazeVisibilityKm : 0;
    f[22] = hazeExtinction * a.hazeAlbedo;
    f[23] = hazeExtinction;
    f[24] = a.hazeScaleHeight;
    this.device.queue.writeBuffer(this.atmosphere, 0, f);
    return true;
  }
}
