import { config } from '../config';
import { BLOCKS, MATERIAL_COLORS, MATERIAL_NAMES, type MaterialName } from '../world/blocks';
import { createShaderModule } from './shader';

/** Shape of public/textures/manifest.json (written by scripts/build-textures.ts). */
export interface TextureManifest {
  version: number;
  kinds: readonly string[];
  layers: { material: string; variant: number; source: string }[];
  materials: Record<string, { firstLayer: number; count: number; rotate: boolean; pom: boolean; alphaTest: boolean }>;
  packs: Record<string, { file: string; bytes: number }>;
}

export interface MaterialStats {
  /** 'ambientCG' when the built packs loaded, 'fallback' for flat colours. */
  source: 'ambientCG' | 'fallback';
  resolution: number;
  layers: number;
  /** All three texture arrays including mip chains. */
  bytes: number;
}

const KIND_COUNT = 3;
const FLAG_ROTATE = 1;
const FLAG_POM = 2;
const FLAG_ALPHA_TEST = 4;
// MaterialParams in material.wgsl: 7 scalars + pad → 32 bytes.
const PARAMS_SIZE = 48;
const TEXTURE_BASE = `${import.meta.env.BASE_URL}textures/`;

/** Layer data for all materials at one resolution: [kind][layer][texel RGBA8]. */
export interface LayerSet {
  resolution: number;
  layers: number;
  pack: Uint8Array;
  materials: { firstLayer: number; count: number; flags: number }[];
  source: MaterialStats['source'];
}

/**
 * Material textures: three rgba8unorm texture_2d_arrays (albedo, normal, specular) with
 * compute-generated mipmaps, the per-material layer table and the per-block face table.
 * Bound at @group(2) of pipelines that include material.wgsl.
 */
export class MaterialSystem {
  private arrays: GPUTexture[] = [];
  private sampler!: GPUSampler;
  private readonly params: GPUBuffer;
  private materialTable!: GPUBuffer;
  private readonly blockTable: GPUBuffer;
  private mipPipeline!: GPUComputePipeline;
  private readonly mipParams: GPUBuffer[] = [];
  private version = 0;
  private readonly bindGroups = new Map<GPUBindGroupLayout, { version: number; group: GPUBindGroup }>();
  private manifest: TextureManifest | null = null;
  private current: MaterialStats = { source: 'fallback', resolution: 0, layers: 0, bytes: 0 };
  private readonly paramData = new ArrayBuffer(PARAMS_SIZE);

  constructor(private readonly device: GPUDevice) {
    this.params = device.createBuffer({ label: 'material-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const faces = new Uint32Array(BLOCKS.length * 4);
    BLOCKS.forEach((b, i) => faces.set([b.materials.top, b.materials.side, b.materials.bottom, 0], i * 4));
    this.blockTable = device.createBuffer({ label: 'block-faces', size: faces.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.blockTable, 0, faces);
  }

  get stats(): MaterialStats {
    return this.current;
  }

  async init(): Promise<void> {
    const module = await createShaderModule(this.device, 'mipgen.wgsl');
    this.mipPipeline = await this.device.createComputePipelineAsync({
      label: 'mipgen',
      layout: 'auto',
      compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: config.render.workgroupSize } },
    });
    for (let kind = 0; kind < KIND_COUNT; kind++) {
      const buf = this.device.createBuffer({ label: `mipgen-kind-${kind}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buf, 0, new Uint32Array([kind, 0, 0, 0]));
      this.mipParams.push(buf);
    }
    try {
      const res = await fetch(`${TEXTURE_BASE}manifest.json`);
      if (res.ok) this.manifest = (await res.json()) as TextureManifest;
    } catch {
      // No packs built: flat-colour fallback below.
    }
    if (!this.manifest) {
      console.warn('[materials] public/textures/manifest.json not found — using flat colours. Run `npm run textures`.');
    }
    this.createSampler();
    await this.load(config.textures.resolution);
  }

  /** Switches texture resolution (reloads the pack; ~instant for cached files). */
  async load(resolution: number): Promise<void> {
    const set = (await this.loadPack(resolution)) ?? fallbackSet();
    this.upload(set);
  }

  /** Re-creates the sampler (anisotropy is a sampler property). */
  createSampler(): void {
    this.sampler = this.device.createSampler({
      label: 'materials',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      maxAnisotropy: config.textures.maxAnisotropy,
    });
    this.version++;
  }

  /** Per-frame parameters (ray-cone spread depends on the camera and resolution). */
  update(pixelSpread: number): void {
    const f32 = new Float32Array(this.paramData);
    const u32 = new Uint32Array(this.paramData);
    const t = config.textures;
    f32[0] = pixelSpread;
    f32[1] = t.lodBias;
    f32[2] = t.pom ? t.pomDepth : 0;
    u32[3] = t.pomSteps;
    f32[4] = t.alphaCutoff;
    f32[5] = this.current.resolution;
    f32[6] = t.pomMaxDistance;
    f32[7] = t.variantRegionScale;
    f32[8] = t.variantWarp;
    this.device.queue.writeBuffer(this.params, 0, this.paramData);
  }

  /** Bind group for @group(2); `bindings` = the material.wgsl bindings the pipeline uses. */
  bindGroup(layout: GPUBindGroupLayout, bindings: readonly number[] = [0, 1, 2, 3, 4, 5, 6]): GPUBindGroup {
    const cached = this.bindGroups.get(layout);
    if (cached && cached.version === this.version) return cached.group;
    const [albedo, normal, specular] = this.arrays;
    if (!albedo || !normal || !specular) throw new Error('materials not loaded');
    const resources: GPUBindingResource[] = [
      { buffer: this.params },
      { buffer: this.materialTable },
      { buffer: this.blockTable },
      albedo.createView({ dimension: '2d-array' }),
      normal.createView({ dimension: '2d-array' }),
      specular.createView({ dimension: '2d-array' }),
      this.sampler,
    ];
    const group = this.device.createBindGroup({
      label: 'materials',
      layout,
      entries: bindings.map((binding) => ({ binding, resource: resources[binding]! })),
    });
    this.bindGroups.set(layout, { version: this.version, group });
    return group;
  }

  private async loadPack(resolution: number): Promise<LayerSet | null> {
    const m = this.manifest;
    const pack = m?.packs[String(resolution)];
    if (!m || !pack) return null;
    try {
      const res = await fetch(`${TEXTURE_BASE}${pack.file}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const bytes = new Uint8Array(await res.arrayBuffer());
      const expected = KIND_COUNT * m.layers.length * resolution * resolution * 4;
      if (bytes.byteLength !== expected) throw new Error(`size ${bytes.byteLength}, expected ${expected}`);
      const materials = MATERIAL_NAMES.map((name) => {
        const info = m.materials[name];
        if (!info) throw new Error(`manifest has no material "${name}"`);
        const flags = (info.rotate ? FLAG_ROTATE : 0) | (info.pom ? FLAG_POM : 0) | (info.alphaTest ? FLAG_ALPHA_TEST : 0);
        return { firstLayer: info.firstLayer, count: info.count, flags };
      });
      return { resolution, layers: m.layers.length, pack: bytes, materials, source: 'ambientCG' };
    } catch (err) {
      console.warn(`[materials] could not load ${pack.file}:`, err);
      return null;
    }
  }

  private upload(set: LayerSet): void {
    for (const t of this.arrays) t.destroy();
    const { resolution: res, layers } = set;
    const mips = Math.log2(res) + 1;
    const layerBytes = res * res * 4;
    this.arrays = ['albedo', 'normal', 'specular'].map((kind, k) => {
      const tex = this.device.createTexture({
        label: `material-${kind}`,
        size: { width: res, height: res, depthOrArrayLayers: layers },
        mipLevelCount: mips,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.device.queue.writeTexture(
        { texture: tex },
        set.pack.subarray(k * layers * layerBytes, (k + 1) * layers * layerBytes),
        { bytesPerRow: res * 4, rowsPerImage: res },
        { width: res, height: res, depthOrArrayLayers: layers },
      );
      return tex;
    });
    this.generateMips(mips, layers);

    const table = new Uint32Array(MATERIAL_NAMES.length * 4);
    set.materials.forEach((m, i) => table.set([m.firstLayer, m.count, m.flags, 0], i * 4));
    this.materialTable?.destroy();
    this.materialTable = this.device.createBuffer({ label: 'material-table', size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.materialTable, 0, table);

    // Full mip chain ≈ 4/3 of the base level.
    let bytes = 0;
    for (let l = 0; l < mips; l++) bytes += KIND_COUNT * layers * Math.max(1, res >> l) ** 2 * 4;
    this.current = { source: set.source, resolution: res, layers, bytes };
    this.version++;
    console.info(`[materials] ${set.source} ${res}px, ${layers} layers, ${(bytes / 2 ** 20).toFixed(1)} MB`);
  }

  private generateMips(mips: number, layers: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'mipgen' });
    const pass = encoder.beginComputePass({ label: 'mipgen' });
    pass.setPipeline(this.mipPipeline);
    const w = config.render.workgroupSize;
    this.arrays.forEach((tex, kind) => {
      for (let level = 1; level < mips; level++) {
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: this.mipPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.mipParams[kind]! } },
              { binding: 1, resource: tex.createView({ dimension: '2d-array', baseMipLevel: level - 1, mipLevelCount: 1 }) },
              { binding: 2, resource: tex.createView({ dimension: '2d-array', baseMipLevel: level, mipLevelCount: 1 }) },
            ],
          }),
        );
        const size = Math.max(1, tex.width >> level);
        pass.dispatchWorkgroups(Math.ceil(size / w), Math.ceil(size / w), layers);
      }
    });
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

/** One flat-colour layer per material (no texture packs built). Exported for tests. */
export function fallbackSet(): LayerSet {
  const res = 4;
  const layers = MATERIAL_NAMES.length;
  const texels = res * res;
  const pack = new Uint8Array(KIND_COUNT * layers * texels * 4);
  const emissive: Partial<Record<MaterialName, number>> = { torch: 200, lava: 150 };
  MATERIAL_NAMES.forEach((name, l) => {
    const c = MATERIAL_COLORS[name];
    for (let i = 0; i < texels; i++) {
      pack.set([(c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff, 255], (l * texels + i) * 4);
      pack.set([128, 128, 255, 128], ((layers + l) * texels + i) * 4); // flat normal, AO 1
      pack.set([27, 10, 0, emissive[name] ?? 255], ((2 * layers + l) * texels + i) * 4); // roughness ≈ 0.8
    }
  });
  return {
    resolution: res,
    layers,
    pack,
    materials: MATERIAL_NAMES.map((_, i) => ({ firstLayer: i, count: 1, flags: 0 })),
    source: 'fallback',
  };
}
