/**
 * Builds the material texture packs from assets/raw (see fetch-textures.ts):
 *
 *   npm run textures:build
 *
 * For every material variant in texture-spec.ts it produces three RGBA8 layers at a
 * 512 px working size, then box-downsamples them to every resolution in
 * TEXTURE_RESOLUTIONS and writes public/textures/pack-<res>.bin plus manifest.json.
 * Mipmaps are generated on the GPU at load time.
 *
 * Channel layout (LabPBR-like):
 *   albedo    R,G,B = sRGB colour           A = opacity (alpha test; 255 = opaque)
 *   normal    R,G   = tangent-space normal XY (OpenGL, +Y up), B = AO, A = height
 *   specular  R = perceptual smoothness (1 − √roughness)
 *             G = F0: 10 (≈ 0.04, dielectric) or 255 (metal)
 *             B = subsurface scattering amount (leaves)
 *             A = emission 0‥254, 255 = none
 */
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';
import {
  createImage,
  fromBytes,
  labPbr,
  normalsFromHeight,
  rng,
  seamBlend,
  stamp,
  tileableNoise,
  type FloatImage,
  type Sprite,
} from './lib/image-ops.ts';
import { config } from '../src/config.ts';
import {
  TEXTURE_RESOLUTIONS,
  TEXTURE_SPEC,
  type AmbientCgSource,
  type GrassSideSource,
  type LeavesSource,
  type ProceduralSource,
  type VariantSource,
} from './texture-spec.ts';

const ROOT = new URL('..', import.meta.url).pathname;
const RAW = join(ROOT, 'assets/raw');
const OUT = join(ROOT, 'public/textures');
const WORK = 512;
const SOURCE = 1024;
const KINDS = ['albedo', 'normal', 'specular'] as const;

/** A material layer at working resolution; every map is WORK². */
interface Layer {
  color: FloatImage; // 4: sRGB + opacity
  normal: FloatImage; // 3: vector
  ao: FloatImage;
  height: FloatImage;
  roughness: FloatImage;
  metal: FloatImage;
  emission: FloatImage;
  sss: FloatImage;
}

function blankLayer(): Layer {
  const n = createImage(WORK, WORK, 3);
  for (let i = 0; i < WORK * WORK; i++) n.data[i * 3 + 2] = 1;
  return {
    color: createImage(WORK, WORK, 4, 1),
    normal: n,
    ao: createImage(WORK, WORK, 1, 1),
    height: createImage(WORK, WORK, 1, 0.5),
    roughness: createImage(WORK, WORK, 1, 0.8),
    metal: createImage(WORK, WORK, 1, 0),
    emission: createImage(WORK, WORK, 1, 0),
    sss: createImage(WORK, WORK, 1, 0),
  };
}

// ---------------------------------------------------------------------------- loading

async function loadMap(id: string, map: string, channels: 1 | 3, region: { size: number; out: number }): Promise<FloatImage | null> {
  const file = join(RAW, id, `${map}.jpg`);
  if (!existsSync(file)) return null;
  let img = sharp(file).extract({ left: 0, top: 0, width: region.size, height: region.size });
  img = channels === 1 ? img.greyscale() : img.removeAlpha();
  const { data, info } = await img
    .resize(region.out, region.out, { kernel: 'lanczos3' })
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.channels !== channels) throw new Error(`${id}/${map}: expected ${channels} channels, got ${info.channels}`);
  return fromBytes(new Uint8Array(data), info.width, info.height, channels);
}

function decodeNormal(rgb: FloatImage): FloatImage {
  const out = createImage(rgb.width, rgb.height, 3);
  for (let i = 0; i < rgb.width * rgb.height; i++) {
    const x = rgb.data[i * 3]! * 2 - 1;
    const y = rgb.data[i * 3 + 1]! * 2 - 1;
    const z = rgb.data[i * 3 + 2]! * 2 - 1;
    const len = Math.hypot(x, y, z) || 1;
    out.data.set([x / len, y / len, z / len], i * 3);
  }
  return out;
}

function luminance(rgb: FloatImage): FloatImage {
  const out = createImage(rgb.width, rgb.height, 1);
  for (let i = 0; i < rgb.width * rgb.height; i++) {
    out.data[i] = 0.2126 * rgb.data[i * 3]! + 0.7152 * rgb.data[i * 3 + 1]! + 0.0722 * rgb.data[i * 3 + 2]!;
  }
  return out;
}

async function loadAmbientCg(src: AmbientCgSource | { id: string; crop: number }): Promise<Layer> {
  const region = { size: Math.round(SOURCE * src.crop), out: WORK };
  const retile = src.crop < 1;
  const tile = (img: FloatImage | null, isNormal = false) => (img && retile ? seamBlend(img, 0.25, isNormal) : img);
  const layer = blankLayer();

  const color = tile(await loadMap(src.id, 'Color', 3, region));
  if (!color) throw new Error(`${src.id}: missing Color map (run npm run textures:fetch)`);
  const opacity = tile(await loadMap(src.id, 'Opacity', 1, region));
  for (let i = 0; i < WORK * WORK; i++) {
    layer.color.data.set([color.data[i * 3]!, color.data[i * 3 + 1]!, color.data[i * 3 + 2]!, opacity ? opacity.data[i]! : 1], i * 4);
  }
  const normal = await loadMap(src.id, 'NormalGL', 3, region);
  if (normal) layer.normal = tile(decodeNormal(normal), true)!;
  layer.ao = tile(await loadMap(src.id, 'AmbientOcclusion', 1, region)) ?? layer.ao;
  layer.height = tile(await loadMap(src.id, 'Displacement', 1, region)) ?? layer.height;
  layer.roughness = tile(await loadMap(src.id, 'Roughness', 1, region)) ?? layer.roughness;
  layer.metal = tile(await loadMap(src.id, 'Metalness', 1, region)) ?? layer.metal;
  const emission = await loadMap(src.id, 'Emission', 3, region);
  if (emission) layer.emission = tile(luminance(emission))!;
  return layer;
}

// ------------------------------------------------------------------------- composites

/** Dirt with a band of grass hanging down from the top edge, jagged by tileable noise. */
async function buildGrassSide(src: GrassSideSource): Promise<Layer> {
  const dirt = await loadAmbientCg({ id: src.dirt, crop: 0.7 });
  const grass = await loadAmbientCg({ id: src.grass, crop: 0.7 });
  const noise = tileableNoise(WORK, 16, 3, src.seed);
  const out = dirt;
  const soft = 0.015 * WORK;
  for (let y = 0; y < WORK; y++) {
    for (let x = 0; x < WORK; x++) {
      // Grass depth along the top: ~18 % of the face, ±10 % jagged; noise row 0 is tileable in x.
      const edge = (0.18 + 0.2 * (noise.data[x]! - 0.5)) * WORK;
      const m = Math.min(1, Math.max(0, (edge - y) / soft + 0.5));
      if (m <= 0) continue;
      const i = y * WORK + x;
      const lerp = (a: FloatImage, b: FloatImage, c: number) => {
        for (let k = 0; k < c; k++) a.data[i * c + k] = a.data[i * c + k]! + (b.data[i * c + k]! - a.data[i * c + k]!) * m;
      };
      lerp(out.color, grass.color, 4);
      lerp(out.normal, grass.normal, 3);
      lerp(out.ao, grass.ao, 1);
      lerp(out.roughness, grass.roughness, 1);
      // The grass overhang sits a little proud of the dirt.
      out.height.data[i] = out.height.data[i]! + (Math.min(1, grass.height.data[i]! + 0.15) - out.height.data[i]!) * m;
    }
  }
  return out;
}

/** Splits a two-leaf ambientCG sheet into sprites cropped to their opaque area. */
async function leafSprites(id: string): Promise<Sprite[]> {
  const size = 256;
  const region = { size: SOURCE, out: size };
  const color = await loadMap(id, 'Color', 3, region);
  const opacity = await loadMap(id, 'Opacity', 1, region);
  const normal = await loadMap(id, 'NormalGL', 3, region);
  const rough = await loadMap(id, 'Roughness', 1, region);
  if (!color || !opacity || !normal) throw new Error(`${id}: leaf sprites need Color, Opacity and NormalGL`);
  const n = decodeNormal(normal);
  const sprites: Sprite[] = [];
  for (const x0 of [0, size / 2]) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -1;
    let maxY = -1;
    for (let y = 0; y < size; y++) {
      for (let x = x0; x < x0 + size / 2; x++) {
        if (opacity.data[y * size + x]! > 0.05) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }
    if (maxX < 0) continue;
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const s: Sprite = { color: createImage(w, h, 4), normal: createImage(w, h, 3), roughness: createImage(w, h, 1) };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const si = (minY + y) * size + minX + x;
        const di = y * w + x;
        s.color.data.set([color.data[si * 3]!, color.data[si * 3 + 1]!, color.data[si * 3 + 2]!, opacity.data[si]!], di * 4);
        s.normal.data.set([n.data[si * 3]!, n.data[si * 3 + 1]!, n.data[si * 3 + 2]!], di * 3);
        s.roughness.data[di] = rough ? rough.data[si]! : 0.6;
      }
    }
    sprites.push(s);
  }
  return sprites;
}

/** Tileable canopy of leaves: many rotated leaf sprites, later ones on top. */
async function buildLeaves(src: LeavesSource): Promise<Layer> {
  const [primary, ...others] = await Promise.all(src.sprites.map(leafSprites));
  if (!primary?.length) throw new Error('no leaf sprites');
  const secondary = others.flat();
  const layer = blankLayer();
  for (let i = 0; i < WORK * WORK; i++) layer.color.data[i * 4 + 3] = 0;
  layer.ao.data.fill(0.4);
  layer.height.data.fill(0);
  layer.roughness.data.fill(0.6);
  const r = rng(src.seed);
  const count = 180;
  for (let i = 0; i < count; i++) {
    // Mostly the first (main) leaf type, a few of the others for variety.
    const pool = secondary.length && r() < 0.2 ? secondary : primary;
    const sprite = pool[Math.floor(r() * pool.length)]!;
    const tint = 0.72 + 0.3 * r();
    const level = 0.35 + (0.65 * i) / count;
    stamp(layer, sprite, r() * WORK, r() * WORK, WORK * (0.12 + 0.06 * r()), r() * Math.PI * 2, [tint * (0.95 + 0.1 * r()), tint, tint * 0.9], level);
  }
  // Fill transparent pixels with the average leaf colour so mip levels do not bleed dark.
  const mean = [0, 0, 0];
  let covered = 0;
  for (let i = 0; i < WORK * WORK; i++) {
    const a = layer.color.data[i * 4 + 3]!;
    if (a < 0.5) continue;
    for (let k = 0; k < 3; k++) mean[k]! += layer.color.data[i * 4 + k]!;
    covered++;
  }
  for (let i = 0; i < WORK * WORK; i++) {
    const a = layer.color.data[i * 4 + 3]!;
    for (let k = 0; k < 3; k++) {
      const c = layer.color.data[i * 4 + k]!;
      layer.color.data[i * 4 + k] = c * a + (mean[k]! / covered) * (1 - a);
    }
  }
  layer.sss.data.fill(0.8);
  return layer;
}

// ------------------------------------------------------------------------ procedural

function buildProcedural(src: ProceduralSource): Layer {
  const layer = blankLayer();
  const N = WORK * WORK;
  const set = (i: number, rgb: [number, number, number]) => layer.color.data.set([...rgb, 1], i * 4);
  switch (src.generator) {
    case 'water': {
      const h = tileableNoise(WORK, 6, 4, src.seed);
      layer.height = h;
      layer.normal = normalsFromHeight(h, 3);
      for (let i = 0; i < N; i++) {
        const v = (h.data[i]! - 0.5) * 0.06;
        set(i, [0.16 + v, 0.35 + v, 0.66 + v]);
      }
      layer.roughness.data.fill(0.06);
      break;
    }
    case 'glass': {
      const n = tileableNoise(WORK, 8, 3, src.seed);
      const frame = 0.035 * WORK;
      for (let y = 0; y < WORK; y++) {
        for (let x = 0; x < WORK; x++) {
          const i = y * WORK + x;
          const edge = Math.min(x, y, WORK - 1 - x, WORK - 1 - y) < frame;
          const v = 0.9 + (n.data[i]! - 0.5) * 0.04 - (edge ? 0.2 : 0);
          set(i, [v * 0.92, v * 0.97, v]);
        }
      }
      layer.normal = normalsFromHeight(n, 0.5);
      layer.roughness.data.fill(0.04);
      break;
    }
    case 'torch': {
      const n = tileableNoise(WORK, 8, 3, src.seed);
      for (let y = 0; y < WORK; y++) {
        for (let x = 0; x < WORK; x++) {
          const i = y * WORK + x;
          const flame = Math.max(0, 1 - Math.hypot((x - WORK / 2) / (WORK * 0.3), (y - WORK * 0.35) / (WORK * 0.35)) + (n.data[i]! - 0.5) * 0.4);
          const f = Math.min(1, flame * 1.5);
          set(i, [0.45 + 0.55 * f, 0.28 + 0.55 * f, 0.12 + 0.2 * f * f]);
          layer.emission.data[i] = f;
        }
      }
      layer.roughness.data.fill(0.7);
      break;
    }
    case 'log-top': {
      const n = tileableNoise(WORK, 8, 3, src.seed);
      const r = rng(src.seed);
      const rings = 14 + Math.floor(r() * 6);
      const cx = WORK * (0.47 + 0.06 * r());
      const cy = WORK * (0.47 + 0.06 * r());
      for (let y = 0; y < WORK; y++) {
        for (let x = 0; x < WORK; x++) {
          const i = y * WORK + x;
          const d = Math.hypot(x - cx, y - cy) / (WORK / 2) + (n.data[i]! - 0.5) * 0.12;
          const bark = Math.max(Math.abs(x - WORK / 2), Math.abs(y - WORK / 2)) / (WORK / 2) > 0.94 - n.data[i]! * 0.05;
          // Thin dark latewood lines on lighter earlywood, with grain noise.
          const ring = Math.pow(0.5 + 0.5 * Math.cos(d * rings * Math.PI * 2), 6);
          const grain = (n.data[(i * 7) % (WORK * WORK)]! - 0.5) * 0.05;
          const base: [number, number, number] = bark
            ? [0.3 + grain, 0.22 + grain, 0.14 + grain]
            : [0.7 - 0.12 * ring + grain, 0.53 - 0.11 * ring + grain, 0.33 - 0.08 * ring + grain];
          set(i, base);
          layer.height.data[i] = bark ? 0.8 : 0.45 + 0.1 * ring;
        }
      }
      layer.normal = normalsFromHeight(layer.height, 2);
      layer.roughness.data.fill(0.75);
      break;
    }
  }
  return layer;
}

// --------------------------------------------------------------------------- packing

async function buildVariant(v: VariantSource): Promise<Layer> {
  switch (v.kind) {
    case 'ambientcg':
      return loadAmbientCg(v);
    case 'grass-side':
      return buildGrassSide(v);
    case 'leaves':
      return buildLeaves(v);
    case 'procedural':
      return buildProcedural(v);
  }
}

/**
 * Pulls the variants of one material towards a common average colour (multiplicatively,
 * in linear light, alpha-weighted), so neighbouring voxels using different variants do
 * not look patchy. Detail and contrast inside each texture are kept.
 */
function matchTones(variants: Layer[], strength = 0.85): void {
  if (variants.length < 2) return;
  const toLin = toLinear;
  const means = variants.map((v) => {
    const m = [0, 0, 0];
    let w = 0;
    for (let i = 0; i < WORK * WORK; i++) {
      const a = v.color.data[i * 4 + 3]!;
      for (let k = 0; k < 3; k++) m[k]! += toLin(v.color.data[i * 4 + k]!) * a;
      w += a;
    }
    return m.map((x) => x / w);
  });
  const target = [0, 1, 2].map((k) => means.reduce((s, m) => s + m[k]!, 0) / means.length);
  variants.forEach((v, j) => {
    const gain = [0, 1, 2].map((k) => (target[k]! / means[j]![k]!) ** strength);
    for (let i = 0; i < WORK * WORK; i++) {
      for (let k = 0; k < 3; k++) {
        v.color.data[i * 4 + k] = toSrgb(Math.min(1, toLin(v.color.data[i * 4 + k]!) * gain[k]!));
      }
    }
  });
}

const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);

/** Opacity-weighted mean linear luminance of a set of variants. */
function meanAlbedo(variants: Layer[]): number {
  let sum = 0;
  let weight = 0;
  for (const v of variants) {
    for (let i = 0; i < WORK * WORK; i++) {
      const a = v.color.data[i * 4 + 3]!;
      const c = v.color.data;
      sum += a * (0.2126 * toLinear(c[i * 4]!) + 0.7152 * toLinear(c[i * 4 + 1]!) + 0.0722 * toLinear(c[i * 4 + 2]!));
      weight += a;
    }
  }
  return sum / Math.max(weight, 1e-6);
}

/**
 * Scales linear RGB so the material's mean luminance matches its physical target
 * (config.textures.albedoTargets). Two passes, since clamping at 1 can undershoot.
 */
function calibrateAlbedo(variants: Layer[], target: number): void {
  for (let pass = 0; pass < 2; pass++) {
    const gain = target / Math.max(meanAlbedo(variants), 1e-6);
    for (const v of variants) {
      for (let i = 0; i < WORK * WORK; i++) {
        for (let k = 0; k < 3; k++) v.color.data[i * 4 + k] = toSrgb(Math.min(1, toLinear(v.color.data[i * 4 + k]!) * gain));
      }
    }
  }
}

/** The three RGBA8 layers at working resolution. */
function encode(layer: Layer): Record<(typeof KINDS)[number], Uint8Array> {
  const N = WORK * WORK;
  const albedo = new Uint8Array(N * 4);
  const normal = new Uint8Array(N * 4);
  const specular = new Uint8Array(N * 4);
  const u8 = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 4; k++) albedo[i * 4 + k] = u8(layer.color.data[i * 4 + k]!);
    normal[i * 4] = u8(layer.normal.data[i * 3]! * 0.5 + 0.5);
    normal[i * 4 + 1] = u8(layer.normal.data[i * 3 + 1]! * 0.5 + 0.5);
    normal[i * 4 + 2] = u8(layer.ao.data[i]!);
    normal[i * 4 + 3] = u8(layer.height.data[i]!);
    specular[i * 4] = u8(labPbr.smoothness(layer.roughness.data[i]!));
    specular[i * 4 + 1] = labPbr.f0(layer.metal.data[i]!);
    specular[i * 4 + 2] = u8(layer.sss.data[i]!);
    specular[i * 4 + 3] = labPbr.emission(layer.emission.data[i]!);
  }
  return { albedo, normal, specular };
}

/**
 * Box-downsamples RGBA8 by an integer factor. `alphaWeighted`: colour is averaged
 * weighted by alpha (albedo), so transparent texels do not tint their neighbours.
 */
function downsample(src: Uint8Array, size: number, factor: number, alphaWeighted: boolean): Uint8Array {
  const out = size / factor;
  const dst = new Uint8Array(out * out * 4);
  for (let y = 0; y < out; y++) {
    for (let x = 0; x < out; x++) {
      const sum = [0, 0, 0, 0];
      let weight = 0;
      for (let j = 0; j < factor; j++) {
        for (let i = 0; i < factor; i++) {
          const s = ((y * factor + j) * size + x * factor + i) * 4;
          const w = alphaWeighted ? src[s + 3]! / 255 + 1e-3 : 1;
          for (let k = 0; k < 3; k++) sum[k]! += src[s + k]! * w;
          sum[3]! += src[s + 3]!;
          weight += w;
        }
      }
      const d = (y * out + x) * 4;
      for (let k = 0; k < 3; k++) dst[d + k] = Math.round(sum[k]! / weight);
      dst[d + 3] = Math.round(sum[3]! / (factor * factor));
    }
  }
  return dst;
}

async function main(): Promise<void> {
  const t0 = performance.now();
  const layers: { material: string; variant: number; source: string; data: ReturnType<typeof encode> }[] = [];
  const materials: Record<string, { firstLayer: number; count: number; rotate: boolean; pom: boolean; alphaTest: boolean }> = {};
  const albedo: Record<string, number> = {};
  for (const [material, spec] of Object.entries(TEXTURE_SPEC)) {
    materials[material] = { firstLayer: layers.length, count: spec.variants.length, rotate: spec.rotate, pom: spec.pom, alphaTest: spec.alphaTest };
    const built: Layer[] = [];
    const labels: string[] = [];
    for (const v of spec.variants) {
      labels.push(v.kind === 'ambientcg' ? v.id : v.kind === 'procedural' ? `procedural:${v.generator}` : v.kind);
      built.push(await buildVariant(v));
    }
    matchTones(built);
    const target = config.textures.albedoTargets[material];
    const before = meanAlbedo(built);
    if (target != null) calibrateAlbedo(built, target);
    const after = meanAlbedo(built);
    albedo[material] = after;
    built.forEach((layer, i) => layers.push({ material, variant: i, source: labels[i]!, data: encode(layer) }));
    const note = target != null ? `albedo ${before.toFixed(3)} → ${after.toFixed(3)} (target ${target})` : `albedo ${after.toFixed(3)} (kept)`;
    console.log(`  ${material.padEnd(13)} ${note}  [${labels.join(', ')}]`);
  }

  mkdirSync(OUT, { recursive: true });
  const packs: Record<number, { file: string; bytes: number }> = {};
  for (const res of TEXTURE_RESOLUTIONS) {
    const layerBytes = res * res * 4;
    const pack = new Uint8Array(KINDS.length * layers.length * layerBytes);
    KINDS.forEach((kind, k) => {
      layers.forEach((layer, l) => {
        const scaled = downsample(layer.data[kind], WORK, WORK / res, kind === 'albedo');
        pack.set(scaled, (k * layers.length + l) * layerBytes);
      });
    });
    const file = `pack-${res}.bin`;
    writeFileSync(join(OUT, file), pack);
    packs[res] = { file, bytes: pack.byteLength };
  }

  const manifest = {
    version: 1,
    kinds: KINDS,
    layers: layers.map(({ material, variant, source }) => ({ material, variant, source })),
    materials,
    /** Mean linear albedo per material after calibration (opacity-weighted). */
    albedo,
    packs,
  };
  writeFileSync(join(OUT, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);

  // Contact sheet of every layer's albedo at 128 px, for a quick visual check.
  const cols = 8;
  const tile = 128;
  const rows = Math.ceil(layers.length / cols);
  const sheet = sharp({ create: { width: cols * tile, height: rows * tile, channels: 4, background: '#202020' } });
  const composites = layers.map((layer, i) => ({
    input: Buffer.from(downsample(layer.data.albedo, WORK, WORK / tile, true)),
    raw: { width: tile, height: tile, channels: 4 as const },
    left: (i % cols) * tile,
    top: Math.floor(i / cols) * tile,
  }));
  await sheet.composite(composites).png().toFile(join(OUT, 'preview.png'));

  const mb = (b: number) => (b / 2 ** 20).toFixed(1);
  console.log(`${layers.length} layers → ${TEXTURE_RESOLUTIONS.map((r) => `${r}px ${mb(packs[r]!.bytes)} MB`).join(', ')}`);
  console.log(`done in ${((performance.now() - t0) / 1000).toFixed(1)} s → ${OUT}`);
}

await main();
