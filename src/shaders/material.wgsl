// Material sampling for voxel faces: face UVs and tangent frames, per-voxel texture
// variants and 90° rotations, ray-cone texture filtering, parallax occlusion mapping and
// decoding of the LabPBR-like channel layout (see scripts/build-textures.ts):
//   albedo    rgb = sRGB colour, a = opacity
//   normal    rg = tangent-space normal XY (OpenGL, +Y up), b = AO, a = height
//   specular  r = perceptual smoothness, g = F0 (≥ 230/255 = metal), b = subsurface,
//             a = emission (255 = none)
//
// Bind the materials (MaterialSystem.bindGroup) at @group(2).

struct MaterialParams {
  /// Ray-cone spread: footprint width per unit distance of one pixel (primary rays).
  pixel_spread: f32,
  /// Added to the texture LOD (positive = blurrier).
  lod_bias: f32,
  /// Parallax occlusion depth in blocks (0 = off) and march steps.
  pom_depth: f32,
  pom_steps: u32,
  alpha_cutoff: f32,
  /// Texture edge in texels (for scalar LOD estimates).
  texture_size: f32,
  /// Parallax is skipped beyond this distance (sub-pixel there anyway).
  pom_max_distance: f32,
  /// Variant regions: world-space noise feature size (blocks; 0 = random per block) and
  /// domain warp (in feature sizes).
  variant_scale: f32,
  variant_warp: f32,
  /// World-space layers: edge (m) of the area one layer covers, and on/off.
  world_span: f32,
  world_enabled: u32,
  /// Detail (config.detail; material-detail.wgsl). Colours are three scalars each.
  hex_density: f32,
  hex_contrast: f32,
  variant_blend: f32,
  variant_edge_noise: f32,
  macro_scale: f32,
  macro_strength: f32,
  slope_strength: f32,
  moss: f32,
  foliage_strength: f32,
  foliage_scale: f32,
  warm_r: f32, warm_g: f32, warm_b: f32,
  cool_r: f32, cool_g: f32, cool_b: f32,
  dust_r: f32, dust_g: f32, dust_b: f32,
  moss_r: f32, moss_g: f32, moss_b: f32,
  dry_r: f32, dry_g: f32, dry_b: f32,
  lush_r: f32, lush_g: f32, lush_b: f32,
  /// Distance LOD (blocks) of hex tiling and moss.
  hex_max_distance: f32,
  moss_max_distance: f32,
  height_blend_depth: f32,
  _pad3: f32,
  _pad4: f32,
};

/// Per material: first texture layer, number of variants, flags.
struct MaterialInfo {
  first_layer: u32,
  variants: u32,
  flags: u32,
  /// Mean opacity of the albedo texels (foliage coverage, for GI rays).
  coverage: f32,
};

const MATERIAL_ROTATE: u32 = 1u;
const MATERIAL_POM: u32 = 2u;
const MATERIAL_ALPHA_TEST: u32 = 4u;
/// Natural material with world-space layers; their first index is flags >> 8.
const MATERIAL_WORLD: u32 = 8u;
const MATERIAL_WORLD_FIRST_SHIFT: u32 = 8u;
/// Detail categories: natural rock / soil, foliage, moisture source (moss nearby).
const MATERIAL_NATURAL: u32 = 16u;
const MATERIAL_FOLIAGE: u32 = 32u;
const MATERIAL_MOIST: u32 = 64u;
const F0_METAL: f32 = 230.0 / 255.0;

@group(2) @binding(0) var<uniform> material_params: MaterialParams;
@group(2) @binding(1) var<storage, read> materials: array<MaterialInfo>;
/// Per block id: face material indices (top, side, bottom, unused).
@group(2) @binding(2) var<storage, read> block_faces: array<vec4u>;
@group(2) @binding(3) var tex_albedo: texture_2d_array<f32>;
@group(2) @binding(4) var tex_normal: texture_2d_array<f32>;
@group(2) @binding(5) var tex_specular: texture_2d_array<f32>;
@group(2) @binding(6) var tex_sampler: sampler;
/// World-space layers of natural materials (world_span × world_span m each).
@group(2) @binding(7) var tex_world_albedo: texture_2d_array<f32>;
@group(2) @binding(8) var tex_world_normal: texture_2d_array<f32>;
@group(2) @binding(9) var tex_world_specular: texture_2d_array<f32>;
/// Smooth tileable value noise (materials.ts detailNoiseVolume): 64³ texels, a lattice
/// cell every 4 texels, period 16 cells.
@group(2) @binding(10) var detail_noise: texture_3d<f32>;
const NOISE_PERIOD: f32 = 16.0;

/// Tangent frame of a voxel face: T = +u direction, B = "image up" (−v). Chosen so no
/// face is mirrored (T × B = N) and side faces have image-up = world-up.
struct FaceFrame {
  t: vec3f,
  b: vec3f,
  n: vec3f,
};

fn faceFrame(face: u32) -> FaceFrame {
  switch (face) {
    case 0u: { return FaceFrame(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0)); }
    case 1u: { return FaceFrame(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), vec3f(-1.0, 0.0, 0.0)); }
    case 2u: { return FaceFrame(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0)); }
    case 3u: { return FaceFrame(vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, -1.0, 0.0)); }
    case 4u: { return FaceFrame(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0)); }
    default: { return FaceFrame(vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, -1.0)); }
  }
}

fn faceMaterial(block_id: u32, face: u32) -> u32 {
  let f = block_faces[block_id];
  return select(select(f.y, f.z, face == 3u), f.x, face == 2u);
}

fn hashCellFace(c: vec3i, face: u32) -> u32 {
  var h = (u32(c.x) * 0x8da6b343u) ^ (u32(c.y) * 0xd8163841u) ^ (u32(c.z) * 0xcb1ab31fu) ^ (face * 0x165667b1u);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

fn hash3(c: vec3i) -> f32 {
  return f32(hashCellFace(c, 7u) >> 8u) / 16777216.0;
}

/// Smooth 3D value noise in [0, 1] from the baked volume: one trilinear lookup (the
/// hashed valueNoise below costs eight hashes). Repeats every 16 units of `p`.
fn fastNoise(p: vec3f) -> f32 {
  return textureSampleLevel(detail_noise, tex_sampler, p / NOISE_PERIOD, 0.0).r;
}

/// Smooth 3D value noise in [0, 1].
fn valueNoise(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let x00 = mix(hash3(i), hash3(i + vec3i(1, 0, 0)), u.x);
  let x10 = mix(hash3(i + vec3i(0, 1, 0)), hash3(i + vec3i(1, 1, 0)), u.x);
  let x01 = mix(hash3(i + vec3i(0, 0, 1)), hash3(i + vec3i(1, 0, 1)), u.x);
  let x11 = mix(hash3(i + vec3i(0, 1, 1)), hash3(i + vec3i(1, 1, 1)), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

/// Variant index from a low-frequency, domain-warped world-space noise sampled at the
/// block centre: equal variants cluster into regions and veins (all faces of a block
/// agree). Each material gets its own pattern. Not used for alpha-tested materials: their
/// mapping is evaluated inside ray traversal (alpha tests), where it must stay cheap.
fn regionVariant(cell: vec3i, material: u32, count: u32) -> u32 {
  let p = (vec3f(cell) + 0.5) / material_params.variant_scale + f32(material) * 17.31;
  // One warp sample bent along a fixed skew axis: cheap, still turns blobs into veins.
  let warp = valueNoise(p * 0.5 + 31.7) - 0.5;
  let q = p + vec3f(0.8, 0.5, -0.6) * (warp * 2.0 * material_params.variant_warp);
  let n = valueNoise(q) * 0.7 + valueNoise(q * 2.3 + 5.1) * 0.3;
  // Value noise clusters around 0.5: stretch so every variant gets a fair share.
  let v = clamp((n - 0.5) * 2.2 + 0.5, 0.0, 0.9999);
  return u32(v * f32(count));
}

/// Region noise value in [0, 1) at a world position (continuous: region borders can run
/// through blocks), with a fine irregularity on the border line. See regionVariant.
fn regionValue(pos: vec3f, material: u32) -> f32 {
  let p = pos / material_params.variant_scale + f32(material) * 17.31;
  let warp = fastNoise(p * 0.5 + 31.7) - 0.5;
  let q = p + vec3f(0.8, 0.5, -0.6) * (warp * 2.0 * material_params.variant_warp);
  var n = fastNoise(q) * 0.7 + fastNoise(q * 2.3 + 5.1) * 0.3;
  n += (fastNoise(pos * 0.6 + 7.7) - 0.5) * 2.0 * material_params.variant_edge_noise;
  return clamp((n - 0.5) * 2.2 + 0.5, 0.0, 0.9999);
}

/// Height blend (as in terrain splatting): weight of B for linear weight `t`, from the two
/// height maps; `depth` = height range over which both show (0 = plain linear blend).
fn heightBlend(ha: f32, hb: f32, t: f32, depth: f32) -> f32 {
  if (depth <= 0.0) {
    return t;
  }
  let a = ha + (1.0 - t);
  let b = hb + t;
  let top = max(a, b) - depth;
  let wa = max(a - top, 0.0);
  let wb = max(b - top, 0.0);
  return wb / max(wa + wb, 1e-6);
}

/// Where and how a voxel face samples its material.
struct FaceMapping {
  layer: u32,
  /// World-space layers: the neighbouring variant across a region border and its weight
  /// (0 away from borders; 0.5 on the border line, so both sides meet continuously).
  layer2: u32,
  blend: f32,
  flags: u32,
  /// Sampled from the world-space arrays (natural materials).
  world: bool,
  /// World units → UV (1 per block face, 1 / world_span for world-space layers).
  uv_scale: f32,
  /// UV in the (possibly rotated) texture, per block face [0, 1]².
  uv: vec2f,
  /// World directions of +u and image-up after rotation.
  t: vec3f,
  b: vec3f,
  n: vec3f,
};

/// Picks the variant and rotation from the voxel/face hash and maps `local` (position
/// inside the voxel, [0, 1]³) to texture UV.
fn faceMapping(material: u32, cell: vec3i, face: u32, local: vec3f) -> FaceMapping {
  let info = materials[material];
  let frame = faceFrame(face);
  let h = hashCellFace(cell, face);
  var m: FaceMapping;
  let count = max(info.variants, 1u);
  var variant = h % count;
  if (material_params.variant_scale > 0.0 && count > 1u && (info.flags & MATERIAL_ALPHA_TEST) == 0u) {
    variant = regionVariant(cell, material, count);
  }
  m.layer = info.first_layer + variant;
  m.flags = info.flags;
  m.n = frame.n;
  m.world = false;
  m.uv_scale = 1.0;
  m.layer2 = m.layer;
  m.blend = 0.0;
  if ((info.flags & MATERIAL_WORLD) != 0u && material_params.world_enabled != 0u) {
    // Natural material: UV from the world position on the face plane, one layer per
    // world_span × world_span m, no per-block rotation — continuous across blocks, so no
    // block grid. The cell is reduced modulo the span first (exact, small numbers).
    let span = i32(material_params.world_span);
    let q = vec3f(((cell % span) + span) % span) + local;
    m.world = true;
    m.uv_scale = 1.0 / material_params.world_span;
    let first = info.flags >> MATERIAL_WORLD_FIRST_SHIFT;
    m.layer = first + variant;
    m.layer2 = m.layer;
    if (material_params.variant_scale > 0.0 && count > 1u) {
      // Variant from the region noise at the hit point itself, not the block centre; near
      // a border the neighbouring variant is blended in over a band of the noise value.
      let f = regionValue(vec3f(cell) + local, material) * f32(count);
      let i = min(u32(f), count - 1u);
      let frac = f - f32(i);
      let band = material_params.variant_blend;
      m.layer = first + i;
      m.layer2 = m.layer;
      if (band > 0.0) {
        if (frac > 0.5 && i + 1u < count) {
          m.layer2 = first + i + 1u;
          m.blend = 0.5 * smoothstep(1.0 - band, 1.0, frac);
        } else if (frac <= 0.5 && i > 0u) {
          m.layer2 = first + i - 1u;
          m.blend = 0.5 * (1.0 - smoothstep(0.0, band, frac));
        }
      }
    }
    m.uv = vec2f(dot(q, frame.t), -dot(q, frame.b)) * m.uv_scale;
    m.t = frame.t;
    m.b = frame.b;
    return m;
  }
  let p = local - 0.5;
  var uv = vec2f(dot(p, frame.t), -dot(p, frame.b));
  var t = frame.t;
  var b = frame.b;
  if ((info.flags & MATERIAL_ROTATE) != 0u) {
    // Rotate by k·90°: uv' = R·uv. The frame follows so normal maps stay correct:
    // T' = ∇u' = R00·T − R01·B, B' = −∇v' = −R10·T + R11·B.
    let k = (h >> 24u) & 3u;
    let c = array<f32, 4>(1.0, 0.0, -1.0, 0.0)[k];
    let s = array<f32, 4>(0.0, 1.0, 0.0, -1.0)[k];
    uv = vec2f(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
    let t0 = t;
    t = c * t0 + s * b;
    b = -s * t0 + c * b;
  }
  m.uv = uv + 0.5;
  m.t = t;
  m.b = b;
  return m;
}

/// Converts a world-space vector in the face plane to UV units.
fn toUv(m: FaceMapping, v: vec3f) -> vec2f {
  return vec2f(dot(v, m.t), -dot(v, m.b)) * m.uv_scale;
}

// Sampling of the mapped layer from the per-face or the world-space arrays.
fn sampleAlbedo(m: FaceMapping, uv: vec2f, g: Gradients) -> vec4f {
  if (m.world) {
    return textureSampleGrad(tex_world_albedo, tex_sampler, uv, m.layer, g.ddx, g.ddy);
  }
  return textureSampleGrad(tex_albedo, tex_sampler, uv, m.layer, g.ddx, g.ddy);
}

fn sampleNormal(m: FaceMapping, uv: vec2f, g: Gradients) -> vec4f {
  if (m.world) {
    return textureSampleGrad(tex_world_normal, tex_sampler, uv, m.layer, g.ddx, g.ddy);
  }
  return textureSampleGrad(tex_normal, tex_sampler, uv, m.layer, g.ddx, g.ddy);
}

fn sampleSpecular(m: FaceMapping, uv: vec2f, g: Gradients) -> vec4f {
  if (m.world) {
    return textureSampleGrad(tex_world_specular, tex_sampler, uv, m.layer, g.ddx, g.ddy);
  }
  return textureSampleGrad(tex_specular, tex_sampler, uv, m.layer, g.ddx, g.ddy);
}

/// Texture-space footprint of a ray cone of width `width` hitting the face along `dir`:
/// anisotropic gradients (stretched along the view direction on grazing faces).
struct Gradients {
  ddx: vec2f,
  ddy: vec2f,
};

fn coneGradients(m: FaceMapping, dir: vec3f, width: f32) -> Gradients {
  let cos_n = max(abs(dot(dir, m.n)), 0.05);
  var along = dir - m.n * dot(dir, m.n);
  if (dot(along, along) < 1e-6) {
    along = m.t;
  }
  along = normalize(along);
  let across = cross(m.n, along);
  let scale = exp2(material_params.lod_bias);
  return Gradients(toUv(m, along) * (width / cos_n) * scale, toUv(m, across) * width * scale);
}

/// Scalar LOD for the same cone (used where anisotropy is not worth it, e.g. alpha tests).
fn coneLod(dir: vec3f, n: vec3f, width: f32) -> f32 {
  let cos_n = max(abs(dot(dir, n)), 0.05);
  // Geometric mean of the major (width / cos) and minor (width) footprint axes.
  return log2(max(width * inverseSqrt(cos_n) * material_params.texture_size, 1e-6)) + material_params.lod_bias;
}

// ------------------------------------------------------------------ hex tiling
// Anti-tiling for world-space layers (Mikkelsen 2022, "Practical Real-Time Hex-Tiling"):
// the UV plane is covered by a triangle grid; every vertex is the centre of a hex cell
// with its own random rotation and offset into the texture, and a lookup blends the three
// cells around the point with barycentric weights (sharpened by hex_contrast).

struct HexTiles {
  uv0: vec2f,
  uv1: vec2f,
  uv2: vec2f,
  /// Cell rotations (cos, sin).
  r0: vec2f,
  r1: vec2f,
  r2: vec2f,
  w: vec3f,
};

fn rotate2(r: vec2f, v: vec2f) -> vec2f {
  return vec2f(r.x * v.x - r.y * v.y, r.y * v.x + r.x * v.y);
}

/// Random rotation (cos, sin) and offset of the hex cell at triangle-grid vertex `v`;
/// returns the cell's texture UV for `uv`.
fn hexCell(v: vec2f, uv: vec2f, k: f32, r: ptr<function, vec2f>) -> vec2f {
  let h = hashCellFace(vec3i(vec2i(v), 911), 3u);
  let angle = f32(h & 0xffffu) / 65536.0 * 6.28318530718;
  let offset = vec2f(f32((h >> 16u) & 0xffu), f32(h >> 24u)) / 256.0;
  *r = vec2f(cos(angle), sin(angle));
  let centre = vec2f(v.x + v.y * 0.5, v.y * 0.8660254) / k;
  return rotate2(*r, uv - centre) + centre + offset;
}

fn hexTiles(uv: vec2f) -> HexTiles {
  let k = material_params.hex_density * 3.4641016; // 2√3 triangle-grid units per UV unit
  let st = uv * k;
  let skewed = vec2f(st.x - st.y * 0.57735027, st.y * 1.15470054);
  let base = floor(skewed);
  let f = skewed - base;
  let z = 1.0 - f.x - f.y;
  var v0 = base;
  var v1 = base + vec2f(0.0, 1.0);
  var v2 = base + vec2f(1.0, 0.0);
  var w = vec3f(z, f.y, f.x);
  if (z <= 0.0) {
    v0 = base + vec2f(1.0, 1.0);
    w = vec3f(-z, 1.0 - f.y, 1.0 - f.x);
  }
  var t: HexTiles;
  t.uv0 = hexCell(v0, uv, k, &t.r0);
  t.uv1 = hexCell(v1, uv, k, &t.r1);
  t.uv2 = hexCell(v2, uv, k, &t.r2);
  let sharp = pow(max(w, vec3f(0.0)), vec3f(material_params.hex_contrast));
  t.w = sharp / max(sharp.x + sharp.y + sharp.z, 1e-6);
  return t;
}

fn hexSample(tex: texture_2d_array<f32>, layer: u32, t: HexTiles, g: Gradients) -> vec4f {
  return textureSampleGrad(tex, tex_sampler, t.uv0, layer, rotate2(t.r0, g.ddx), rotate2(t.r0, g.ddy)) * t.w.x +
    textureSampleGrad(tex, tex_sampler, t.uv1, layer, rotate2(t.r1, g.ddx), rotate2(t.r1, g.ddy)) * t.w.y +
    textureSampleGrad(tex, tex_sampler, t.uv2, layer, rotate2(t.r2, g.ddx), rotate2(t.r2, g.ddy)) * t.w.z;
}

/// Normal texel of one hex cell with its tangent-space xy rotated back into the surface's
/// frame (x = +u, y = image up = −v); b (AO) and a (height) as stored.
fn hexNormalCell(tex: texture_2d_array<f32>, layer: u32, uv: vec2f, r: vec2f, g: Gradients) -> vec4f {
  let n = textureSampleGrad(tex, tex_sampler, uv, layer, rotate2(r, g.ddx), rotate2(r, g.ddy));
  let xy = n.xy * 2.0 - 1.0;
  // Texture UV = R · surface UV, so a texture-space vector maps back with Rᵀ (in u, v).
  let back = rotate2(vec2f(r.x, -r.y), vec2f(xy.x, -xy.y));
  return vec4f(vec2f(back.x, -back.y) * 0.5 + 0.5, n.b, n.a);
}

fn hexSampleNormal(tex: texture_2d_array<f32>, layer: u32, t: HexTiles, g: Gradients) -> vec4f {
  return hexNormalCell(tex, layer, t.uv0, t.r0, g) * t.w.x + hexNormalCell(tex, layer, t.uv1, t.r1, g) * t.w.y +
    hexNormalCell(tex, layer, t.uv2, t.r2, g) * t.w.z;
}

/// The three maps of one layer at `uv`: hex-tiled for world-space layers when enabled.
struct TexelSet {
  albedo: vec4f,
  normal: vec4f,
  specular: vec4f,
};

fn sampleSet(m: FaceMapping, layer: u32, uv: vec2f, g: Gradients, hex: bool) -> TexelSet {
  if (hex) {
    let t = hexTiles(uv);
    return TexelSet(hexSample(tex_world_albedo, layer, t, g), hexSampleNormal(tex_world_normal, layer, t, g),
      hexSample(tex_world_specular, layer, t, g));
  }
  var ml = m;
  ml.layer = layer;
  return TexelSet(sampleAlbedo(ml, uv, g), sampleNormal(ml, uv, g), sampleSpecular(ml, uv, g));
}

/// Height (0..1) for parallax: world-space layers read the dominant hex cell only (one
/// lookup per march step).
fn parallaxHeight(m: FaceMapping, uv: vec2f, g: Gradients, dominant: HexTiles, hex: bool) -> f32 {
  if (hex) {
    var uv_t = dominant.uv0;
    var r = dominant.r0;
    if (dominant.w.y > dominant.w.x && dominant.w.y >= dominant.w.z) {
      uv_t = dominant.uv1;
      r = dominant.r1;
    } else if (dominant.w.z > dominant.w.x && dominant.w.z > dominant.w.y) {
      uv_t = dominant.uv2;
      r = dominant.r2;
    }
    // Follow the march: the dominant cell's transform applied to the shifted UV.
    let shifted = uv_t + rotate2(r, uv - m.uv);
    return textureSampleGrad(tex_world_normal, tex_sampler, shifted, m.layer, rotate2(r, g.ddx), rotate2(r, g.ddy)).a;
  }
  return sampleNormal(m, uv, g).a;
}

/// Parallax occlusion mapping: marches the height field along the view ray (tangent
/// space) and returns the UV where it enters the surface.
/// Hex tiling for this lookup: world-space layer, enabled, within the LOD distance.
fn useHex(m: FaceMapping, t: f32) -> bool {
  return m.world && material_params.hex_density > 0.0 && t < material_params.hex_max_distance;
}

fn parallaxUv(m: FaceMapping, dir: vec3f, g: Gradients, t: f32) -> vec2f {
  let depth = material_params.pom_depth;
  let steps = max(material_params.pom_steps, 1u);
  let v = -dir;
  let vz = max(dot(v, m.n), 0.2); // grazing views would shear the texture too far
  // UV shift per unit of depth below the surface.
  let shift = -toUv(m, v - m.n * dot(v, m.n)) / vz * depth;
  let layer = 1.0 / f32(steps);
  var uv = m.uv;
  var prev_uv = uv;
  var d = 0.0;
  var prev_gap = 0.0;
  let hex = useHex(m, t);
  var dominant: HexTiles;
  if (hex) {
    dominant = hexTiles(m.uv);
  }
  for (var i = 0u; i <= steps; i++) {
    let surface = 1.0 - parallaxHeight(m, uv, g, dominant, hex);
    let gap = d - surface;
    if (gap >= 0.0) {
      // Interpolate between the last point above and the first point below the surface:
      // the gap crosses zero a fraction prev_gap / (prev_gap − gap) of the way from prev.
      let w = prev_gap / (prev_gap - gap - 1e-6);
      return mix(prev_uv, uv, clamp(w, 0.0, 1.0));
    }
    prev_uv = uv;
    prev_gap = gap;
    d += layer;
    uv += shift * layer;
  }
  return uv;
}

struct Surface {
  albedo: vec3f, // linear
  ao: f32,
  normal: vec3f, // world, normal-mapped
  roughness: f32,
  metalness: f32,
  emission: f32,
  subsurface: f32,
};

fn srgbDecode(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

/// Full material evaluation for a primary hit at distance `t` along `dir`.
fn shadeSurface(material: u32, cell: vec3i, face: u32, local: vec3f, dir: vec3f, t: f32) -> Surface {
  let m = faceMapping(material, cell, face, local);
  let g = coneGradients(m, dir, t * material_params.pixel_spread);
  var uv = m.uv;
  if ((m.flags & MATERIAL_POM) != 0u && material_params.pom_depth > 0.0 && t < material_params.pom_max_distance) {
    uv = parallaxUv(m, dir, g, t);
  }
  let hex = useHex(m, t);
  var texels = sampleSet(m, m.layer, uv, g, hex);
  if (m.blend > 0.0) {
    // Variant region border: blend the neighbouring variant, by height — the higher
    // texels (pebble tops) of the incoming variant take over first.
    let other = sampleSet(m, m.layer2, uv, g, hex);
    let w = heightBlend(texels.normal.a, other.normal.a, m.blend, material_params.height_blend_depth);
    texels = TexelSet(mix(texels.albedo, other.albedo, w), mix(texels.normal, other.normal, w),
      mix(texels.specular, other.specular, w));
  }
  let a = texels.albedo;
  let nm = texels.normal;
  let sp = texels.specular;

  let xy = nm.xy * 2.0 - 1.0;
  let ts = vec3f(xy, sqrt(max(1.0 - dot(xy, xy), 0.0)));
  var s: Surface;
  s.albedo = srgbDecode(a.rgb);
  s.ao = nm.b;
  s.normal = normalize(m.t * ts.x + m.b * ts.y + m.n * ts.z);
  let smoothness = sp.r;
  s.roughness = (1.0 - smoothness) * (1.0 - smoothness);
  s.metalness = select(0.0, 1.0, sp.g >= F0_METAL);
  s.emission = select(sp.a * 255.0 / 254.0, 0.0, sp.a >= 1.0);
  s.subsurface = sp.b;
  applyDetail(&s, material, m.flags, cell, face, local, m.n, t);
  return s;
}

/// Alpha test for rays passing through a voxel face (leaves). True = opaque here.
fn alphaOpaque(block_id: u32, cell: vec3i, normal: vec3i, local: vec3f, dir: vec3f, t: f32) -> bool {
  if (all(normal == vec3i(0))) {
    return true;
  }
  let face = faceIndex(normal);
  let material = faceMaterial(block_id, face);
  if ((materials[material].flags & MATERIAL_ALPHA_TEST) == 0u) {
    return true;
  }
  let m = faceMapping(material, cell, face, local);
  let lod = coneLod(dir, m.n, t * material_params.pixel_spread);
  return textureSampleLevel(tex_albedo, tex_sampler, m.uv, m.layer, lod).a >= material_params.alpha_cutoff;
}

#include "material-detail.wgsl"
