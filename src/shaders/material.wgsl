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
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
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
const F0_METAL: f32 = 230.0 / 255.0;

@group(2) @binding(0) var<uniform> material_params: MaterialParams;
@group(2) @binding(1) var<storage, read> materials: array<MaterialInfo>;
/// Per block id: face material indices (top, side, bottom, unused).
@group(2) @binding(2) var<storage, read> block_faces: array<vec4u>;
@group(2) @binding(3) var tex_albedo: texture_2d_array<f32>;
@group(2) @binding(4) var tex_normal: texture_2d_array<f32>;
@group(2) @binding(5) var tex_specular: texture_2d_array<f32>;
@group(2) @binding(6) var tex_sampler: sampler;

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

/// Where and how a voxel face samples its material.
struct FaceMapping {
  layer: u32,
  flags: u32,
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
  return vec2f(dot(v, m.t), -dot(v, m.b));
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

/// Parallax occlusion mapping: marches the height field along the view ray (tangent
/// space) and returns the UV where it enters the surface.
fn parallaxUv(m: FaceMapping, dir: vec3f, g: Gradients) -> vec2f {
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
  for (var i = 0u; i <= steps; i++) {
    let surface = 1.0 - textureSampleGrad(tex_normal, tex_sampler, uv, m.layer, g.ddx, g.ddy).a;
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
    uv = parallaxUv(m, dir, g);
  }
  let a = textureSampleGrad(tex_albedo, tex_sampler, uv, m.layer, g.ddx, g.ddy);
  let nm = textureSampleGrad(tex_normal, tex_sampler, uv, m.layer, g.ddx, g.ddy);
  let sp = textureSampleGrad(tex_specular, tex_sampler, uv, m.layer, g.ddx, g.ddy);

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
