struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

// Single oversized triangle covering the screen; no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
  let p = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var out: VertexOut;
  out.position = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  return out;
}

#include "color.wgsl"

struct BlitParams {
  /// 0 = show linear values as they are; 1 = AgX, 2 = ACES, 3 = AgX "punchy" look
  /// (pre-exposed HDR input, with the white balance from the exposure pass applied first).
  tonemap: u32,
  /// AgX punchy look: contrast power and saturation in AgX log space.
  punchy_contrast: f32,
  punchy_saturation: f32,
  _pad: f32,
};

/// Exposure pass state (exposure.wgsl): only the white-balance matrix is used here.
struct ExposureState {
  exposure: f32,
  // Scalars, not a vec3f: a vec3f would be 16-byte aligned and shift everything below.
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  wb_gains: vec4f,
  wb0: vec4f,
  wb1: vec4f,
  wb2: vec4f,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BlitParams;
@group(0) @binding(3) var<storage, read> exposure_state: ExposureState;

// ACES filmic fit by Stephen Hill (RRT + ODT), sRGB primaries in and out.
const ACES_IN: mat3x3f = mat3x3f(
  vec3f(0.59719, 0.07600, 0.02840),
  vec3f(0.35458, 0.90834, 0.13383),
  vec3f(0.04823, 0.01566, 0.83777),
);
const ACES_OUT: mat3x3f = mat3x3f(
  vec3f(1.60475, -0.10208, -0.00327),
  vec3f(-0.53108, 1.10813, -0.07276),
  vec3f(-0.07367, -0.00605, 1.07602),
);

fn aces(color: vec3f) -> vec3f {
  let v = ACES_IN * color;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(ACES_OUT * (a / b), vec3f(0.0), vec3f(1.0));
}

// AgX (Troy Sobotka), minimal fit by Benjamin Wrensch: log encoding in an inset gamut,
// a sigmoid contrast curve, then back out. Compresses highlights towards white smoothly
// and keeps hues from skewing. Returns linear sRGB.
const AGX_IN: mat3x3f = mat3x3f(
  vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
  vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
  vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
);
const AGX_OUT: mat3x3f = mat3x3f(
  vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
  vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
  vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
);
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(color: vec3f, punchy: bool) -> vec3f {
  var v = AGX_IN * max(color, vec3f(1e-10));
  v = clamp(log2(v), vec3f(AGX_MIN_EV), vec3f(AGX_MAX_EV));
  v = (v - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
  v = agxContrast(v);
  if (punchy) {
    // "Punchy" look (Wrensch): more contrast and saturation in AgX space.
    v = pow(max(v, vec3f(0.0)), vec3f(params.punchy_contrast));
    let luma = dot(v, vec3f(0.2126, 0.7152, 0.0722));
    v = luma + params.punchy_saturation * (v - luma);
  }
  v = AGX_OUT * v;
  return pow(clamp(v, vec3f(0.0), vec3f(1.0)), vec3f(2.2));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
  var c = textureSample(source, source_sampler, in.uv).rgb;
  if (params.tonemap != 0u) {
    let wb = mat3x3f(exposure_state.wb0.xyz, exposure_state.wb1.xyz, exposure_state.wb2.xyz);
    c = max(wb * c, vec3f(0.0));
    if (params.tonemap == 2u) {
      c = aces(c);
    } else {
      c = agx(c, params.tonemap == 3u);
    }
  }
  // Scene textures are linear; the swap chain is a plain (non-sRGB) format.
  return vec4f(linearToSrgb(c), 1.0);
}
