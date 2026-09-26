// SVGF-style denoiser for the half-resolution GI signals (Schied et al. 2017), diffuse
// and specular side by side but with separate statistics and filter strengths:
//
//   temporal  reprojects last frame's accumulation with the motion vectors (bilinear, each
//             tap validated by geometric face and expected face-plane coordinate), rescales it for the
//             exposure change, and blends in this frame's sample with weight 1 / history
//             length. The history cap drops from `history_still` to `history_moving` with
//             camera and pixel motion. Checkerboard pixels without a sample keep their
//             history; without history they are rebuilt from traced neighbours.
//             Also accumulates the first two luminance moments of each signal.
//   variance  per-pixel luminance variance: temporal (from the moments) once 4+ frames are
//             accumulated, a 7×7 spatial estimate before that.
//   atrous    one iteration of the edge-stopping à-trous wavelet (3×3 B-spline kernel
//             [1/4, 1/2, 1/4], step 2^i; five iterations reach ±31 half-res pixels):
//             other face planes (onPlane) and luminance (variance-scaled) stop the
//             filter at edges. Variance is filtered alongside.
//
// Accumulated textures: rgb + a = history length (frames). Filtered textures: rgb +
// a = luminance variance.
#include "gi-common.wgsl"
#include "sky.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: GiParams;
@group(0) @binding(2) var raw_diffuse: texture_2d<f32>;
@group(0) @binding(3) var raw_specular: texture_2d<f32>;
@group(0) @binding(4) var guide: texture_2d<u32>;
@group(0) @binding(5) var prev_guide: texture_2d<u32>;
@group(0) @binding(6) var gmotion: texture_2d<f32>;
@group(0) @binding(7) var prev_diffuse: texture_2d<f32>;
@group(0) @binding(8) var prev_specular: texture_2d<f32>;
@group(0) @binding(9) var prev_moments: texture_2d<f32>;
@group(0) @binding(10) var out_diffuse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var out_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var out_moments: texture_storage_2d<rgba32float, write>;
/// Exposure the history was stored with (written by the variance pass).
@group(0) @binding(13) var<storage, read> history_exposure: array<f32>;
// Variance pass.
@group(0) @binding(14) var acc_diffuse: texture_2d<f32>;
@group(0) @binding(15) var acc_specular: texture_2d<f32>;
@group(0) @binding(16) var moments: texture_2d<f32>;
@group(0) @binding(17) var var_diffuse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(18) var var_specular: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var<storage, read_write> history_exposure_out: array<f32>;
// À-trous pass.
struct AtrousStep {
  step: u32,
  /// 1 while the specular signal is still being filtered.
  specular: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(20) var<uniform> atrous: AtrousStep;
@group(0) @binding(21) var in_diffuse: texture_2d<f32>;
@group(0) @binding(22) var in_specular: texture_2d<f32>;
@group(0) @binding(23) var filtered_diffuse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(24) var filtered_specular: texture_storage_2d<rgba16float, write>;
// Emissive direct light (ReSTIR DI), a third signal: temporal / pass-through / à-trous.
@group(0) @binding(25) var raw_emissive: texture_2d<f32>;
@group(0) @binding(26) var prev_emissive: texture_2d<f32>;
@group(0) @binding(27) var out_emissive: texture_storage_2d<rgba16float, write>;
@group(0) @binding(28) var acc_emissive: texture_2d<f32>;
@group(0) @binding(29) var var_emissive: texture_storage_2d<rgba16float, write>;
@group(0) @binding(30) var in_emissive: texture_2d<f32>;
@group(0) @binding(31) var filtered_emissive: texture_storage_2d<rgba16float, write>;
/// Emissive signal edge-stopping: relative luminance difference (keeps torch shadows).
const EMISSIVE_SIGMA: f32 = 0.5;

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;
/// Frames of history before the temporal variance is trusted (spatial estimate before).
const MIN_VARIANCE_FRAMES: f32 = 4.0;

fn inHalf(p: vec2i) -> bool {
  return all(p >= vec2i(0)) && all(p < vec2i(params.half_size));
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn temporal(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inHalf(p)) {
    return;
  }
  let g = textureLoad(guide, p, 0);
  if (gbIsSky(g.z)) {
    textureStore(out_diffuse, p, vec4f(0.0));
    textureStore(out_specular, p, vec4f(0.0));
    textureStore(out_moments, p, vec4f(0.0));
    textureStore(out_emissive, p, vec4f(0.0));
    return;
  }
  let q = fullPixel(gid.xy, params.full_size, params.scale);
  let depth = guideDepth(g);
  let face = gbFace(g.z);
  let pos = pixelPosition(q, params.full_size, depth);
  let motion = textureLoad(gmotion, vec2i(q), 0).xy;
  let traced = isTraced(gid.xy, cam.frame_index, params.checkerboard);
  var d_sample = textureLoad(raw_diffuse, p, 0).rgb;
  var s_sample = textureLoad(raw_specular, p, 0).rgb;
  var e_sample = textureLoad(raw_emissive, p, 0).rgb;

  // Reprojection: bilinear over the previous half-res grid, each tap validated.
  let uv = (vec2f(q) + 0.5) / vec2f(params.full_size);
  let prev = giGridCoord(uv - motion, params.full_size, params.scale);
  let base = vec2i(floor(prev));
  let f = prev - vec2f(base);
  // Last frame's plane coordinate of this point (camera moved by prev_delta).
  let expected_plane = dot(faceNormal(face), pos + cam.prev_delta);
  var h_d = vec4f(0.0);
  var h_s = vec3f(0.0);
  var h_e = vec3f(0.0);
  var h_m = vec4f(0.0);
  var h_w = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let t = base + o;
    if (!inHalf(t)) {
      continue;
    }
    if (!onPlane(textureLoad(prev_guide, t, 0), face, expected_plane, params.plane_tolerance)) {
      continue;
    }
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    h_d += textureLoad(prev_diffuse, t, 0) * w;
    h_s += textureLoad(prev_specular, t, 0).rgb * w;
    h_e += textureLoad(prev_emissive, t, 0).rgb * w;
    h_m += textureLoad(prev_moments, t, 0) * w;
    h_w += w;
  }
  let valid = h_w > 1e-3;
  if (valid) {
    // The history was stored with last frame's exposure.
    let rescale = preExposure() / max(history_exposure[0], 1e-20);
    h_d = vec4f(h_d.rgb / h_w * rescale, h_d.a / h_w);
    h_s = h_s / h_w * rescale;
    h_e = h_e / h_w * rescale;
    h_m = h_m / h_w * vec4f(rescale, rescale * rescale, rescale, rescale * rescale);
  }

  if (!traced && !valid) {
    // Gap in the traced pattern without history: rebuild from this frame's traced
    // neighbours (every 2×2 block holds at least one).
    var sum_d = vec3f(0.0);
    var sum_s = vec3f(0.0);
    var sum_e = vec3f(0.0);
    var wsum = 0.0;
    for (var k = 0; k < 9; k++) {
      let t = p + vec2i(k % 3 - 1, k / 3 - 1);
      if (k == 4 || !inHalf(t)) {
        continue;
      }
      if (!isTraced(vec2u(t), cam.frame_index, params.checkerboard) ||
          !onPlane(textureLoad(guide, t, 0), face, guidePlane(g), params.plane_tolerance)) {
        continue;
      }
      sum_d += textureLoad(raw_diffuse, t, 0).rgb;
      sum_s += textureLoad(raw_specular, t, 0).rgb;
      sum_e += textureLoad(raw_emissive, t, 0).rgb;
      wsum += 1.0;
    }
    d_sample = sum_d / max(wsum, 1.0);
    s_sample = sum_s / max(wsum, 1.0);
    e_sample = sum_e / max(wsum, 1.0);
  }

  let moving = clamp(params.camera_motion + length(motion * vec2f(params.full_size)) / params.motion_pixels, 0.0, 1.0);
  let max_frames = mix(params.history_still, params.history_moving, moving);
  let ld = luminance(d_sample);
  let ls = luminance(s_sample);
  var frames = 1.0;
  var acc_d = d_sample;
  var acc_s = s_sample;
  var acc_e = e_sample;
  var mom = vec4f(ld, ld * ld, ls, ls * ls);
  if (valid) {
    frames = min(select(h_d.a, h_d.a + 1.0, traced), max_frames);
    let a = select(0.0, 1.0 / frames, traced);
    acc_d = mix(h_d.rgb, d_sample, a);
    acc_s = mix(h_s, s_sample, a);
    acc_e = mix(h_e, e_sample, a);
    mom = mix(h_m, mom, a);
  }
  textureStore(out_diffuse, p, vec4f(acc_d, frames));
  textureStore(out_specular, p, vec4f(acc_s, frames));
  textureStore(out_moments, p, mom);
  textureStore(out_emissive, p, vec4f(acc_e, frames));
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn variance(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (all(p == vec2i(0))) {
    history_exposure_out[0] = preExposure();
  }
  if (!inHalf(p)) {
    return;
  }
  let g = textureLoad(guide, p, 0);
  let d = textureLoad(acc_diffuse, p, 0);
  let s = textureLoad(acc_specular, p, 0);
  textureStore(var_emissive, p, vec4f(textureLoad(acc_emissive, p, 0).rgb, 0.0));
  if (gbIsSky(g.z)) {
    textureStore(var_diffuse, p, vec4f(0.0));
    textureStore(var_specular, p, vec4f(0.0));
    return;
  }
  var m = textureLoad(moments, p, 0);
  if (d.a < MIN_VARIANCE_FRAMES) {
    // Too little history: estimate the moments over a 7×7 neighbourhood of the same surface.
    let face = gbFace(g.z);
    var sum = vec4f(0.0);
    var wsum = 0.0;
    for (var y = -3; y <= 3; y++) {
      for (var x = -3; x <= 3; x++) {
        let t = p + vec2i(x, y);
        if (!inHalf(t)) {
          continue;
        }
        if (!onPlane(textureLoad(guide, t, 0), face, guidePlane(g), params.plane_tolerance)) {
          continue;
        }
        let ld = luminance(textureLoad(acc_diffuse, t, 0).rgb);
        let ls = luminance(textureLoad(acc_specular, t, 0).rgb);
        sum += vec4f(ld, ld * ld, ls, ls * ls);
        wsum += 1.0;
      }
    }
    m = sum / max(wsum, 1.0);
    // Few frames: be generous so the à-trous filter smooths harder.
    let boost = MIN_VARIANCE_FRAMES / max(d.a, 1.0);
    m = vec4f(m.x, m.x * m.x + (m.y - m.x * m.x) * boost, m.z, m.z * m.z + (m.w - m.z * m.z) * boost);
  }
  textureStore(var_diffuse, p, vec4f(d.rgb, max(m.y - m.x * m.x, 0.0)));
  textureStore(var_specular, p, vec4f(s.rgb, max(m.w - m.z * m.z, 0.0)));
}

/// 3×3 Gaussian of the variance (SVGF): steadier luminance edge-stopping. Only the first
/// iteration needs it; later ones read variance that is already filtered.
fn blurredVariance(tex: texture_2d<f32>, p: vec2i) -> f32 {
  if (atrous.step > 1u) {
    return textureLoad(tex, p, 0).a;
  }
  var sum = 0.0;
  var wsum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let t = p + vec2i(x, y);
      if (!inHalf(t)) {
        continue;
      }
      let w = select(0.5, 1.0, x == 0) * select(0.5, 1.0, y == 0);
      sum += textureLoad(tex, t, 0).a * w;
      wsum += w;
    }
  }
  return sum / wsum;
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn atrousStep(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (!inHalf(p)) {
    return;
  }
  let g = textureLoad(guide, p, 0);
  let cd = textureLoad(in_diffuse, p, 0);
  let cs = textureLoad(in_specular, p, 0);
  let ce = textureLoad(in_emissive, p, 0);
  if (gbIsSky(g.z)) {
    textureStore(filtered_diffuse, p, cd);
    textureStore(filtered_specular, p, cs);
    textureStore(filtered_emissive, p, ce);
    return;
  }
  let le = luminance(ce.rgb);
  // No emitters in the light list: the emissive signal is zero, skip its filtering.
  let has_emitters = params.light_count > 0u && params.restir_enabled != 0u;
  let face = gbFace(g.z);
  let plane_p = guidePlane(g);
  let rough = guideRoughness(g);
  // Rough surfaces carry no specular signal (their specular comes from the irradiance).
  let filter_spec = atrous.specular != 0u && specularProbability(rough, params.spec_threshold) > 0.0;
  let ld = luminance(cd.rgb);
  let ls = luminance(cs.rgb);
  let sd = params.sigma_luminance * sqrt(blurredVariance(in_diffuse, p)) + 1e-4;
  var ss = 1.0;
  if (filter_spec) {
    ss = params.sigma_luminance_spec * sqrt(blurredVariance(in_specular, p)) + 1e-4;
  }

  let kernel = array<f32, 2>(0.5, 0.25);
  var sum_d = vec4f(0.0);
  var sum_s = vec4f(0.0);
  var sum_e = vec3f(0.0);
  var w_d = 0.0;
  var w_s = 0.0;
  var w_e = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let t = p + vec2i(x, y) * i32(atrous.step);
      if (!inHalf(t)) {
        continue;
      }
      let tg = textureLoad(guide, t, 0);
      if (!onPlane(tg, face, plane_p, params.plane_tolerance)) {
        continue;
      }
      let w_geo = kernel[abs(x)] * kernel[abs(y)];
      let td = textureLoad(in_diffuse, t, 0);
      let wd = w_geo * exp(-abs(luminance(td.rgb) - ld) / sd);
      sum_d += vec4f(td.rgb * wd, td.a * wd * wd);
      w_d += wd;
      if (has_emitters) {
        let te = textureLoad(in_emissive, t, 0).rgb;
        let lte = luminance(te);
        let we = w_geo * exp(-abs(lte - le) / (EMISSIVE_SIGMA * max(lte, le) + 1e-6));
        sum_e += te * we;
        w_e += we;
      }
      if (filter_spec) {
        let ts = textureLoad(in_specular, t, 0);
        let ws = w_geo * exp(-abs(luminance(ts.rgb) - ls) / ss) *
          exp(-abs(guideRoughness(tg) - rough) * params.spec_roughness_sigma);
        sum_s += vec4f(ts.rgb * ws, ts.a * ws * ws);
        w_s += ws;
      }
    }
  }
  // The centre tap always has weight > 0, so the sums are never empty.
  textureStore(filtered_diffuse, p, vec4f(sum_d.rgb / w_d, sum_d.a / (w_d * w_d)));
  textureStore(filtered_emissive, p, vec4f(sum_e / max(w_e, 1e-6), 0.0));
  if (filter_spec) {
    textureStore(filtered_specular, p, vec4f(sum_s.rgb / w_s, sum_s.a / (w_s * w_s)));
  } else {
    textureStore(filtered_specular, p, cs);
  }
}
