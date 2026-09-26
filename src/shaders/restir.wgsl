// Simplified ReSTIR DI (Bitterli et al. 2020) for emissive blocks, at GI (half)
// resolution. One reservoir per pixel selects one light; the final shadow ray and
// shading happen in gi-trace.wgsl.
//
//   prepare  writes this frame's GI guide, then: RIS over `restir_candidates` lights drawn
//            uniformly from the light list (the emitters nearest the camera), target
//            p̂ = luminance(Le) · projected cube area · cos / d²; temporal reuse of last
//            frame's final reservoir at the reprojected pixel (M capped).
//   spatial  combines the reservoir with a few random neighbours of similar depth/normal.
//
// Simplifications: p̂ ignores visibility and neighbours are reused without the unbiased
// MIS weights (slight bias towards occluded lights, hidden by the final shadow ray).
//
// Reservoir (rgba32uint): x, z = light cell x, z (i32 bits); y = (cell.y + 32768) & 0xffff |
// block id << 16; w = pack2x16float(W, M). M = 0 means empty.
#include "gi-common.wgsl"
#include "brickmap.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: GiParams;
@group(0) @binding(2) var gbuffer0: texture_2d<u32>;
@group(0) @binding(3) var gdepth: texture_2d<f32>;
@group(0) @binding(4) var gmotion: texture_2d<f32>;
@group(0) @binding(5) var guide_out: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(6) var prev_guide: texture_2d<u32>;
@group(0) @binding(7) var prev_reservoirs: texture_2d<u32>;
@group(0) @binding(8) var reservoirs_out: texture_storage_2d<rgba32uint, write>;
/// Light list: absolute cell (xyz) and block id (w).
@group(0) @binding(9) var<storage, read> lights: array<vec4i>;
/// Emitted radiance per block id (rgb, emissive strength included).
@group(0) @binding(10) var<storage, read> block_radiance: array<vec4f>;
// Spatial pass.
@group(0) @binding(11) var guide: texture_2d<u32>;
@group(0) @binding(12) var reservoirs_in: texture_2d<u32>;

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

struct Reservoir {
  cell: vec3i,
  block: u32,
  w_sum: f32,
  m: f32,
  /// Unbiased contribution weight of the selected sample.
  w: f32,
};

fn emptyReservoir() -> Reservoir {
  return Reservoir(vec3i(0), 0u, 0.0, 0.0, 0.0);
}

fn encodeReservoir(r: Reservoir) -> vec4u {
  let y = (u32(r.cell.y + 32768) & 0xffffu) | (r.block << 16u);
  return vec4u(bitcast<u32>(r.cell.x), y, bitcast<u32>(r.cell.z), pack2x16float(vec2f(r.w, r.m)));
}

fn decodeReservoir(e: vec4u) -> Reservoir {
  let wm = unpack2x16float(e.w);
  let cell = vec3i(bitcast<i32>(e.x), i32(e.y & 0xffffu) - 32768, bitcast<i32>(e.z));
  return Reservoir(cell, e.y >> 16u, 0.0, wm.y, wm.x);
}

/// Target function: unshadowed irradiance luminance from a light cube at `cell`.
fn targetPdf(pos: vec3f, n: vec3f, cell: vec3i, block: u32) -> f32 {
  let to = vec3f(cell - cam.origin_cell) + 0.5 - pos;
  let d2 = max(dot(to, to), 0.25);
  let l = to * inverseSqrt(d2);
  let cos_n = dot(n, l);
  if (cos_n <= 0.0) {
    return 0.0;
  }
  // Projected area of a unit cube seen along l.
  let area = abs(l.x) + abs(l.y) + abs(l.z);
  return luminance(block_radiance[block].rgb) * area * cos_n / d2;
}

/// Streams a sample into the reservoir (weighted reservoir sampling).
fn addSample(r: ptr<function, Reservoir>, cell: vec3i, block: u32, weight: f32, m: f32, u: f32) {
  (*r).w_sum += weight;
  (*r).m += m;
  if (weight > 0.0 && u * (*r).w_sum < weight) {
    (*r).cell = cell;
    (*r).block = block;
  }
}

fn finalize(r: ptr<function, Reservoir>, pos: vec3f, n: vec3f) {
  let p_hat = targetPdf(pos, n, (*r).cell, (*r).block);
  (*r).w = select(0.0, (*r).w_sum / ((*r).m * p_hat), p_hat > 0.0 && (*r).m > 0.0);
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn prepare(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= params.half_size)) {
    return;
  }
  let q = fullPixel(p, params.full_size, params.scale);
  let g0 = textureLoad(gbuffer0, vec2i(q), 0);
  let depth = textureLoad(gdepth, vec2i(q), 0).x;
  let n = octDecode(unpack2x16snorm(g0.y));
  let rel = pixelPosition(q, params.full_size, depth);
  let plane = dot(faceNormal(gbFace(g0.w)), rel);
  textureStore(guide_out, vec2i(p), packGuide(depth, n, g0.w, unpack4x8unorm(g0.z).x, plane));
  if (gbIsSky(g0.w) || params.restir_enabled == 0u || params.light_count == 0u) {
    textureStore(reservoirs_out, vec2i(p), vec4u(0u));
    return;
  }
  let pos = cam.origin_frac + rel;
  var counter = 0u;
  var r = emptyReservoir();

  // Initial candidates, drawn uniformly from the light list (source pdf 1 / count).
  let count = f32(params.light_count);
  for (var i = 0u; i < params.restir_candidates; i++) {
    let k = min(u32(giRandom(p, cam.frame_index, &counter) * count), params.light_count - 1u);
    let light = lights[k];
    let weight = targetPdf(pos, n, light.xyz, u32(light.w)) * count;
    addSample(&r, light.xyz, u32(light.w), weight, 1.0, giRandom(p, cam.frame_index, &counter));
  }
  finalize(&r, pos, n);

  // Temporal reuse: last frame's final reservoir where this surface was.
  let uv = (vec2f(q) + 0.5) / vec2f(params.full_size);
  let prev_uv = uv - textureLoad(gmotion, vec2i(q), 0).xy;
  if (all(prev_uv >= vec2f(0.0)) && all(prev_uv < vec2f(1.0))) {
    let pp = clamp(vec2i(round(giGridCoord(prev_uv, params.full_size, params.scale))), vec2i(0), vec2i(params.half_size) - 1);
    let face = gbFace(g0.w);
    let expected_plane = dot(faceNormal(face), rel + cam.prev_delta);
    if (onPlane(textureLoad(prev_guide, pp, 0), face, expected_plane, params.plane_tolerance)) {
      var prev = decodeReservoir(textureLoad(prev_reservoirs, pp, 0));
      // The light may have been removed since.
      if (prev.m > 0.0 && getVoxel(prev.cell) == prev.block) {
        prev.m = min(prev.m, params.restir_max_m);
        var merged = emptyReservoir();
        addSample(&merged, r.cell, r.block, targetPdf(pos, n, r.cell, r.block) * r.w * r.m, r.m, 0.0);
        addSample(&merged, prev.cell, prev.block, targetPdf(pos, n, prev.cell, prev.block) * prev.w * prev.m, prev.m,
          giRandom(p, cam.frame_index, &counter));
        finalize(&merged, pos, n);
        r = merged;
      }
    }
  }
  textureStore(reservoirs_out, vec2i(p), encodeReservoir(r));
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn spatial(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= params.half_size)) {
    return;
  }
  let g = textureLoad(guide, vec2i(p), 0);
  let own = decodeReservoir(textureLoad(reservoirs_in, vec2i(p), 0));
  if (gbIsSky(g.z) || params.restir_enabled == 0u || params.light_count == 0u) {
    textureStore(reservoirs_out, vec2i(p), vec4u(0u));
    return;
  }
  let q = fullPixel(p, params.full_size, params.scale);
  let depth = guideDepth(g);
  let pos = cam.origin_frac + pixelPosition(q, params.full_size, depth);
  let n = guideNormal(g);
  var counter = 17u;
  var r = emptyReservoir();
  addSample(&r, own.cell, own.block, targetPdf(pos, n, own.cell, own.block) * own.w * own.m, own.m, 0.0);
  for (var i = 0u; i < params.restir_spatial_count; i++) {
    let a = giRandom(p, cam.frame_index, &counter) * 2.0 * GI_PI;
    let d = sqrt(giRandom(p, cam.frame_index, &counter)) * params.restir_spatial_radius;
    let np = vec2i(p) + vec2i(round(vec2f(cos(a), sin(a)) * d));
    if (any(np < vec2i(0)) || any(np >= vec2i(params.half_size)) || all(np == vec2i(p))) {
      continue;
    }
    let ng = textureLoad(guide, np, 0);
    if (!onPlane(ng, gbFace(g.z), guidePlane(g), params.plane_tolerance) || dot(guideNormal(ng), n) < 0.9) {
      continue;
    }
    let nr = decodeReservoir(textureLoad(reservoirs_in, np, 0));
    if (nr.m <= 0.0) {
      continue;
    }
    addSample(&r, nr.cell, nr.block, targetPdf(pos, n, nr.cell, nr.block) * nr.w * nr.m, nr.m,
      giRandom(p, cam.frame_index, &counter));
  }
  finalize(&r, pos, n);
  textureStore(reservoirs_out, vec2i(p), encodeReservoir(r));
}
