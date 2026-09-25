// G-buffer encoding shared by the writer (primary.wgsl) and readers.
//
// gbuffer0 (rgba32uint):
//   x  pack4x8unorm(linear albedo rgb, ambient occlusion)
//   y  shading normal (normal-mapped), octahedral pack2x16snorm
//   z  pack4x8unorm(roughness, metalness, emission, subsurface)
//   w  bits 0-7 block id · 8-15 material index · 16-18 geometric face
//      (0..5 = +X −X +Y −Y +Z −Z) · 19-30 DDA steps (clamped) · 31 sky flag
// depth  (r32float):  linear view depth (distance along the camera forward axis)
// motion (rg32float): screen UV delta, current − previous
// Bit positions are mirrored in src/gpu/gbuffer.ts.

const GB_SKY_FLAG: u32 = 0x80000000u;
const GB_FACE_SHIFT: u32 = 16u;
const GB_STEPS_SHIFT: u32 = 19u;
const GB_STEPS_MAX: u32 = 0xfffu;

fn octEncode(n: vec3f) -> vec2f {
  var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z < 0.0) {
    p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
  }
  return p;
}

fn octDecode(e: vec2f) -> vec3f {
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

/// Face index 0..5 (+X −X +Y −Y +Z −Z) of an axis-aligned unit normal.
fn faceIndex(n: vec3i) -> u32 {
  if (n.x != 0) {
    return select(1u, 0u, n.x > 0);
  }
  if (n.y != 0) {
    return select(3u, 2u, n.y > 0);
  }
  return select(5u, 4u, n.z > 0);
}

fn faceNormal(face: u32) -> vec3f {
  let axis = face >> 1u;
  let s = select(1.0, -1.0, (face & 1u) == 1u);
  return vec3f(select(0.0, s, axis == 0u), select(0.0, s, axis == 1u), select(0.0, s, axis == 2u));
}

fn packMaterialWord(block_id: u32, material: u32, face: u32, steps: u32, sky: bool) -> u32 {
  return (block_id & 0xffu) | ((material & 0xffu) << 8u) | ((face & 7u) << GB_FACE_SHIFT) |
    (min(steps, GB_STEPS_MAX) << GB_STEPS_SHIFT) | select(0u, GB_SKY_FLAG, sky);
}

fn gbIsSky(word: u32) -> bool {
  return (word & GB_SKY_FLAG) != 0u;
}

fn gbBlockId(word: u32) -> u32 {
  return word & 0xffu;
}

fn gbMaterial(word: u32) -> u32 {
  return (word >> 8u) & 0xffu;
}

fn gbFace(word: u32) -> u32 {
  return (word >> GB_FACE_SHIFT) & 7u;
}

fn gbSteps(word: u32) -> u32 {
  return (word >> GB_STEPS_SHIFT) & GB_STEPS_MAX;
}
