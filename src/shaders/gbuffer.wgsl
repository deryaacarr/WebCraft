// G-buffer encoding shared by the writer (primary.wgsl) and readers.
//
// gbuffer0 (rgba32uint):
//   x  albedo, pack4x8unorm (linear rgb, a unused)
//   y  normal, octahedral pack2x16snorm
//   z  face UV, pack2x16unorm
//   w  bits 0-7 block id · 8-19 material index · 20-30 DDA steps (clamped) · 31 sky flag
// depth  (r32float):  linear view depth (distance along the camera forward axis)
// motion (rg32float): screen UV delta, current − previous

const GB_SKY_FLAG: u32 = 0x80000000u;
const GB_STEPS_MAX: u32 = 0x7ffu;

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

fn packMaterialWord(block_id: u32, material: u32, steps: u32, sky: bool) -> u32 {
  return (block_id & 0xffu) | ((material & 0xfffu) << 8u) | (min(steps, GB_STEPS_MAX) << 20u) | select(0u, GB_SKY_FLAG, sky);
}

fn gbIsSky(word: u32) -> bool {
  return (word & GB_SKY_FLAG) != 0u;
}

fn gbBlockId(word: u32) -> u32 {
  return word & 0xffu;
}

fn gbMaterial(word: u32) -> u32 {
  return (word >> 8u) & 0xfffu;
}

fn gbSteps(word: u32) -> u32 {
  return (word >> 20u) & GB_STEPS_MAX;
}
