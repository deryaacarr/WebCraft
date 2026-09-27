// Large-scale variation of material albedo (stage 8.5, item 2), applied after texturing in
// shadeSurface. Everything is a gentle multiplicative change of albedo — no light is added.
//
//   natural (stone, cobblestone, gravel, dirt, sand)
//     macro   tone shifts over ~macro_scale m: darker ↔ lighter and warm ↔ cool
//     slope   flat (upward) faces lighter and dusty, steep faces darker and bare
//     moss    patches where it is damp: near water, leaves and logs (5 voxel probes around
//             the block), under overhangs and in cave mouths (solid blocks above the face),
//             on north-facing (−Z) steep faces; shaped by noise, not on downward faces
//   foliage (grass top / side, leaves)
//     moisture noise over ~foliage_scale m tints between dry (yellowish) and lush (dark
//     green); on grass sides only the green texels change (the dirt stays)
//
// Included at the end of material.wgsl (needs the brickmap for the moss probes).
#include "brickmap.wgsl"

fn detailColor(r: f32, g: f32, b: f32) -> vec3f {
  return vec3f(r, g, b);
}

/// Shade from above: fraction of the probes 1, 2 and 4 blocks above the face's open side
/// that are solid (overhangs, cave mouths — low sky visibility).
fn overhang(cell: vec3i, n: vec3f) -> f32 {
  let open = cell + vec3i(round(n));
  var solid = 0.0;
  for (var k = 0; k < 2; k++) {
    if (getVoxel(open + vec3i(0, array<i32, 2>(1, 3)[k], 0)) != 0u) {
      solid += 1.0;
    }
  }
  return solid / 2.0;
}

/// Fraction (0..1) of the probes around `cell` that hold water, leaves or logs.
fn moisture(cell: vec3i) -> f32 {
  let probes = array<vec3i, 5>(
    vec3i(2, 0, 0), vec3i(-2, 0, 0), vec3i(0, 0, 2), vec3i(0, 0, -2), vec3i(0, 3, 0),
  );
  var wet = 0.0;
  for (var i = 0; i < 5; i++) {
    let id = getVoxel(cell + probes[i]);
    if (id != 0u && (materials[faceMaterial(id, 1u)].flags & MATERIAL_MOIST) != 0u) {
      wet += 1.0;
    }
  }
  return clamp(wet / 2.0, 0.0, 1.0);
}

fn applyDetail(s: ptr<function, Surface>, material: u32, flags: u32, cell: vec3i, face: u32, local: vec3f, n: vec3f, t: f32) {
  let p = material_params;
  let pos = vec3f(cell) + local;

  if ((flags & MATERIAL_NATURAL) != 0u) {
    var albedo = (*s).albedo;
    if (p.macro_strength > 0.0) {
      let q = pos / p.macro_scale;
      let light = fastNoise(q) * 0.65 + fastNoise(q * 2.7 + 13.1) * 0.35;
      let hue = fastNoise(q * 0.8 + 41.3);
      let tone = mix(detailColor(p.cool_r, p.cool_g, p.cool_b), detailColor(p.warm_r, p.warm_g, p.warm_b), hue);
      let shade = 1.0 + (light - 0.5) * 0.7; // ±35 % around the texture's own brightness
      albedo *= mix(vec3f(1.0), tone * shade, p.macro_strength);
    }
    if (p.slope_strength > 0.0) {
      // Upward faces collect dust, vertical ones stay bare and a little darker; the dust
      // comes in patches.
      let dust = smoothstep(0.35, 0.75, fastNoise(pos * 0.35 + 3.3));
      let up = max(n.y, 0.0);
      let flat_tint = mix(vec3f(1.0), detailColor(p.dust_r, p.dust_g, p.dust_b), dust);
      let steep = 1.0 - 0.12 * (1.0 - abs(n.y));
      albedo *= mix(vec3f(1.0), mix(vec3f(steep), flat_tint, up), p.slope_strength);
    }
    if (p.moss > 0.0 && n.y > -0.5 && t < p.moss_max_distance) {
      // Dampness: moisture sources nearby, shade from above, north-facing steep faces.
      let north = smoothstep(0.5, 0.9, -n.z) * (1.0 - abs(n.y));
      let wet = max(moisture(cell), max(overhang(cell, n) * 0.8, north * 0.3));
      if (wet > 0.0) {
        let blotch = fastNoise(pos * 0.9 + 19.7) * 0.7 + fastNoise(pos * 3.1) * 0.3;
        let cover = smoothstep(0.35, 0.6, blotch * 0.6 + wet * 0.5) * p.moss * (0.6 + 0.4 * max(n.y, 0.0));
        albedo = mix(albedo, detailColor(p.moss_r, p.moss_g, p.moss_b) * (0.8 + 0.4 * blotch), cover);
        (*s).roughness = mix((*s).roughness, 0.9, cover);
      }
    }
    (*s).albedo = albedo;
  }

  if ((flags & MATERIAL_FOLIAGE) != 0u && p.foliage_strength > 0.0) {
    let q = pos / p.foliage_scale;
    let wetness = fastNoise(q + 71.9) * 0.7 + fastNoise(q * 3.3) * 0.3;
    let tint = mix(detailColor(p.dry_r, p.dry_g, p.dry_b), detailColor(p.lush_r, p.lush_g, p.lush_b), smoothstep(0.3, 0.7, wetness));
    // Only green texels (grass sides keep their dirt).
    let a = (*s).albedo;
    let green = smoothstep(0.0, 0.04, a.g - max(a.r, a.b));
    (*s).albedo = a * mix(vec3f(1.0), tint, p.foliage_strength * green);
  }
}
