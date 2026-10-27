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
        // Crisp patches (narrow threshold), not a tint over the whole face: the block must
        // still read as its own material.
        let blotch = fastNoise(pos * 1.4 + 19.7) * 0.65 + fastNoise(pos * 4.3) * 0.35;
        let cover = smoothstep(0.44, 0.52, blotch * 0.6 + wet * 0.4) * p.moss * (0.6 + 0.4 * max(n.y, 0.0));
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

// ------------------------------------------------------------------ worn edges
// Worn edges (applied in the lighting pass; shading normal, albedo and roughness only —
// no geometry, no gloss):
//   - soft edge: near an exposed edge of the hit face (the block across it lets light and
//     view through) the normal leans towards the side face. The radius varies irregularly
//     along the edge (two noise octaves at ~0.3 m and ~0.1 m, clipped at 0): nearly sharp
//     in places, worn to ~2× the mean elsewhere (`bevel_width` = mean, `bevel_variation`).
//     The lean is jittered by ~2 cm noise so the edge looks rough, never like a cylinder.
//   - colour: exposed outer edges slightly lighter and dusty; inside corners (the face
//     meets a wall: the block across the edge and the one diagonally beyond are solid)
//     slightly darker — dust and grime (`edge_tint`).
//   - chip marks: sparse small dark, rough spots along exposed edges (`chip_marks`).
//   - roughness in the band only goes up (`bevel_roughness`). Grass is skipped.
// Edges whose neighbour is solid and flush (the surface continues) stay as they are.

/// Whether block `id` lets light and view through (air counts).
fn seeThrough(id: u32) -> bool {
  return id == 0u || (block_faces[id].w & 1u) != 0u;
}

/// Blocks that never get a soft edge (grass: its top edge is a grass mat).
fn noBevel(id: u32) -> bool {
  return (block_faces[id].w & 2u) != 0u;
}

/// Colour / chip band along edges (blocks).
const EDGE_BAND: f32 = 0.05;

fn applyBevel(s: ptr<function, Surface>, flags: u32, cell: vec3i, face: u32, local: vec3f, t: f32) {
  let p = material_params;
  // Alpha-tested foliage is never touched; skip it before any lookup.
  if ((p.bevel_enabled & 1u) == 0u || t > p.bevel_max_distance || (flags & MATERIAL_ALPHA_TEST) != 0u) {
    return;
  }
  let frame = faceFrame(face);
  let axes = array<vec3f, 2>(abs(frame.t), abs(frame.b));
  let ca = dot(local, axes[0]);
  let cb = dot(local, axes[1]);
  let nearest = min(min(ca, 1.0 - ca), min(cb, 1.0 - cb));
  let widest = max(p.bevel_width * (1.0 + 1.5 * p.bevel_jag), EDGE_BAND);
  if (nearest >= widest) {
    return;
  }
  let pos = vec3f(cell) + local;
  // Irregular radius along the edge: clipped two-octave noise, mean ≈ bevel_width.
  let n = fastNoise(pos * 3.0 + 41.0) * 0.7 + fastNoise(pos * 9.0 + 7.0) * 0.3;
  let width = p.bevel_width * max(1.0 + p.bevel_jag * (n - 0.5) * 5.0, 0.0);
  var tilt = vec3f(0.0);
  var wear = 0.0;   // outer edge proximity (0..1)
  var grime = 0.0;  // inside corner proximity (0..1)
  var checked_self = false;
  for (var a = 0; a < 2; a++) {
    let ax = axes[a];
    let c = dot(local, ax);
    for (var side = 0; side < 2; side++) {
      let d = select(c, 1.0 - c, side == 1);
      if (d >= widest) {
        continue;
      }
      let e = vec3i(ax * select(-1.0, 1.0, side == 1));
      if (!checked_self) {
        let own = getVoxel(cell);
        if (seeThrough(own) || ((p.bevel_enabled & 2u) != 0u && noBevel(own))) {
          return;
        }
        checked_self = true;
      }
      if (seeThrough(getVoxel(cell + e))) {
        // Exposed outer edge.
        let band = max(width, 1e-4);
        if (d < band) {
          let k = 1.0 - d / band;
          tilt += vec3f(e) * tan(0.785398 * k * k);
        }
        wear = max(wear, 1.0 - d / EDGE_BAND);
      } else if (!seeThrough(getVoxel(cell + e + vec3i(round(frame.n))))) {
        // Inside corner: this face meets a wall.
        grime = max(grime, 1.0 - d / EDGE_BAND);
      }
    }
  }
  if (wear <= 0.0 && grime <= 0.0) {
    return;
  }
  let q = pos * 48.0;
  if (any(tilt != vec3f(0.0))) {
    if (p.bevel_noise > 0.0) {
      // Rough edge: jitter the lean and its direction with ~2 cm noise.
      // Two lookups: both jitter components and the lean scale come from them.
      let j = vec2f(fastNoise(q), fastNoise(q + 5.3)) - 0.5;
      let jitter = vec3f(j.x, j.y, j.x - j.y);
      tilt = tilt * (1.0 + p.bevel_noise * (j.x + j.y)) + jitter * (2.0 * p.bevel_noise * length(tilt));
    }
    (*s).normal = normalize((*s).normal + tilt);
  }
  // Colour: dusty, lighter outer edges; darker inside corners. Never glossier.
  let dust = wear * wear * (0.6 + 0.4 * fastNoise(pos * 6.0 + 3.0));
  (*s).albedo *= (1.0 + p.edge_wear * dust) * (1.0 - p.edge_wear * 1.5 * grime * grime);
  (*s).roughness = min((*s).roughness + p.bevel_roughness * max(wear, grime), 1.0);
  if (p.bevel_chips > 0.0 && wear > 0.4) {
    // Chip marks: sparse small spots along exposed edges — darker, rougher, bumpier.
    let chip = smoothstep(0.72, 0.8, fastNoise(pos * 22.0 + 13.0)) * smoothstep(0.4, 0.8, wear);
    if (chip > 0.0) {
      (*s).albedo *= 1.0 - 0.45 * p.bevel_chips * chip;
      (*s).roughness = min((*s).roughness + 0.3 * chip, 1.0);
      let b = fastNoise(q + 23.0) - 0.5;
      let bump = vec3f(b, -b, b * 0.5);
      (*s).normal = normalize((*s).normal + bump * (1.2 * p.bevel_chips * chip));
    }
  }
}
