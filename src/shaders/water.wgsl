// Water as a transparent medium (no surface albedo). The primary fix-up (primary.wgsl)
// refracts rays entering a water surface and traces on to the bottom, so the G-buffer holds
// the bottom and every pass lights it as usual; the lighting pass then composites the
// water: Fresnel-weighted reflection + transmission, Beer-Lambert absorption and single
// scattering along the path in water.
//
// gwater (rgba32float, cleared every frame), for pixels whose ray bends at a water surface:
//   x  distance along the camera ray to the surface (0 = no bend)
//   y  path length after the bend to the G-buffer surface; the trace distance if nothing
//      was hit (the G-buffer says sky)
//   z  surface normal (pointing out of the water), octahedral pack2x16snorm (u32 bits)
//   w  WATER_FROM_AIR: camera above, refracted into the water; camera under water
//      (primary UNDERWATER pipeline): WATER_EXIT refracted out into the air, WATER_TIR
//      mirrored back down (total internal reflection); WATER_FOG: no bend and nothing
//      hit within the trace distance (x = 0, y = that distance).
// The G-buffer depth of such a pixel is where the camera ray meets the plane of the hit
// face (exact plane coordinates for the denoisers, continuous across a face); rays that
// start at the surface (shadow, GI) take the true bottom position from waterBottom.
#include "gbuffer.wgsl"

/// BlockId.water (src/world/blocks.ts; a test checks they match).
const WATER_ID: u32 = 7u;

const WATER_FROM_AIR: f32 = 0.0;
const WATER_EXIT: f32 = 1.0;
const WATER_TIR: f32 = 2.0;
const WATER_FOG: f32 = 3.0;

struct WaterParams {
  /// Absorption and scattering coefficients per block (metre), linear rgb. Extinction is
  /// their sum; `scatter_albedo` is the share of the scattered light that is not absorbed
  /// by the particles (the water's own colour: suspended matter takes the red).
  absorption: vec3f,
  ior: f32,
  scattering: vec3f,
  /// Refracted rays give up after this distance in water (deep water).
  max_distance: f32,
  scatter_albedo: vec3f,
  /// Shadows on the bottom soften in turbid water: light reaching a point at depth d came
  /// through the surface within a disk of radius shadow_blur · mean scattering · d².
  shadow_blur: f32,
  /// The bottom's detail blurs with the path through the water: its texture footprint
  /// grows by detail_blur · mean scattering · path² (metres).
  detail_blur: f32,
  /// 0: water is drawn as an opaque block (the old look).
  enabled: u32,
  /// Roughness of the sun's highlight on the surface (micro-ripples).
  sun_roughness: f32,
  _pad: f32,
  /// Waves (water-waves.wgsl): unit direction they travel with the wind (xz), swell height
  /// scale (metres), seconds.
  wind_dir: vec2f,
  wave_amplitude: f32,
  time: f32,
  /// Longest swell wavelength (metres), ripple strength, rest level of the surface inside
  /// the top water voxel (0..1).
  wavelength: f32,
  ripple: f32,
  level: f32,
  _pad2: f32,
  /// Shore (water-shore.wgsl): sediment scattering per metre at zero depth, falling off
  /// over turbidity_depth metres; foam strength and strip width (metres).
  turbidity: f32,
  turbidity_depth: f32,
  foam: f32,
  foam_width: f32,
  /// Sediment's scattering colour (linear albedo); caustics strength.
  sediment: vec3f,
  caustics: f32,
  /// Wet ground beside the water: strength, distance limit (blocks).
  wet: f32,
  wet_max_distance: f32,
  _pad3: f32,
  _pad4: f32,
  /// Camera under water (Aşama 9e; primary UNDERWATER pipeline): 1 / 0, and the rest level
  /// of the surface above it (camera-cell frame, y).
  camera_under: u32,
  camera_surface: f32,
  /// Wetness (wetness.wgsl): global (rain: up-facing ground open to the sky), puddle
  /// amount, puddle size (metres), local zones in use.
  wet_global: f32,
  puddles: f32,
  puddle_scale: f32,
  zone_count: u32,
  /// Light shafts under water (0 = off).
  light_shafts: f32,
  _pad5: f32,
  /// Local wet zones: centre (absolute xyz) and radius; their amounts.
  zones: array<vec4f, 4>,
  zone_amount: vec4f,
};

fn waterMeanScattering(w: WaterParams) -> f32 {
  return dot(w.scattering, vec3f(1.0 / 3.0));
}

fn waterExtinction(w: WaterParams) -> vec3f {
  return w.absorption + w.scattering;
}

fn throughWater(g: vec4f) -> bool {
  return g.x > 0.0;
}

fn waterSurfaceNormal(g: vec4f) -> vec3f {
  return octDecode(unpack2x16snorm(bitcast<u32>(g.z)));
}

fn packWaterSample(surface_t: f32, path: f32, normal: vec3f, kind: f32) -> vec4f {
  return vec4f(surface_t, path, bitcast<f32>(pack2x16snorm(octEncode(normal))), kind);
}

/// Direction of the ray after its bend at the surface (see gwater).
fn waterBentDir(view: vec3f, g: vec4f, ior: f32) -> vec3f {
  let n = waterSurfaceNormal(g);
  if (g.w == WATER_FROM_AIR) {
    return refract(view, n, 1.0 / ior);
  }
  if (g.w == WATER_EXIT) {
    // Right at the critical angle the packed normal can put the ray just beyond it: it
    // leaves along the surface then (a zero direction made NaNs, a mirrored one sampled the
    // sky below the horizon — black lines along the edge of Snell's window).
    let out_dir = refract(view, -n, ior);
    if (all(out_dir == vec3f(0.0))) {
      return normalize(view - n * dot(view, n) + n * 1e-3);
    }
    return out_dir;
  }
  return reflect(view, -n);
}

/// Camera-relative position of what is seen at view direction `view` behind a bend.
fn waterBottom(view: vec3f, g: vec4f, ior: f32) -> vec3f {
  return view * g.x + waterBentDir(view, g, ior) * g.y;
}

/// Direction towards the light from under a surface with normal `n` (pointing out of the
/// water), for light arriving from `l` above it; zero if the light is below the surface.
fn waterLightDir(l: vec3f, n: vec3f, ior: f32) -> vec3f {
  if (dot(l, n) <= 0.0) {
    return vec3f(0.0);
  }
  return -refract(-l, n, 1.0 / ior);
}

/// Unpolarised Fresnel reflectance of a dielectric interface (Walter et al. 2007), cosine
/// of the incidence angle `cos_i`, relative index `eta` = n_transmitted / n_incident.
/// 1 under total internal reflection.
fn fresnelDielectric(cos_i: f32, eta: f32) -> f32 {
  let c = clamp(cos_i, 0.0, 1.0);
  let g2 = eta * eta - 1.0 + c * c;
  if (g2 < 0.0) {
    return 1.0;
  }
  let g = sqrt(g2);
  let a = (g - c) / (g + c);
  let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return 0.5 * a * a * (1.0 + b * b);
}
