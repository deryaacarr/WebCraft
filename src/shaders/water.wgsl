// Water as a transparent medium (no surface albedo). The primary fix-up (primary.wgsl)
// refracts rays entering a water surface and traces on to the bottom, so the G-buffer holds
// the bottom and every pass lights it as usual; the lighting pass then composites the
// water: Fresnel-weighted reflection + transmission, Beer-Lambert absorption and single
// scattering along the path in water.
//
// gwater (rgba32float, cleared every frame), only for pixels seen through a water surface:
//   x  distance along the camera ray to the surface (0 = not through water)
//   y  path length in water from the surface to the bottom (the G-buffer surface); the
//      trace distance if nothing was hit (deep water: the G-buffer says sky)
//   z  surface normal, octahedral pack2x16snorm (u32 bits)
//   w  unused
// The G-buffer depth of such a pixel is where the camera ray meets the plane of the hit
// face (exact plane coordinates for the denoisers, continuous across a face); rays that
// start at the surface (shadow, GI) take the true bottom position from waterBottom.
#include "gbuffer.wgsl"

/// BlockId.water (src/world/blocks.ts; a test checks they match).
const WATER_ID: u32 = 7u;

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

fn packWaterSample(surface_t: f32, path: f32, normal: vec3f) -> vec4f {
  return vec4f(surface_t, path, bitcast<f32>(pack2x16snorm(octEncode(normal))), 0.0);
}

/// Camera-relative position of the bottom seen through the water at view direction `view`.
fn waterBottom(view: vec3f, g: vec4f, ior: f32) -> vec3f {
  let d = refract(view, waterSurfaceNormal(g), 1.0 / ior);
  return view * g.x + d * g.y;
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
