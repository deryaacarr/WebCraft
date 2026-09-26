// Sky and light sources for the lighting passes. Bind SkySystem.bindGroup at @group(3).
// Every radiance returned here is pre-exposed (multiplied by the current exposure) so it
// stays inside rgba16float range from noon to night.
#include "atmosphere.wgsl"

struct SkyParams {
  sun_dir: vec3f,
  /// Illuminance of the sun at the top of the atmosphere (relative units).
  sun_illuminance: f32,
  moon_dir: vec3f,
  /// Illuminance of the moon (phase and night boost included).
  moon_illuminance: f32,
  /// Direction of the dominant light (sun by day, moon by night) for shadows and shading.
  light_dir: vec3f,
  /// Angular radius (radians) of the dominant light's disk: soft-shadow cone.
  light_radius: f32,
  /// Celestial pole direction and star-sphere rotation (radians).
  pole: vec3f,
  star_rotation: f32,
  sun_radius: f32,
  moon_radius: f32,
  /// Viewer distance from the planet centre (km).
  viewer_r: f32,
  star_brightness: f32,
  moon_albedo: f32,
  /// 1 when the dominant light is the moon.
  light_is_moon: f32,
  /// Debug fill only: sky ambient raised until direct : sky is at most this (0 = off).
  max_direct_to_sky: f32,
  _pad: f32,
};

/// Written by sky-ambient.wgsl each frame (not pre-exposed).
struct Lighting {
  /// Illuminance of the dominant light at the viewer (after the atmosphere).
  light_illuminance: vec3f,
  /// Direct : sky ratio from the atmosphere alone, before calibration.
  raw_direct_to_sky: f32,
  /// Irradiance from the whole sky on an upward-facing surface.
  sky_irradiance: vec3f,
  /// Direct : sky illuminance ratio on open horizontal ground after calibration.
  direct_to_sky: f32,
};

@group(3) @binding(0) var<uniform> sky: SkyParams;
@group(3) @binding(1) var<uniform> atm: AtmosphereParams;
@group(3) @binding(2) var transmittance_tex: texture_2d<f32>;
@group(3) @binding(3) var skyview_tex: texture_2d_array<f32>;
@group(3) @binding(4) var sky_sampler: sampler;
@group(3) @binding(5) var<storage, read> lighting: Lighting;
/// Current exposure (written by exposure.wgsl).
@group(3) @binding(6) var<storage, read> exposure: array<f32>;

fn preExposure() -> f32 {
  return exposure[0];
}

/// Transmittance from the viewer to space along `dir` (0 through the planet).
fn viewerTransmittance(dir: vec3f) -> vec3f {
  let ro = vec3f(0.0, sky.viewer_r, 0.0);
  if (raySphere(ro, dir, atm.bottom_radius) > 0.0) {
    return vec3f(0.0);
  }
  return textureSampleLevel(transmittance_tex, sky_sampler, transmittanceUv(atm, sky.viewer_r, dir.y), 0.0).rgb;
}

fn skyViewSample(dir: vec3f, light: vec3f, layer: u32) -> vec3f {
  // Azimuth between the view and light directions, around the vertical.
  let vh = dir.xz;
  let lh = light.xz;
  let lv = length(vh) * length(lh);
  let cos_az = select(1.0, dot(vh, lh) / lv, lv > 1e-6);
  let uv = skyViewUv(atm, sky.viewer_r, dir.y, acos(clamp(cos_az, -1.0, 1.0)));
  return textureSampleLevel(skyview_tex, sky_sampler, uv, layer, 0.0).rgb;
}

/// Scattered sky light (no disks, no stars), not pre-exposed.
fn skyScattering(dir: vec3f) -> vec3f {
  return skyViewSample(dir, sky.sun_dir, 0u) * sky.sun_illuminance + skyViewSample(dir, sky.moon_dir, 1u) * sky.moon_illuminance;
}

fn sunDisk(dir: vec3f) -> vec3f {
  let c = dot(dir, sky.sun_dir);
  let cos_r = cos(sky.sun_radius);
  if (c < cos_r) {
    return vec3f(0.0);
  }
  // Limb darkening (simple power law on the distance from the disk centre).
  let x = clamp((1.0 - c) / (1.0 - cos_r), 0.0, 1.0);
  let limb = pow(max(1.0 - x, 0.0), 0.3);
  let radiance = sky.sun_illuminance / (PI * sky.sun_radius * sky.sun_radius);
  return radiance * limb * viewerTransmittance(dir);
}

const EARTHSHINE: f32 = 0.05;

fn moonDisk(dir: vec3f) -> vec3f {
  let c = dot(dir, sky.moon_dir);
  let cos_r = cos(sky.moon_radius);
  if (c < cos_r) {
    return vec3f(0.0);
  }
  // Treat the disk as a sphere lit by the sun: the phase comes from the surface normal.
  let offset = (dir - sky.moon_dir * c) / sin(sky.moon_radius);
  let n = normalize(offset - sky.moon_dir * sqrt(max(1.0 - dot(offset, offset), 0.0)));
  let lit = max(dot(n, sky.sun_dir), 0.0);
  let radiance = sky.sun_illuminance * sky.moon_albedo / PI * lit;
  // Earthshine keeps the unlit part faintly visible at night.
  return (radiance + EARTHSHINE * sky.star_brightness) * viewerTransmittance(dir);
}

fn hash2(p: vec2u) -> u32 {
  var h = (p.x * 0x8da6b343u) ^ (p.y * 0xd8163841u);
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  h *= 0x846ca68bu;
  h ^= h >> 16u;
  return h;
}

/// Procedural star field fixed to the celestial sphere (rotates with the night).
fn stars(dir: vec3f) -> vec3f {
  if (dir.y <= 0.0 || sky.star_brightness <= 0.0) {
    return vec3f(0.0);
  }
  // Rotate into the star sphere's frame: angle around the celestial pole.
  let p = sky.pole;
  let a = -sky.star_rotation;
  let d = dir * cos(a) + cross(p, dir) * sin(a) + p * dot(p, dir) * (1.0 - cos(a));
  // Equal-area-ish grid on the sphere: longitude × sin(latitude).
  let cells = vec2f(1440.0, 720.0);
  let uv = vec2f(atan2(d.z, d.x) / (2.0 * PI) + 0.5, d.y * 0.5 + 0.5) * cells;
  let cell = vec2u(floor(uv));
  let h = hash2(cell);
  if ((h & 0xffu) > 6u) {
    return vec3f(0.0); // ~2.7 % of cells hold a star
  }
  let centre = vec2f(f32((h >> 8u) & 0xffu), f32((h >> 16u) & 0xffu)) / 255.0 * 0.6 + 0.2;
  let dist = length(fract(uv) - centre);
  let size = 0.12;
  if (dist > size) {
    return vec3f(0.0);
  }
  let magnitude = pow(f32(h >> 24u) / 255.0, 6.0) * 6.0 + 0.05;
  let tint = mix(vec3f(0.75, 0.85, 1.0), vec3f(1.0, 0.85, 0.7), f32((h >> 12u) & 1u));
  return tint * magnitude * sky.star_brightness * (1.0 - dist / size) * viewerTransmittance(dir);
}

/// Everything seen when a view ray escapes to the sky, pre-exposed.
fn skyRadiance(dir: vec3f) -> vec3f {
  return (skyScattering(dir) + sunDisk(dir) + moonDisk(dir) + stars(dir)) * preExposure();
}
