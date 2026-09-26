// Atmosphere look-up tables (Hillaire 2020):
//   transmittance_lut  transmittance to the top of the atmosphere per (height, zenith cos)
//   multiscatter_lut   isotropic multiple-scattering contribution Ψ_ms per (height, sun cos)
//   skyview_lut        sky radiance around the viewer per (view zenith, azimuth to light);
//                      layer 0 lit by the sun, layer 1 by the moon
// The first two only depend on the atmosphere; the sky view changes with the lights.
#include "atmosphere.wgsl"

struct LutParams {
  sun_dir: vec3f,
  /// Viewer distance from the planet centre (km).
  viewer_r: f32,
  moon_dir: vec3f,
  _pad: f32,
};

override WORKGROUP_SIZE: u32 = 8u;
const TRANSMITTANCE_STEPS: u32 = 40u;
const MULTISCATTER_DIRECTIONS: u32 = 8u; // squared → 64 directions over the sphere
const MULTISCATTER_STEPS: u32 = 20u;
const SKYVIEW_STEPS: u32 = 30u;

@group(0) @binding(0) var<uniform> atm: AtmosphereParams;
@group(0) @binding(1) var<uniform> lut: LutParams;
@group(0) @binding(2) var transmittance_tex: texture_2d<f32>;
@group(0) @binding(3) var multiscatter_tex: texture_2d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;
@group(0) @binding(5) var transmittance_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var multiscatter_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var skyview_out: texture_storage_2d_array<rgba16float, write>;

/// Transmittance from a point at radius `r` towards direction with zenith cos `mu`,
/// zero if the planet is in the way.
fn transmittanceTo(pos: vec3f, dir: vec3f) -> vec3f {
  if (raySphere(pos, dir, atm.bottom_radius) > 0.0) {
    return vec3f(0.0);
  }
  let r = length(pos);
  let mu = dot(pos / r, dir);
  return textureSampleLevel(transmittance_tex, lut_sampler, transmittanceUv(atm, r, mu), 0.0).rgb;
}

fn multiScattering(pos: vec3f, light: vec3f) -> vec3f {
  let r = length(pos);
  let uv = multiScatteringUv(atm, r, dot(pos / r, light));
  return textureSampleLevel(multiscatter_tex, lut_sampler, uv, 0.0).rgb;
}

/// Distance the ray travels through the atmosphere (to the ground or the top).
fn atmosphereDistance(ro: vec3f, rd: vec3f) -> vec2f {
  let ground = raySphere(ro, rd, atm.bottom_radius);
  let top = raySphere(ro, rd, atm.top_radius);
  // y = 1 when the ray ends on the ground.
  if (ground > 0.0) {
    return vec2f(ground, 1.0);
  }
  return vec2f(max(top, 0.0), 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn transmittance_lut(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(transmittance_out);
  if (any(gid.xy >= size)) {
    return;
  }
  let p = transmittanceParams(atm, (vec2f(gid.xy) + 0.5) / vec2f(size));
  let ro = vec3f(0.0, p.x, 0.0);
  let rd = vec3f(sqrt(max(1.0 - p.y * p.y, 0.0)), p.y, 0.0);
  let dist = max(raySphere(ro, rd, atm.top_radius), 0.0);
  let dt = dist / f32(TRANSMITTANCE_STEPS);
  var depth = vec3f(0.0);
  for (var i = 0u; i < TRANSMITTANCE_STEPS; i++) {
    let pos = ro + rd * ((f32(i) + 0.5) * dt);
    depth += sampleMedium(atm, length(pos) - atm.bottom_radius).extinction * dt;
  }
  textureStore(transmittance_out, vec2i(gid.xy), vec4f(exp(-depth), 1.0));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn multiscatter_lut(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(multiscatter_out);
  if (any(gid.xy >= size)) {
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
  let sun_cos = uv.x * 2.0 - 1.0;
  let r = mix(atm.bottom_radius + 1e-3, atm.top_radius - 1e-3, uv.y);
  let ro = vec3f(0.0, r, 0.0);
  let light = vec3f(sqrt(max(1.0 - sun_cos * sun_cos, 0.0)), sun_cos, 0.0);
  let iso = 1.0 / (4.0 * PI);

  var l2 = vec3f(0.0);
  var fms = vec3f(0.0);
  let n = MULTISCATTER_DIRECTIONS;
  for (var j = 0u; j < n; j++) {
    for (var i = 0u; i < n; i++) {
      // Uniform directions on the sphere (stratified).
      let z = 1.0 - 2.0 * (f32(j) + 0.5) / f32(n);
      let phi = 2.0 * PI * (f32(i) + 0.5) / f32(n);
      let s = sqrt(max(1.0 - z * z, 0.0));
      let rd = vec3f(s * cos(phi), z, s * sin(phi));
      let end = atmosphereDistance(ro, rd);
      let dt = end.x / f32(MULTISCATTER_STEPS);
      var throughput = vec3f(1.0);
      var lum = vec3f(0.0);
      var ms = vec3f(0.0);
      for (var k = 0u; k < MULTISCATTER_STEPS; k++) {
        let pos = ro + rd * ((f32(k) + 0.5) * dt);
        let m = sampleMedium(atm, length(pos) - atm.bottom_radius);
        let step_t = exp(-m.extinction * dt);
        let scattering = m.rayleigh + m.mie;
        // Energy-conserving integration of in-scattering over the step.
        let integ = (vec3f(1.0) - step_t) / max(m.extinction, vec3f(1e-6));
        lum += throughput * scattering * transmittanceTo(pos, light) * iso * integ;
        ms += throughput * scattering * integ;
        throughput *= step_t;
      }
      if (end.y > 0.5) {
        // Light bounced off the ground (Lambertian).
        let pos = ro + rd * end.x;
        let nrm = normalize(pos);
        lum += throughput * transmittanceTo(pos, light) * max(dot(nrm, light), 0.0) * atm.ground_albedo / PI;
      }
      l2 += lum;
      fms += ms;
    }
  }
  // Average over the sphere with the isotropic phase (4π/N · 1/(4π) = 1/N).
  let count = f32(n * n);
  l2 /= count;
  fms = fms * iso * 4.0 * PI / count;
  let psi = l2 / max(vec3f(1.0) - fms, vec3f(1e-3));
  textureStore(multiscatter_out, vec2i(gid.xy), vec4f(psi, 1.0));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn skyview_lut(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(skyview_out);
  if (any(gid.xy >= size)) {
    return;
  }
  let world_light = select(lut.sun_dir, lut.moon_dir, gid.z == 1u);
  let r = lut.viewer_r;
  let ro = vec3f(0.0, r, 0.0);
  // Light in the LUT's frame: azimuth 0, same zenith angle as in the world.
  let light = vec3f(sqrt(max(1.0 - world_light.y * world_light.y, 0.0)), world_light.y, 0.0);
  let angles = skyViewAngles(atm, r, (vec2f(gid.xy) + 0.5) / vec2f(size));
  let st = sin(angles.x);
  let rd = vec3f(st * cos(angles.y), cos(angles.x), st * sin(angles.y));
  let cos_theta = dot(rd, light);
  let phase_r = rayleighPhase(cos_theta);
  let phase_m = miePhase(atm.mie_g, cos_theta);

  let end = atmosphereDistance(ro, rd);
  var throughput = vec3f(1.0);
  var lum = vec3f(0.0);
  let n = f32(SKYVIEW_STEPS);
  for (var k = 0u; k < SKYVIEW_STEPS; k++) {
    // Quadratic step distribution: more samples near the viewer.
    let t0 = end.x * (f32(k) / n) * (f32(k) / n);
    let t1 = end.x * (f32(k + 1u) / n) * (f32(k + 1u) / n);
    let dt = t1 - t0;
    let pos = ro + rd * (0.5 * (t0 + t1));
    let m = sampleMedium(atm, length(pos) - atm.bottom_radius);
    let step_t = exp(-m.extinction * dt);
    let single = (m.rayleigh * phase_r + m.mie * phase_m) * transmittanceTo(pos, light);
    let multi = (m.rayleigh + m.mie) * multiScattering(pos, light);
    let integ = (vec3f(1.0) - step_t) / max(m.extinction, vec3f(1e-6));
    lum += throughput * (single + multi) * integ;
    throughput *= step_t;
  }
  if (end.y > 0.5) {
    // The planet's surface beyond the loaded world: Lambertian ground seen through the air.
    let ground = ro + rd * end.x;
    let nrm = normalize(ground);
    lum += throughput * transmittanceTo(ground + nrm * 1e-3, light) * max(dot(nrm, light), 0.0) * atm.ground_albedo / PI;
  }
  textureStore(skyview_out, vec2i(gid.xy), gid.z, vec4f(lum, 1.0));
}
