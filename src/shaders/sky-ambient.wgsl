// Per-frame lighting summary: illuminance of the dominant light at the viewer and the
// sky's irradiance on an upward-facing surface (cosine-weighted hemisphere integral of
// the sky-view LUT). One workgroup; the result feeds every lit pixel.
#include "sky.wgsl"

const SAMPLES_THETA: u32 = 8u;
const SAMPLES_PHI: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> lighting_out: Lighting;

var<workgroup> partial: array<vec3f, 128>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) i: u32) {
  // Stratified cosine-weighted directions: each sample carries weight π / N.
  let j = i / SAMPLES_PHI;
  let k = i % SAMPLES_PHI;
  let u = (f32(j) + 0.5) / f32(SAMPLES_THETA);
  let v = (f32(k) + 0.5) / f32(SAMPLES_PHI);
  let sin_t = sqrt(u);
  let phi = 2.0 * PI * v;
  let dir = vec3f(sin_t * cos(phi), sqrt(1.0 - u), sin_t * sin(phi));
  partial[i] = skyScattering(dir) * (PI / f32(SAMPLES_THETA * SAMPLES_PHI));
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (i < s) {
      partial[i] += partial[i + s];
    }
    workgroupBarrier();
  }
  if (i == 0u) {
    let e = select(sky.sun_illuminance, sky.moon_illuminance, sky.light_is_moon > 0.5);
    lighting_out.light_illuminance = e * viewerTransmittance(sky.light_dir);
    lighting_out.sky_irradiance = partial[0];
  }
}
