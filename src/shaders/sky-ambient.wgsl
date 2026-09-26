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
    let direct = e * viewerTransmittance(sky.light_dir);
    let lum = vec3f(0.2126, 0.7152, 0.0722);
    let ratio = dot(direct, lum) * max(sky.light_dir.y, 0.0) / max(dot(partial[0], lum), 1e-9);
    // Physical model: max_direct_to_sky = 0, the atmosphere's own irradiance. The
    // debug-fill comparison raises it to the given ratio (never lowers it).
    var gain = 1.0;
    if (sky.max_direct_to_sky > 0.0 && ratio > sky.max_direct_to_sky) {
      gain = ratio / sky.max_direct_to_sky;
    }
    lighting_out.light_illuminance = direct;
    lighting_out.raw_direct_to_sky = ratio;
    // The added light stands in for what the sky model lacks (clouds, haze and terrain
    // bounce, all lit by the sun), so it takes the direct light's colour, not the blue sky's.
    let tint = direct / max(dot(direct, lum), 1e-9);
    lighting_out.sky_irradiance = partial[0] + tint * dot(partial[0], lum) * (gain - 1.0);
    lighting_out.direct_to_sky = ratio / gain;
  }
}
