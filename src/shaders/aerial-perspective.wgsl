// Aerial perspective LUT (Hillaire 2020 §5.5): a camera-aligned froxel volume, x/y = screen
// uv, z = view depth slice (quadratic). Each froxel holds the light scattered into the view
// ray between the camera and that depth (rgb, pre-exposed, sun + moon) and the mean
// transmittance over the same path (a). The lighting pass applies
// L = L_surface · T + S. Terrain shadowing of the in-scattering is not modelled.
#include "sky.wgsl"
#include "camera.wgsl"

struct AerialParams {
  /// View depth (km) of the last slice.
  max_depth_km: f32,
  samples_per_slice: u32,
  _pad: vec2f,
};

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: AerialParams;
@group(0) @binding(2) var multiscatter_tex: texture_2d<f32>;
@group(0) @binding(3) var aerial_out: texture_storage_3d<rgba16float, write>;

/// Transmittance from `pos` to space towards `dir` (0 if the planet is in the way).
fn transmittanceTo(pos: vec3f, dir: vec3f) -> vec3f {
  if (raySphere(pos, dir, atm.bottom_radius) > 0.0) {
    return vec3f(0.0);
  }
  let r = length(pos);
  return textureSampleLevel(transmittance_tex, sky_sampler, transmittanceUv(atm, r, dot(pos / r, dir)), 0.0).rgb;
}

/// Single + multiple scattering towards the viewer at `pos` from one light, per unit illuminance.
fn inScattering(pos: vec3f, view: vec3f, light: vec3f, m: Medium) -> vec3f {
  let cos_theta = dot(view, light);
  let single = (m.rayleigh * rayleighPhase(cos_theta) + m.mie * miePhase(atm.mie_g, cos_theta)) * transmittanceTo(pos, light);
  let r = length(pos);
  let uv = multiScatteringUv(atm, r, dot(pos / r, light));
  let multi = (m.rayleigh + m.mie) * textureSampleLevel(multiscatter_tex, sky_sampler, uv, 0.0).rgb;
  return single + multi;
}

/// View depth (km) at the far end of slice `s` (quadratic spacing).
fn sliceDepth(s: f32, slices: f32) -> f32 {
  let x = s / slices;
  return params.max_depth_km * x * x;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(aerial_out);
  if (any(gid.xy >= size.xy)) {
    return;
  }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(size.xy);
  let dir = rayDir(cam.inv_view_proj, uv);
  // View depth → distance along this ray.
  let depth_to_ray = 1.0 / max(dot(dir, cam.forward), 1e-3);
  let ro = vec3f(0.0, sky.viewer_r, 0.0);
  let slices = f32(size.z);
  let steps = max(params.samples_per_slice, 1u);

  var throughput = vec3f(1.0);
  var lum = vec3f(0.0);
  var t = 0.0;
  for (var s = 0u; s < size.z; s++) {
    let t_end = sliceDepth(f32(s + 1u), slices) * depth_to_ray;
    let dt = (t_end - t) / f32(steps);
    for (var k = 0u; k < steps; k++) {
      let pos = ro + dir * (t + (f32(k) + 0.5) * dt);
      let m = sampleMedium(atm, length(pos) - atm.bottom_radius);
      let scattered = inScattering(pos, dir, sky.sun_dir, m) * sky.sun_illuminance +
        inScattering(pos, dir, sky.moon_dir, m) * sky.moon_illuminance;
      let step_t = exp(-m.extinction * dt);
      // Energy-conserving integration over the step (as in the sky-view LUT).
      lum += throughput * scattered * (vec3f(1.0) - step_t) / max(m.extinction, vec3f(1e-6));
      throughput *= step_t;
    }
    t = t_end;
    let mean_t = dot(throughput, vec3f(1.0 / 3.0));
    textureStore(aerial_out, vec3u(gid.xy, s), vec4f(lum * preExposure(), mean_t));
  }
}
