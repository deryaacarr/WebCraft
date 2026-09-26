// Direct lighting of the G-buffer: dominant light with GGX specular + Burley diffuse,
// sky ambient scaled by sky visibility, emission, leaf translucency; sky pixels show the
// atmosphere, sun, moon and stars. Output is pre-exposed HDR (see sky.wgsl).
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "sky.wgsl"
#include "color.wgsl"

struct LightingParams {
  /// 0 lit, 1 shadow (sun visibility), 2 sky visibility, 3 history length (debug views).
  mode: u32,
  emissive_strength: f32,
  subsurface: f32,
  /// Debug-fill comparison only (0 in the physical model): minimum sky visibility,
  /// bounce colour and the share of it arriving from all directions.
  ambient_floor: f32,
  bounce_albedo: vec3f,
  bounce_isotropic: f32,
  /// History length shown as fully "hot" in the history view.
  history_max: f32,
  /// Aerial perspective LUT: depth of its last slice (km) and on/off.
  aerial_max_depth_km: f32,
  aerial_enabled: u32,
  _pad: f32,
};

override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;
const F0_DIELECTRIC: f32 = 0.04;
const MIN_ROUGHNESS: f32 = 0.03;
/// World units (blocks) are metres; the atmosphere works in km.
const KM_PER_UNIT: f32 = 0.001;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: LightingParams;
@group(0) @binding(2) var gbuffer0: texture_2d<u32>;
@group(0) @binding(3) var visibility: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var aerial_tex: texture_3d<f32>;
@group(0) @binding(6) var gdepth: texture_2d<f32>;

/// Aerial perspective between the camera and a surface at view depth `depth` (world units):
/// rgb = in-scattered light (pre-exposed), a = transmittance.
fn aerialPerspective(uv: vec2f, depth: f32) -> vec4f {
  let slices = f32(textureDimensions(aerial_tex).z);
  // Texel s holds the path to depth max · ((s + 1) / N)² (see aerial-perspective.wgsl).
  let s = sqrt(depth * KM_PER_UNIT / params.aerial_max_depth_km) * slices - 1.0;
  let a = textureSampleLevel(aerial_tex, sky_sampler, vec3f(uv, (max(s, 0.0) + 0.5) / slices), 0.0);
  // Closer than the first slice: fade towards no atmosphere at the camera.
  let fade = clamp(s + 1.0, 0.0, 1.0);
  return vec4f(a.rgb * fade, mix(1.0, a.a, fade));
}

fn ggxD(n_h: f32, a2: f32) -> f32 {
  let d = n_h * n_h * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

/// Height-correlated Smith visibility term V = G / (4 n·l n·v).
fn smithV(n_l: f32, n_v: f32, a2: f32) -> f32 {
  let gv = n_l * sqrt(n_v * n_v * (1.0 - a2) + a2);
  let gl = n_v * sqrt(n_l * n_l * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn fresnel(f0: vec3f, v_h: f32) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(1.0 - v_h, 5.0);
}

/// Disney / Burley diffuse (without the 1/π, applied by the caller).
fn burley(n_l: f32, n_v: f32, l_h: f32, roughness: f32) -> f32 {
  let f90 = 0.5 + 2.0 * roughness * l_h * l_h;
  let fl = 1.0 + (f90 - 1.0) * pow(1.0 - n_l, 5.0);
  let fv = 1.0 + (f90 - 1.0) * pow(1.0 - n_v, 5.0);
  return fl * fv;
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= cam.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  let g = textureLoad(gbuffer0, px, 0);
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(cam.size);
  let view = rayDir(cam.inv_view_proj, uv);
  let vis = textureLoad(visibility, px, 0).rg;

  if (params.mode == 3u) {
    // History length: blue = fresh (1 frame), red = full history.
    let frames = textureLoad(visibility, px, 0).a;
    let t = clamp((frames - 1.0) / max(params.history_max - 1.0, 1.0), 0.0, 1.0);
    let heat = mix(mix(vec3f(0.0, 0.1, 0.9), vec3f(0.1, 0.9, 0.2), clamp(t * 2.0, 0.0, 1.0)), vec3f(1.0, 0.1, 0.0), clamp(t * 2.0 - 1.0, 0.0, 1.0));
    textureStore(output, px, vec4f(select(srgbToLinear(heat), vec3f(0.0), gbIsSky(g.w)), 1.0));
    return;
  }
  if (params.mode != 0u) {
    let v = select(vis.y, vis.x, params.mode == 1u);
    textureStore(output, px, vec4f(select(vec3f(v), vec3f(0.0), gbIsSky(g.w)), 1.0));
    return;
  }
  if (gbIsSky(g.w)) {
    textureStore(output, px, vec4f(skyRadiance(view), 1.0));
    return;
  }

  let albedo_ao = unpack4x8unorm(g.x);
  let albedo = albedo_ao.xyz;
  let ao = albedo_ao.w;
  let n = octDecode(unpack2x16snorm(g.y));
  let surf = unpack4x8unorm(g.z); // roughness, metalness, emission, subsurface
  let roughness = max(surf.x, MIN_ROUGHNESS);
  let metal = surf.y;
  let a2 = roughness * roughness * roughness * roughness;

  let v = -view;
  let l = sky.light_dir;
  let h = normalize(v + l);
  let n_v = max(dot(n, v), 1e-4);
  let n_l_raw = dot(n, l);
  let n_l = max(n_l_raw, 0.0);
  let n_h = max(dot(n, h), 0.0);
  let l_h = max(dot(l, h), 0.0);
  let f0 = mix(vec3f(F0_DIELECTRIC), albedo, metal);
  let f = fresnel(f0, max(dot(v, h), 0.0));
  let diffuse_color = albedo * (1.0 - metal);

  // Dominant light, shadowed by the accumulated sun visibility.
  let e_light = lighting.light_illuminance * vis.x;
  let specular = f * ggxD(n_h, a2) * smithV(n_l, n_v, a2);
  let diffuse = (vec3f(1.0) - f) * diffuse_color / PI * burley(n_l, n_v, l_h, roughness);
  var color = (diffuse + specular) * e_light * n_l;

  // Leaves: light scattered through from behind (wrapped, tinted by the albedo).
  let back = max(-n_l_raw, 0.0);
  color += diffuse_color * surf.w * params.subsurface * back * e_light / PI;

  // Sky light from above, occluded by the traced sky visibility. The floor and bounce
  // terms are zero unless the non-physical debug-fill model is selected.
  let facing = 0.5 + 0.5 * n.y;
  let sky_vis = max(vis.y, params.ambient_floor);
  let e_sky = lighting.sky_irradiance;
  let e_bounce = params.bounce_albedo * (lighting.light_illuminance * max(l.y, 0.0) + e_sky);
  let bounce_weight = mix(1.0 - facing, 1.0, params.bounce_isotropic);
  color += diffuse_color / PI * (e_sky * facing * sky_vis + e_bounce * bounce_weight * sky_vis) * ao;
  // Leaves also let some sky light through from the other side.
  color += diffuse_color * surf.w * params.subsurface * e_sky * sky_vis * 0.5 / PI;
  // Specular sky reflection (unshadowed beyond sky visibility; no GI yet).
  let r = reflect(view, n);
  color += f * skyScattering(r) * vis.y * (1.0 - roughness);

  color += albedo * surf.z * params.emissive_strength;
  var result = color * preExposure();
  if (params.aerial_enabled != 0u) {
    let air = aerialPerspective(uv, textureLoad(gdepth, px, 0).x);
    result = result * air.a + air.rgb;
  }
  textureStore(output, px, vec4f(result, 1.0));
}
