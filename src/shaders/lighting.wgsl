// Shading of the G-buffer: dominant light with GGX specular + Burley diffuse, emission,
// leaf translucency, and indirect light — with GI on, the denoised half-resolution GI
// (gi-denoise.wgsl) upsampled with depth / face awareness; with GI off, the approximate
// sky ambient scaled by sky visibility. Sky pixels show the atmosphere, sun, moon and
// stars. Output is pre-exposed HDR (see sky.wgsl).
//
// Pixels seen through water (water.wgsl): the G-buffer holds the bottom, lit as usual but
// with the light attenuated on its way down through the water (refracted sun: Fresnel
// transmission and Beer-Lambert along its path; indirect light along a mean diffuse path).
// waterComposite then adds the water itself: Fresnel reflection, absorption along the view
// path and single scattering of sun and sky light in it. The surface has no albedo.
#include "gbuffer.wgsl"
#include "camera.wgsl"
#include "sky.wgsl"
#include "color.wgsl"
#include "gi-common.wgsl"
// Worn edges (applyBevel) are shaded here rather than in the primary pass: in the
// traversal shader the extra code cost ~0.5 ms through register pressure alone.
#include "material.wgsl"
#include "water.wgsl"
#include "water-shore.wgsl"

struct LightingParams {
  /// 0 lit, 1 shadow (sun visibility), 2 sky visibility, 3 history length, 4 indirect
  /// light only (GI) — debug views.
  mode: u32,
  /// Seconds (flicker of emitted light).
  time: f32,
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
  /// 1: indirect light from the GI textures (replaces the sky-ambient approximation).
  gi_enabled: u32,
  /// Specular rays threshold (GiParams.spec_threshold): decides which GI specular to use.
  gi_spec_threshold: f32,
  /// Same-surface tolerance between face planes (GiParams.plane_tolerance).
  gi_plane_tolerance: f32,
  /// Flicker of emitted light (config.lighting.flicker).
  flicker_amount: f32,
  flicker_speed: f32,
  flicker_scale: f32,
  /// GI grid divisor (GiParams.scale).
  gi_scale: f32,
  _pad1: f32,
  _pad2: f32,
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
/// Half-resolution GI: irradiance (diffuse), radiance (specular), guide (gi-common.wgsl).
@group(0) @binding(7) var gi_diffuse: texture_2d<f32>;
@group(0) @binding(8) var gi_specular: texture_2d<f32>;
@group(0) @binding(9) var gi_guide: texture_2d<u32>;
@group(0) @binding(10) var gi_emissive: texture_2d<f32>;
/// Emitted radiance per block id: rgb = mean over a face, a = emissive coverage of it.
@group(0) @binding(11) var<storage, read> block_radiance: array<vec4f>;
@group(0) @binding(12) var gwater: texture_2d<f32>;
@group(0) @binding(13) var<uniform> water: WaterParams;
/// Ray-traced water reflections at reduced resolution (water-reflection.wgsl): rgb
/// radiance (pre-exposed), a = sun visibility at the surface, a < 0 = no water there.
@group(0) @binding(14) var water_reflection: texture_2d<f32>;

/// Mean path length of diffuse (sky and bounce) light per metre of depth under water: it
/// arrives within Snell's window, mostly steeply.
const WATER_DIFFUSE_PATH: f32 = 1.2;
/// Fresnel reflectance of the water surface averaged over a uniform sky (light lost to
/// reflection before it can enter the water).
const WATER_SKY_FRESNEL: f32 = 0.066;

struct GiSample {
  diffuse: vec3f,
  specular: vec3f,
  /// Emissive direct light (ReSTIR DI) without flicker.
  emissive: vec3f,
};

fn giFetch(t: vec2i) -> GiSample {
  return GiSample(textureLoad(gi_diffuse, t, 0).rgb, textureLoad(gi_specular, t, 0).rgb, textureLoad(gi_emissive, t, 0).rgb);
}

fn giScaled(a: GiSample, w: f32) -> GiSample {
  return GiSample(a.diffuse * w, a.specular * w, a.emissive * w);
}

fn giSum(a: GiSample, b: GiSample) -> GiSample {
  return GiSample(a.diffuse + b.diffuse, a.specular + b.specular, a.emissive + b.emissive);
}

/// Joint bilateral upsampling of the half-res GI: the 2×2 half-res pixels around this one,
/// bilinear weights, only those on the same face plane. When none is (faces thinner than
/// the half-res grid), the same-plane pixels of the 3×3 around; failing that, the mean of
/// all surface pixels there — never a single pixel of another surface, which showed as
/// bright streaks on thin staircase faces.
fn giUpsample(px: vec2i, full: vec2u, depth: f32, face: u32) -> GiSample {
  let half = vec2i(textureDimensions(gi_guide));
  let plane_p = dot(faceNormal(face), pixelPosition(vec2u(px), full, depth));
  let tolerance = params.gi_plane_tolerance;
  let f = vec2f(px) / params.gi_scale;
  let base = vec2i(floor(f));
  let fr = f - vec2f(base);
  let zero = GiSample(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var sum = zero;
  var wsum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let t = min(base + o, half - 1);
    if (!onPlane(textureLoad(gi_guide, t, 0), face, plane_p, tolerance)) {
      continue;
    }
    let w = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1) + 1e-3;
    sum = giSum(sum, giScaled(giFetch(t), w));
    wsum += w;
  }
  if (wsum > 0.0) {
    return giScaled(sum, 1.0 / wsum);
  }
  var same = zero;
  var n_same = 0.0;
  var any_surface = zero;
  var n_any = 0.0;
  for (var k = 0; k < 9; k++) {
    let t = base + vec2i(k % 3 - 1, k / 3 - 1);
    if (any(t < vec2i(0)) || any(t >= half)) {
      continue;
    }
    let g = textureLoad(gi_guide, t, 0);
    if (gbIsSky(g.z)) {
      continue;
    }
    let v = giFetch(t);
    any_surface = giSum(any_surface, v);
    n_any += 1.0;
    if (onPlane(g, face, plane_p, tolerance)) {
      same = giSum(same, v);
      n_same += 1.0;
    }
  }
  if (n_same > 0.0) {
    return giScaled(same, 1.0 / n_same);
  }
  if (n_any > 0.0) {
    return giScaled(any_surface, 1.0 / n_any);
  }
  return zero;
}

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

/// Directional albedo of the specular lobe for environment light (Karis 2014 analytic fit
/// of the split-sum BRDF term): unlike Schlick's F(n·v) it stays low at grazing angles on
/// rough surfaces (plain Fresnel turned edge-on rock faces into bright mirrors).
fn envBrdf(f0: vec3f, roughness: f32, n_v: f32) -> vec3f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}

/// ∫₀^path exp(−a·s) ds per channel, a = σ_t · (1 + k): light scattered along a view path
/// whose depth grows k times as fast (relative to the path) as the light's own path.
fn waterInscatter(sigma_t: vec3f, k: f32, path: f32) -> vec3f {
  let a = sigma_t * (1.0 + k);
  return (1.0 - exp(-a * path)) / max(a, vec3f(1e-6));
}

/// Upsampled water reflection at full-res pixel `px`: bilinear over the reduced-resolution
/// texels around it that lie on water; a < 0 if none does.
fn waterReflection(px: vec2i) -> vec4f {
  let size = vec2i(textureDimensions(water_reflection));
  let scale = f32(cam.size.x) / f32(size.x);
  let c = (vec2f(px) + 0.5) / scale - 0.5;
  let base = vec2i(floor(c));
  let fr = c - vec2f(base);
  var sum = vec4f(0.0);
  var wsum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let v = textureLoad(water_reflection, clamp(base + o, vec2i(0), size - 1), 0);
    if (v.a < 0.0) {
      continue;
    }
    let w = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1) + 1e-3;
    sum += v * w;
    wsum += w;
  }
  return select(vec4f(0.0, 0.0, 0.0, -1.0), sum / wsum, wsum > 0.0);
}

/// Water over the pre-exposed radiance `bottom` leaving the bottom (zero for deep water):
/// Fresnel-weighted reflection (traced, upsampled; the sun's highlight analytically, GGX of
/// roughness sun_roughness, shadowed by the reflection pass's sun visibility at the
/// surface) and transmission, Beer-Lambert absorption along the view path and single
/// scattering (isotropic phase) of the refracted sun (`sun_vis`: shadowed at the bottom)
/// and the sky light under the surface, both attenuated with depth.
fn waterComposite(px: vec2i, view: vec3f, w: vec4f, bottom: vec3f, sun_vis: f32, depth: f32) -> vec3f {
  let n_s = waterSurfaceNormal(w);
  let f_v = fresnelDielectric(dot(-view, n_s), water.ior);
  let d = refract(view, n_s, 1.0 / water.ior);
  let medium = waterMedium(depth);
  let sigma_t = medium.extinction;
  let path = w.y;
  // Depth gained per metre along the refracted view ray.
  let mu_v = max(-dot(d, n_s), 0.0);
  var scatter = vec3f(0.0);
  let cos_l = dot(sky.light_dir, n_s);
  if (cos_l > 0.0) {
    let mu_l = max(dot(waterLightDir(sky.light_dir, n_s, water.ior), n_s), 1e-3);
    // Perpendicular irradiance of the refracted beam (narrower: × cos θ_air / cos θ_water).
    let e_sun = lighting.light_illuminance * (1.0 - fresnelDielectric(cos_l, water.ior)) * cos_l / mu_l * sun_vis;
    scatter += e_sun * waterInscatter(sigma_t, mu_v / mu_l, path);
  }
  // Diffuse light: scalar irradiance ≈ planar irradiance × the mean path factor.
  let e_sky = lighting.sky_irradiance * (1.0 - WATER_SKY_FRESNEL) * WATER_DIFFUSE_PATH;
  scatter += e_sky * waterInscatter(sigma_t, mu_v * WATER_DIFFUSE_PATH, path);
  let transmitted = bottom * exp(-sigma_t * path) + medium.scatter / (4.0 * PI) * scatter * preExposure();
  let traced = waterReflection(px);
  var reflected = traced.rgb;
  if (traced.a < 0.0) {
    // No traced texel nearby (thin water at the reduced resolution): the sky.
    reflected = skyScattering(reflect(view, n_s)) * preExposure();
  }
  var glint = vec3f(0.0);
  if (cos_l > 0.0) {
    let v = -view;
    let h = normalize(v + sky.light_dir);
    let a = water.sun_roughness * water.sun_roughness;
    let a2 = a * a;
    let n_v = max(dot(n_s, v), 1e-4);
    let spec = fresnelDielectric(dot(v, h), water.ior) * ggxD(max(dot(n_s, h), 0.0), a2) * smithV(cos_l, n_v, a2);
    glint = spec * lighting.light_illuminance * cos_l * select(1.0, traced.a, traced.a >= 0.0) * preExposure();
  }
  let water_color = mix(transmitted, reflected, f_v) + glint;
  // Foam where the surface meets the shore: a white surface lit by the sun and the sky.
  let foam = waterFoam(cam.origin_frac + view * w.x, water.time);
  if (foam <= 0.0) {
    return water_color;
  }
  let e = lighting.light_illuminance * max(cos_l, 0.0) * sun_vis + lighting.sky_irradiance;
  return mix(water_color, FOAM_ALBEDO / PI * e * preExposure(), foam);
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
  if (params.mode != 0u && params.mode != 4u) {
    let v = select(vis.y, vis.x, params.mode == 1u);
    textureStore(output, px, vec4f(select(vec3f(v), vec3f(0.0), gbIsSky(g.w)), 1.0));
    return;
  }
  // Seen through water (primary.wgsl only flags it while water is on): the bottom, or
  // nothing within the trace distance (deep water, flagged as sky).
  let under = gbThroughWater(g.w);
  if (gbIsSky(g.w) && !under) {
    textureStore(output, px, vec4f(skyRadiance(view), 1.0));
    return;
  }
  var wsample = vec4f(0.0);
  if (under) {
    wsample = textureLoad(gwater, px, 0);
  }
  var result = vec3f(0.0);
  var sun_vis = 1.0;
  /// Water above the bottom (metres); deep water (no bottom hit): from the path length.
  var water_depth = wsample.y * 0.7;
  if (!gbIsSky(g.w)) {
    let albedo_ao = unpack4x8unorm(g.x);
    let surf = unpack4x8unorm(g.z); // roughness, metalness, emission, subsurface
    let ao = albedo_ao.w;
    // Worn edges: the hit block and the position inside it from the G-buffer depth (a hair
    // inside the face, so the cell is the hit block's).
    var worn: Surface;
    worn.albedo = albedo_ao.xyz;
    worn.normal = octDecode(unpack2x16snorm(g.y));
    worn.roughness = surf.x;
    let face_id = gbFace(g.w);
    let depth = textureLoad(gdepth, px, 0).x;
    // Seen through water: the true bottom position and the refracted view ray.
    let pos = select(viewRelativePosition(view, cam.forward, depth), waterBottom(view, wsample, water.ior), under);
    let rel = cam.origin_frac + pos - faceNormal(face_id) * 1e-3;
    let cell_rel = vec3i(floor(rel));
    let dist = select(depth / max(dot(view, cam.forward), 1e-4), wsample.x + wsample.y, under);
    let mat_flags = materials[gbMaterial(g.w)].flags;
    applyBevel(&worn, mat_flags, cam.origin_cell + cell_rel, face_id, rel - vec3f(cell_rel), dist);
    if (!under && (mat_flags & MATERIAL_NATURAL) != 0u) {
      // Ground beside the water is wet.
      applyWetness(&worn, shoreWetness(cam.origin_cell + cell_rel, dist));
    }
    let albedo = worn.albedo;
    let n = worn.normal;
    let roughness = max(worn.roughness, MIN_ROUGHNESS);
    let metal = surf.y;
    let a2 = roughness * roughness * roughness * roughness;

    var v = -view;
    var l = sky.light_dir;
    // Light reaching a bottom under water (1 elsewhere): direct and indirect.
    var att_direct = vec3f(1.0);
    var att_indirect = vec3f(1.0);
    if (under) {
      let n_s = waterSurfaceNormal(wsample);
      v = -refract(view, n_s, 1.0 / water.ior);
      water_depth = max(view.y * wsample.x - pos.y, 0.0);
      let sigma_t = waterMedium(water_depth).extinction;
      let cos_l = dot(sky.light_dir, n_s);
      l = waterLightDir(sky.light_dir, n_s, water.ior);
      // The beam narrows as it bends towards the normal: its perpendicular irradiance grows
      // by cos θ_air / cos θ_water (the flux through the surface is E · cos θ_air). Lighting
      // the bottom with the refracted cosine alone made it ~4× too bright under a low sun.
      let mu_w = max(dot(l, n_s), 1e-3);
      let caustics = waterCaustics(vec2f(cam.origin_cell.xz) + cam.origin_frac.xz + pos.xz, water_depth, water.time);
      att_direct = (1.0 - fresnelDielectric(cos_l, water.ior)) * max(cos_l, 0.0) / mu_w * caustics * exp(-sigma_t * water_depth / mu_w);
      att_indirect = (1.0 - WATER_SKY_FRESNEL) * exp(-sigma_t * water_depth * WATER_DIFFUSE_PATH);
    }
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
    let e_light = lighting.light_illuminance * vis.x * att_direct;
    let specular = f * ggxD(n_h, a2) * smithV(n_l, n_v, a2);
    let diffuse = (vec3f(1.0) - f) * diffuse_color / PI * burley(n_l, n_v, l_h, roughness);
    var color = (diffuse + specular) * e_light * n_l;

    // Leaves: light scattered through from behind (wrapped, tinted by the albedo).
    let back = max(-n_l_raw, 0.0);
    color += diffuse_color * surf.w * params.subsurface * back * e_light / PI;

    // World position (flicker field): camera cell + fraction + camera-relative offset.
    let world = vec3f(cam.origin_cell) + cam.origin_frac + pos;
    let flick = flicker(world, params.time, params.flicker_amount, params.flicker_speed, params.flicker_scale);
    // Emission: the block's radiance concentrated on its glowing texels (surf.z = their
    // emission), so a torch flame is much brighter than the face average.
    let emitter = block_radiance[gbBlockId(g.w)];
    color += emitter.rgb / max(emitter.a, 1e-3) * surf.z * flick;
    result = color * preExposure();

    // Indirect light (pre-exposed).
    var indirect = vec3f(0.0);
    if (params.gi_enabled != 0u) {
      // GI: sky, bounce and emissive light, occluded by real geometry. Replaces the
      // approximation below entirely (adding both would wash the image out).
      let gi = giUpsample(px, cam.size, textureLoad(gdepth, px, 0).x, gbFace(g.w));
      let f_env = envBrdf(f0, roughness, n_v);
      // Smooth surfaces traced their specular lobe; rough ones take it from the irradiance.
      let smooth_spec = specularProbability(surf.x, params.gi_spec_threshold) > 0.0;
      let spec_in = select((gi.diffuse + gi.emissive * flick) / PI, gi.specular, smooth_spec);
      // Emissive light flickers here (it was accumulated and denoised without flicker).
      let e_diffuse = gi.diffuse + gi.emissive * flick;
      indirect = (vec3f(1.0) - f_env) * diffuse_color / PI * e_diffuse * ao + f_env * spec_in;
    } else {
      // Approximation without GI: sky light from above, occluded by the traced sky
      // visibility. The floor and bounce terms are zero unless the non-physical debug-fill
      // model is selected.
      let facing = 0.5 + 0.5 * n.y;
      let sky_vis = max(vis.y, params.ambient_floor);
      let e_sky = lighting.sky_irradiance;
      let e_bounce = params.bounce_albedo * (lighting.light_illuminance * max(l.y, 0.0) + e_sky);
      let bounce_weight = mix(1.0 - facing, 1.0, params.bounce_isotropic);
      var ambient = diffuse_color / PI * (e_sky * facing * sky_vis + e_bounce * bounce_weight * sky_vis) * ao;
      // Specular sky reflection (unshadowed beyond sky visibility).
      let r = reflect(-v, n);
      ambient += f * skyScattering(r) * vis.y * (1.0 - roughness);
      indirect = ambient * preExposure();
    }
    indirect *= att_indirect;
    if (params.mode == 4u) {
      result = vec3f(0.0);
    }
    result += indirect;
    sun_vis = vis.x;
  }
  if (under && params.mode == 0u) {
    result = waterComposite(px, view, wsample, result, sun_vis, water_depth);
  }
  if (params.aerial_enabled != 0u) {
    let air_depth = select(textureLoad(gdepth, px, 0).x, wsample.x * dot(view, cam.forward), under);
    let air = aerialPerspective(uv, air_depth);
    result = result * air.a + air.rgb;
  }
  textureStore(output, px, vec4f(result, 1.0));
}
