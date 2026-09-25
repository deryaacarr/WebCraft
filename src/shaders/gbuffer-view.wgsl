// Turns the G-buffer into a picture: a simple sun-lit view or one channel for debugging.
#include "gbuffer.wgsl"
#include "color.wgsl"

struct ViewParams {
  /// Direction towards the sun (normalised).
  sun_dir: vec3f,
  mode: u32,
  size: vec2u,
  far: f32,
  max_steps: u32,
  /// Motion view: UV delta multiplied by this before display.
  motion_scale: f32,
  /// Depth view: depth at which the grey ramp reaches 50 %.
  depth_half: f32,
  _pad: vec2f,
};

// Must match DEBUG_VIEWS order in config.ts.
const MODE_LIT: u32 = 0u;
const MODE_ALBEDO: u32 = 1u;
const MODE_NORMAL: u32 = 2u;
const MODE_DEPTH: u32 = 3u;
const MODE_STEPS: u32 = 4u;
const MODE_MOTION: u32 = 5u;
const MODE_ROUGHNESS: u32 = 6u;
const MODE_AO: u32 = 7u;
const MODE_MATERIAL: u32 = 8u;

// Placeholder lighting until the path tracer exists (linear colours).
const SKY_ZENITH: vec3f = vec3f(0.18, 0.36, 0.75);
const SKY_HORIZON: vec3f = vec3f(0.65, 0.75, 0.9);
const AMBIENT: f32 = 0.3;
/// Emission multiplier for the placeholder lit view.
const EMISSION_GAIN: f32 = 4.0;

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var<uniform> params: ViewParams;
@group(0) @binding(1) var gbuffer0: texture_2d<u32>;
@group(0) @binding(2) var gdepth: texture_2d<f32>;
@group(0) @binding(3) var gmotion: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

/// Blue → cyan → green → yellow → red, t in [0, 1].
fn heat(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0) * 4.0;
  let stops = array<vec3f, 5>(
    vec3f(0.0, 0.0, 0.6), vec3f(0.0, 0.8, 1.0), vec3f(0.1, 0.9, 0.1), vec3f(1.0, 0.9, 0.0), vec3f(1.0, 0.0, 0.0),
  );
  let i = min(u32(x), 3u);
  return mix(stops[i], stops[i + 1u], x - f32(i));
}

fn hashColor(v: u32) -> vec3f {
  var h = v * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  return vec3f(f32(h & 0xffu), f32((h >> 8u) & 0xffu), f32((h >> 16u) & 0xffu)) / 255.0;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= params.size)) {
    return;
  }
  let px = vec2i(gid.xy);
  let g = textureLoad(gbuffer0, px, 0);
  let depth = textureLoad(gdepth, px, 0).x;
  let motion = textureLoad(gmotion, px, 0).xy;
  let sky = gbIsSky(g.w);
  let albedo_ao = unpack4x8unorm(g.x);
  let albedo = albedo_ao.xyz;
  let ao = albedo_ao.w;
  let n = octDecode(unpack2x16snorm(g.y));
  let surface = unpack4x8unorm(g.z); // roughness, metalness, emission, subsurface
  let v = f32(gid.y) / f32(params.size.y);

  // `display` views show data directly (sRGB-encoded values); decode so the blit's
  // encode reproduces them exactly.
  var display = vec3f(0.0);
  var linear = vec3f(0.0);
  var is_display = true;
  switch (params.mode) {
    case MODE_LIT: {
      is_display = false;
      if (sky) {
        linear = mix(SKY_ZENITH, SKY_HORIZON, v);
      } else {
        let diffuse = AMBIENT * ao + (1.0 - AMBIENT) * max(dot(n, params.sun_dir), 0.0);
        linear = albedo * diffuse + albedo * surface.z * EMISSION_GAIN;
      }
    }
    case MODE_ALBEDO: {
      is_display = false;
      linear = albedo;
    }
    case MODE_NORMAL: {
      display = select(n * 0.5 + 0.5, vec3f(0.0), sky);
    }
    case MODE_DEPTH: {
      display = select(vec3f(1.0 - depth / (depth + params.depth_half)), vec3f(0.0), sky);
    }
    case MODE_STEPS: {
      display = heat(f32(gbSteps(g.w)) / f32(params.max_steps));
    }
    case MODE_MOTION: {
      display = vec3f(motion * params.motion_scale + 0.5, 0.5);
    }
    case MODE_ROUGHNESS: {
      display = select(surface.xyz, vec3f(0.0), sky);
    }
    case MODE_AO: {
      display = select(vec3f(ao), vec3f(0.0), sky);
    }
    case MODE_MATERIAL: {
      display = select(hashColor(gbMaterial(g.w) + 1u), vec3f(0.0), sky);
    }
    default: {
      display = vec3f(1.0, 0.0, 1.0);
    }
  }
  let color = select(linear, srgbToLinear(display), is_display);
  textureStore(output, px, vec4f(color, 1.0));
}
