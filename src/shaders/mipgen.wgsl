// Generates one mip level of a material texture array from the level above (2×2 box).
// `kind` selects the channel semantics (see material.wgsl):
//   0 albedo    sRGB colour averaged in linear light, weighted by opacity; opacity is
//               averaged then square-rooted so alpha-tested foliage keeps its coverage
//   1 normal    XY averaged (shorter = smoother at distance), AO and height averaged
//   2 specular  averaged; emission decoded (255 = none) before averaging

struct Params {
  kind: u32,
};

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d_array<f32>;
@group(0) @binding(2) var dst: texture_storage_2d_array<rgba8unorm, write>;

fn toLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(dst);
  if (any(gid.xy >= size)) {
    return;
  }
  let layer = gid.z;
  var t: array<vec4f, 4>;
  for (var i = 0u; i < 4u; i++) {
    t[i] = textureLoad(src, vec2i(gid.xy * 2u + vec2u(i & 1u, i >> 1u)), layer, 0);
  }

  var out = vec4f(0.0);
  switch (params.kind) {
    case 0u: {
      var color = vec3f(0.0);
      var weight = 0.0;
      var alpha = 0.0;
      for (var i = 0u; i < 4u; i++) {
        let w = t[i].a + 1e-3;
        color += toLinear(t[i].rgb) * w;
        weight += w;
        alpha += t[i].a;
      }
      out = vec4f(toSrgb(color / weight), sqrt(alpha * 0.25));
    }
    case 2u: {
      var sum = vec3f(0.0);
      var emission = 0.0;
      for (var i = 0u; i < 4u; i++) {
        sum += t[i].rgb;
        emission += select(t[i].a * 255.0 / 254.0, 0.0, t[i].a >= 1.0);
      }
      let e = emission * 0.25;
      out = vec4f(sum * 0.25, select(e * 254.0 / 255.0, 1.0, e <= 0.0));
    }
    default: {
      out = (t[0] + t[1] + t[2] + t[3]) * 0.25;
    }
  }
  textureStore(dst, vec2i(gid.xy), layer, out);
}
