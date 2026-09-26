// Physically based atmosphere after Hillaire 2020, "A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique". Distances in km; the planet centre is the origin
// and +Y is up at the viewer. Radiance is per unit of light illuminance (the LUTs are
// multiplied by the sun / moon illuminance when used).

const PI: f32 = 3.14159265358979;

struct AtmosphereParams {
  rayleigh_scattering: vec3f,
  bottom_radius: f32,
  mie_scattering: vec3f,
  top_radius: f32,
  mie_extinction: vec3f,
  mie_g: f32,
  ozone_absorption: vec3f,
  rayleigh_scale_height: f32,
  ground_albedo: vec3f,
  mie_scale_height: f32,
  ozone_center: f32,
  ozone_width: f32,
  _pad: vec2f,
};

struct Medium {
  rayleigh: vec3f,
  mie: vec3f,
  extinction: vec3f,
};

fn sampleMedium(atm: AtmosphereParams, height: f32) -> Medium {
  let h = max(height, 0.0);
  let rd = exp(-h / atm.rayleigh_scale_height);
  let md = exp(-h / atm.mie_scale_height);
  let od = max(0.0, 1.0 - abs(h - atm.ozone_center) / (atm.ozone_width * 0.5));
  var m: Medium;
  m.rayleigh = atm.rayleigh_scattering * rd;
  m.mie = atm.mie_scattering * md;
  m.extinction = m.rayleigh + atm.mie_extinction * md + atm.ozone_absorption * od;
  return m;
}

fn rayleighPhase(cos_theta: f32) -> f32 {
  return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

/// Cornette-Shanks phase function.
fn miePhase(g: f32, cos_theta: f32) -> f32 {
  let g2 = g * g;
  let k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
  return k * (1.0 + cos_theta * cos_theta) / pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4), 1.5);
}

/// Distance to the nearest positive intersection with a sphere at the origin, or −1.
fn raySphere(ro: vec3f, rd: vec3f, radius: f32) -> f32 {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - radius * radius;
  let disc = b * b - c;
  if (disc < 0.0) {
    return -1.0;
  }
  let s = sqrt(disc);
  let t0 = -b - s;
  let t1 = -b + s;
  if (t0 > 0.0) {
    return t0;
  }
  return select(-1.0, t1, t1 > 0.0);
}

// ---------------------------------------------------------------- transmittance LUT
// Bruneton's parametrisation: x = distance to the top of the atmosphere, y = height.

fn transmittanceUv(atm: AtmosphereParams, r: f32, mu: f32) -> vec2f {
  let h = sqrt(max(atm.top_radius * atm.top_radius - atm.bottom_radius * atm.bottom_radius, 0.0));
  let rho = sqrt(max(r * r - atm.bottom_radius * atm.bottom_radius, 0.0));
  let disc = r * r * (mu * mu - 1.0) + atm.top_radius * atm.top_radius;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let d_min = atm.top_radius - r;
  let d_max = rho + h;
  return vec2f((d - d_min) / (d_max - d_min), rho / h);
}

/// Inverse of transmittanceUv: (height from the centre r, cos of the view zenith angle).
fn transmittanceParams(atm: AtmosphereParams, uv: vec2f) -> vec2f {
  let h = sqrt(max(atm.top_radius * atm.top_radius - atm.bottom_radius * atm.bottom_radius, 0.0));
  let rho = h * uv.y;
  let r = sqrt(rho * rho + atm.bottom_radius * atm.bottom_radius);
  let d_min = atm.top_radius - r;
  let d_max = rho + h;
  let d = d_min + uv.x * (d_max - d_min);
  var mu = 1.0;
  if (d > 0.0) {
    mu = clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
  }
  return vec2f(r, mu);
}

// ---------------------------------------------------------------- multi-scattering LUT
// x = cos(sun zenith) mapped to [0, 1], y = height in the atmosphere.

fn multiScatteringUv(atm: AtmosphereParams, r: f32, sun_cos: f32) -> vec2f {
  return vec2f(sun_cos * 0.5 + 0.5, clamp((r - atm.bottom_radius) / (atm.top_radius - atm.bottom_radius), 0.0, 1.0));
}

// ---------------------------------------------------------------- sky-view LUT
// y: view zenith angle, non-linear so the horizon gets most texels (Hillaire §5.3);
// x: azimuth relative to the light, φ ∈ [0, π] (symmetric), denser towards the light.

fn skyViewUv(atm: AtmosphereParams, r: f32, view_zenith_cos: f32, azimuth: f32) -> vec2f {
  let v_horizon = sqrt(max(r * r - atm.bottom_radius * atm.bottom_radius, 0.0));
  let beta = acos(clamp(v_horizon / r, -1.0, 1.0));
  let zenith_horizon = PI - beta;
  let angle = acos(clamp(view_zenith_cos, -1.0, 1.0));
  var v: f32;
  if (angle < zenith_horizon) {
    v = (1.0 - sqrt(max(1.0 - angle / zenith_horizon, 0.0))) * 0.5;
  } else {
    v = sqrt(clamp((angle - zenith_horizon) / beta, 0.0, 1.0)) * 0.5 + 0.5;
  }
  return vec2f(sqrt(clamp(azimuth / PI, 0.0, 1.0)), v);
}

/// Inverse of skyViewUv: (view zenith angle, azimuth to the light).
fn skyViewAngles(atm: AtmosphereParams, r: f32, uv: vec2f) -> vec2f {
  let v_horizon = sqrt(max(r * r - atm.bottom_radius * atm.bottom_radius, 0.0));
  let beta = acos(clamp(v_horizon / r, -1.0, 1.0));
  let zenith_horizon = PI - beta;
  var angle: f32;
  if (uv.y < 0.5) {
    let c = 1.0 - 2.0 * uv.y;
    angle = zenith_horizon * (1.0 - c * c);
  } else {
    let c = uv.y * 2.0 - 1.0;
    angle = zenith_horizon + beta * c * c;
  }
  return vec2f(angle, uv.x * uv.x * PI);
}
