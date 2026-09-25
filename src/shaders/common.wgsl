const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

// Cosine palette (Inigo Quilez): a + b * cos(TAU * (c * t + d)).
fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(TAU * (c * t + d));
}
