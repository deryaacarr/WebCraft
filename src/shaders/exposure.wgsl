// Automatic exposure (eye adaptation). Measures the log-average luminance of the
// pre-exposed HDR frame, divides the previous exposure out to get scene luminance and
// moves the exposure towards key / luminance in log space with an exponential rate.

struct ExposureParams {
  dt: f32,
  key: f32,
  compensation_ev: f32,
  speed: f32,
  min_exposure: f32,
  max_exposure: f32,
  _pad: vec2f,
};

const THREADS: u32 = 256u;
const GRID: vec2u = vec2u(64u, 36u);

@group(0) @binding(0) var<uniform> params: ExposureParams;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> exposure: array<f32>;

var<workgroup> partial: array<f32, THREADS>;

@compute @workgroup_size(THREADS)
fn main(@builtin(local_invocation_index) i: u32) {
  let size = vec2f(textureDimensions(hdr));
  var sum = 0.0;
  let count = GRID.x * GRID.y;
  for (var k = i; k < count; k += THREADS) {
    let cell = vec2f(f32(k % GRID.x), f32(k / GRID.x)) + 0.5;
    let c = textureLoad(hdr, vec2i(cell / vec2f(GRID) * size), 0).rgb;
    sum += log2(max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-6));
  }
  partial[i] = sum;
  workgroupBarrier();
  for (var s = THREADS / 2u; s > 0u; s >>= 1u) {
    if (i < s) {
      partial[i] += partial[i + s];
    }
    workgroupBarrier();
  }
  if (i == 0u) {
    let previous = exposure[0];
    let scene = exp2(partial[0] / f32(count)) / previous;
    let goal = clamp(params.key * exp2(params.compensation_ev) / scene, params.min_exposure, params.max_exposure);
    let blend = 1.0 - exp(-params.dt * params.speed);
    exposure[0] = exp2(mix(log2(previous), log2(goal), blend));
  }
}
