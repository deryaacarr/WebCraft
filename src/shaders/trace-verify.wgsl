// Debug check: traces CPU-chosen rays so the hits can be compared with raycast.ts.
#include "trace.wgsl"

struct RayIn {
  cell: vec3i,
  _pad0: i32,
  frac: vec3f,
  _pad1: f32,
  dir: vec3f,
  max_t: f32,
};

struct RayOut {
  cell: vec3i,
  hit: u32,
  normal: vec3i,
  id: u32,
};

override WORKGROUP_SIZE: u32 = 64u;
override MAX_STEPS: u32 = 512u;

@group(0) @binding(0) var<storage, read> rays: array<RayIn>;
@group(0) @binding(1) var<storage, read_write> results: array<RayOut>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&results)) {
    return;
  }
  let ray = rays[id.x];
  let r = traceRay(ray.cell, ray.frac, ray.dir, ray.max_t, MAX_STEPS);
  results[id.x] = RayOut(r.cell, select(0u, 1u, r.hit), r.normal, r.id);
}
