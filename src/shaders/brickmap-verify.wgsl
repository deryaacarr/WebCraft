// Debug check: evaluates getVoxel() at CPU-chosen positions so the results can be
// compared with World.getBlock() on the CPU.
#include "brickmap.wgsl"

override WORKGROUP_SIZE: u32 = 64u;

@group(0) @binding(0) var<storage, read> positions: array<vec4i>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&results)) {
    return;
  }
  results[id.x] = getVoxel(positions[id.x].xyz);
}
