// Debug view: looks straight down on the brickmap around the player.
// Each pixel scans its column from the top, skipping empty bricks in one step.
#include "brickmap.wgsl"
#include "color.wgsl"

struct Params {
  size: vec2u,
  /// World XZ at the centre of the screen.
  center: vec2f,
  blocks_per_pixel: f32,
  min_y: i32,
  max_y: i32,
  block_count: u32,
  water_id: u32,
};

override WORKGROUP_SIZE: u32 = 8u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;
/// Colour per block id in xyz (display/sRGB values; converted to linear on output).
@group(0) @binding(2) var<storage, read> block_colors: array<vec4f>;

// Debug-view shading only; the real renderer replaces this view.
const LIGHT_DIR: vec3f = vec3f(-1.0, 1.5, -1.0);
const AMBIENT: f32 = 0.35;
const DIFFUSE: f32 = 0.75;
/// Water absorption per block of depth.
const WATER_ABSORPTION: f32 = 0.12;
const MARKER_HALF_SIZE: f32 = 3.0;
const NEIGHBOR_SCAN_BRICKS: i32 = 2;

struct Hit {
  y: i32,
  id: u32,
};

/// Topmost non-air voxel of column (x, z) below `from_y`; id 0 if none.
fn scanDown(x: i32, z: i32, from_y: i32) -> Hit {
  let bs = brickSize();
  let bx = x >> BRICK_BITS;
  let bz = z >> BRICK_BITS;
  let lx = u32(x & (bs - 1));
  let lz = u32(z & (bs - 1));
  var by = from_y >> BRICK_BITS;
  var ly = from_y & (bs - 1);
  let min_by = params.min_y >> BRICK_BITS;
  loop {
    if (by < min_by) {
      break;
    }
    let ptr = brickPointer(vec3i(bx, by, bz));
    if (!brickIsEmpty(ptr)) {
      for (var y = ly; y >= 0; y--) {
        let id = brickVoxel(ptr, vec3u(lx, u32(y), lz));
        if (id != 0u) {
          return Hit(by * bs + y, id);
        }
      }
    }
    by -= 1;
    ly = bs - 1;
  }
  return Hit(params.min_y, 0u);
}

fn colorOf(id: u32) -> vec3f {
  return block_colors[min(id, params.block_count - 1u)].xyz;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= params.size)) {
    return;
  }
  let offset = (vec2f(gid.xy) + 0.5 - vec2f(params.size) * 0.5) * params.blocks_per_pixel;
  let p = vec2i(floor(params.center + offset));
  let top = params.max_y - 1;

  let hit = scanDown(p.x, p.y, top);
  if (hit.id == 0u) {
    textureStore(output, vec2i(gid.xy), vec4f(0.02, 0.02, 0.03, 1.0));
    return;
  }

  // Hillshade from the neighbouring columns (light from the north-west). Their scan starts
  // a little above this hit instead of at the world top: much cheaper, and slopes steeper
  // than NEIGHBOR_SCAN_BRICKS bricks per block are merely clamped for shading.
  let neighbor_top = min(top, hit.y + NEIGHBOR_SCAN_BRICKS * brickSize());
  let hx = scanDown(p.x + 1, p.y, neighbor_top).y;
  let hz = scanDown(p.x, p.y + 1, neighbor_top).y;
  let n = normalize(vec3f(f32(hit.y - hx), 2.0, f32(hit.y - hz)));
  var color = colorOf(hit.id) * (AMBIENT + DIFFUSE * max(dot(n, normalize(LIGHT_DIR)), 0.0));

  // Water: tint the ground below by depth.
  if (hit.id == params.water_id) {
    var below = scanDown(p.x, p.y, hit.y - 1);
    while (below.id == params.water_id) {
      below = scanDown(p.x, p.y, below.y - 1);
    }
    let t = 1.0 - exp(-f32(hit.y - below.y) * WATER_ABSORPTION);
    color = mix(colorOf(below.id) * DIFFUSE, colorOf(params.water_id), t);
  }

  // Player marker.
  let d = abs(vec2f(gid.xy) + 0.5 - vec2f(params.size) * 0.5);
  if (max(d.x, d.y) < MARKER_HALF_SIZE) {
    color = vec3f(1.0, 0.1, 0.1);
  }
  textureStore(output, vec2i(gid.xy), vec4f(srgbToLinear(color), 1.0));
}
