import { config } from '../config';
import { createShaderModule } from './shader';

// Region struct in brick-distance.wgsl: origin vec3i, max_distance u32, extent vec3u,
// pad, size vec3u, pad → 48 bytes.
const REGION_SIZE = 48;

const ENTRY_POINTS = ['pass_x', 'pass_y', 'pass_z'] as const;
/** Bindings each entry point uses ('auto' layouts omit the others). */
const USED_BINDINGS: Record<(typeof ENTRY_POINTS)[number], number[]> = {
  pass_x: [0, 1, 2], // region, grid → dist_x
  pass_y: [0, 2, 3], // region, dist_x → dist_xy
  pass_z: [0, 1, 3], // region, dist_xy → grid
};

/** Integer box of grid cells, max inclusive. */
export interface CellBox {
  min: [number, number, number];
  max: [number, number, number];
}

/**
 * Keeps the Chebyshev distance field in the brick grid up to date: after the CPU wrote
 * grid cells, `update()` recomputes the box around them (expanded by the max distance).
 */
export class BrickDistanceField {
  private readonly passes: GPUComputePipeline[] = [];
  private readonly bindGroups: GPUBindGroup[] = [];
  private readonly region: GPUBuffer;
  private readonly distX: GPUBuffer;
  private readonly distXY: GPUBuffer;
  private readonly regionData = new ArrayBuffer(REGION_SIZE);
  private readonly workgroup = config.trace.distanceWorkgroup;

  private constructor(
    private readonly device: GPUDevice,
    private readonly size: [number, number, number],
    grid: GPUBuffer,
    pipelines: GPUComputePipeline[],
  ) {
    const cells = size[0] * size[1] * size[2];
    const make = (label: string) =>
      device.createBuffer({ label, size: cells * 4, usage: GPUBufferUsage.STORAGE });
    this.distX = make('brick-distance-x');
    this.distXY = make('brick-distance-xy');
    this.region = device.createBuffer({
      label: 'brick-distance-region',
      size: REGION_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const buffers = [this.region, grid, this.distX, this.distXY];
    ENTRY_POINTS.forEach((entry, i) => {
      const pipeline = pipelines[i]!;
      this.passes.push(pipeline);
      this.bindGroups.push(
        device.createBindGroup({
          label: `brick-distance-${entry}`,
          layout: pipeline.getBindGroupLayout(0),
          entries: USED_BINDINGS[entry].map((binding) => ({ binding, resource: { buffer: buffers[binding]! } })),
        }),
      );
    });
  }

  static async create(device: GPUDevice, size: [number, number, number], grid: GPUBuffer): Promise<BrickDistanceField> {
    const module = await createShaderModule(device, 'brick-distance.wgsl');
    const constants = { WORKGROUP_SIZE: config.trace.distanceWorkgroup };
    const pipelines = await Promise.all(
      ENTRY_POINTS.map((entryPoint) =>
        device.createComputePipelineAsync({ label: `brick-distance-${entryPoint}`, layout: 'auto', compute: { module, entryPoint, constants } }),
      ),
    );
    return new BrickDistanceField(device, size, grid, pipelines);
  }

  /** Recomputes the distance field around `dirty` (cells written since the last update). */
  update(dirty: CellBox): void {
    const D = config.trace.distanceMax;
    const origin: [number, number, number] = [0, 0, 0];
    const extent: [number, number, number] = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      const lo = dirty.min[a]! - D;
      const hi = dirty.max[a]! + D;
      if (a === 1) {
        origin[a] = Math.max(0, lo);
        extent[a] = Math.min(this.size[a]! - 1, hi) - origin[a]! + 1;
      } else if (hi - lo + 1 >= this.size[a]!) {
        origin[a] = 0;
        extent[a] = this.size[a]!;
      } else {
        origin[a] = lo; // may be negative: the shader wraps X/Z
        extent[a] = hi - lo + 1;
      }
    }
    const i32 = new Int32Array(this.regionData);
    const u32 = new Uint32Array(this.regionData);
    i32.set(origin, 0);
    u32[3] = D;
    u32.set(extent, 4);
    u32.set(this.size, 8);
    this.device.queue.writeBuffer(this.region, 0, this.regionData);

    const encoder = this.device.createCommandEncoder({ label: 'brick-distance' });
    const pass = encoder.beginComputePass({ label: 'brick-distance' });
    const w = this.workgroup;
    for (let i = 0; i < this.passes.length; i++) {
      pass.setPipeline(this.passes[i]!);
      pass.setBindGroup(0, this.bindGroups[i]!);
      pass.dispatchWorkgroups(Math.ceil(extent[0] / w), Math.ceil(extent[1] / w), Math.ceil(extent[2] / w));
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
