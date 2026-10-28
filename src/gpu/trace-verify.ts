import { config } from '../config';
import type { Vec3 } from '../core/math';
import { raycast } from '../world/raycast';
import { isShaped } from '../world/shapes';
import type { World } from '../world/world';
import { GpuBrickmap } from './brickmap';
import { createShaderModule } from './shader';
import { traceConstants } from './trace-constants';

export interface TraceVerifyResult {
  rays: number;
  /** Rays where GPU and CPU agree on hit/miss, cell, face normal, block and distance. */
  agree: number;
  hits: number;
  /** CPU hits on sub-voxel shaped blocks (shapes.ts). */
  shapedHits: number;
  ms: number;
  firstMismatch?: { origin: Vec3; dir: Vec3; cpu: string; gpu: string };
}

// RayIn: cell vec3i + pad, frac vec3f + pad, dir vec3f + max_t → 48 bytes; RayOut: cell,
// hit, normal, id, t + 3 pad → 48 bytes.
const RAY_IN_WORDS = 12;
const RAY_OUT_WORDS = 12;
/** Hit distances may differ by float32 rounding (GPU) vs float64 (CPU). */
const T_TOLERANCE = 1e-3;

/**
 * Debug check of the GPU hierarchical DDA: random rays from around `origin` are traced on
 * the GPU (trace.wgsl) and on the CPU (raycast.ts, one voxel at a time) and compared.
 */
export async function verifyTrace(device: GPUDevice, brickmap: GpuBrickmap, world: World, origin: Vec3): Promise<TraceVerifyResult> {
  const start = performance.now();
  brickmap.store.flush(world);
  brickmap.flushDistance();

  const n = config.trace.verifySamples;
  const maxT = config.trace.verifyMaxDistance;
  const input = new ArrayBuffer(n * RAY_IN_WORDS * 4);
  const i32 = new Int32Array(input);
  const f32 = new Float32Array(input);
  const origins: Vec3[] = [];
  const dirs: Vec3[] = [];
  for (let i = 0; i < n; i++) {
    // Origins jittered around the camera so rays start at arbitrary sub-voxel positions.
    const o: Vec3 = [
      origin[0] + (Math.random() - 0.5) * 8,
      origin[1] + (Math.random() - 0.5) * 8,
      origin[2] + (Math.random() - 0.5) * 8,
    ];
    // Uniform direction on the sphere.
    const z = Math.random() * 2 - 1;
    const a = Math.random() * 2 * Math.PI;
    const r = Math.sqrt(1 - z * z);
    const d: Vec3 = [r * Math.cos(a), z, r * Math.sin(a)];
    const cell: Vec3 = [Math.floor(o[0]), Math.floor(o[1]), Math.floor(o[2])];
    // Round the fraction to float32 first so both sides trace the exact same ray.
    const frac = Float32Array.from([o[0] - cell[0], o[1] - cell[1], o[2] - cell[2]]);
    const dir = Float32Array.from(d);
    i32.set(cell, i * RAY_IN_WORDS);
    f32.set(frac, i * RAY_IN_WORDS + 4);
    f32.set(dir, i * RAY_IN_WORDS + 8);
    f32[i * RAY_IN_WORDS + 11] = maxT;
    origins.push([cell[0] + frac[0]!, cell[1] + frac[1]!, cell[2] + frac[2]!]);
    dirs.push([dir[0]!, dir[1]!, dir[2]!]);
  }

  const workgroup = config.render.workgroupSize * config.render.workgroupSize;
  const module = await createShaderModule(device, 'trace-verify.wgsl');
  const pipeline = await device.createComputePipelineAsync({
    label: 'trace-verify',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
      // MICRO_WALK: the full tracer, shaped blocks on their micro grid (the fix-up path).
      constants: { WORKGROUP_SIZE: workgroup, MAX_STEPS: config.trace.verifyMaxSteps, ...traceConstants(), MICRO_WALK: 1 },
    },
  });
  const outBytes = n * RAY_OUT_WORDS * 4;
  const inBuf = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input);

  const encoder = device.createCommandEncoder({ label: 'trace-verify' });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuf } },
        { binding: 1, resource: { buffer: outBuf } },
      ],
    }),
  );
  pass.setBindGroup(1, brickmap.bindGroup(pipeline.getBindGroupLayout(1), GpuBrickmap.TRACE_BINDINGS));
  pass.dispatchWorkgroups(Math.ceil(n / workgroup));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const bytes = readBuf.getMappedRange().slice(0);
  const out = new Int32Array(bytes);
  const outT = new Float32Array(bytes);
  readBuf.unmap();
  for (const b of [inBuf, outBuf, readBuf]) b.destroy();

  const result: TraceVerifyResult = { rays: n, agree: 0, hits: 0, shapedHits: 0, ms: 0 };
  for (let i = 0; i < n; i++) {
    const o = i * RAY_OUT_WORDS;
    const gpuHit = out[o + 3] === 1;
    const gpu = gpuHit ? `${out[o]},${out[o + 1]},${out[o + 2]} n=${out[o + 4]},${out[o + 5]},${out[o + 6]} id=${out[o + 7]}` : 'miss';
    const h = raycast(world, origins[i]!, dirs[i]!, maxT);
    const cpu = h ? `${h.cell.join(',')} n=${h.normal.join(',')} id=${h.id}` : 'miss';
    if (h) result.hits++;
    if (h && isShaped(h.id)) result.shapedHits++;
    const dt = h && gpuHit ? Math.abs(outT[o + 8]! - h.t) : 0;
    if (cpu === gpu && dt <= T_TOLERANCE) result.agree++;
    else result.firstMismatch ??= { origin: origins[i]!, dir: dirs[i]!, cpu: `${cpu} t=${h?.t}`, gpu: `${gpu} t=${outT[o + 8]}` };
  }
  result.ms = performance.now() - start;
  return result;
}
