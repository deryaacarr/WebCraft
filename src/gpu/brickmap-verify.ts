import { config } from '../config';
import { CHUNK_SIZE, chunkOrigin } from '../world/coords';
import type { World } from '../world/world';
import type { GpuBrickmap } from './brickmap';
import { createShaderModule } from './shader';

export interface VerifyResult {
  samples: number;
  mismatches: number;
  /** Samples whose expected block was not air (shows the check hit real terrain). */
  solid: number;
  ms: number;
  firstMismatch?: { x: number; y: number; z: number; cpu: number; gpu: number };
}

/**
 * Debug check that the GPU brickmap mirrors the CPU world: samples random voxels in
 * resident chunks, runs getVoxel() on the GPU and compares with World.getBlock().
 */
export async function verifyBrickmap(device: GPUDevice, brickmap: GpuBrickmap, world: World): Promise<VerifyResult> {
  const start = performance.now();
  brickmap.store.flush(world);
  brickmap.flushDistance(); // GPU must be up to date for a fair comparison

  const chunks = [...world.chunkValues()];
  if (chunks.length === 0) throw new Error('no chunks loaded');
  const n = config.brickmap.verifySamples;
  const positions = new Int32Array(n * 4);
  const expected = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const c = chunks[Math.floor(Math.random() * chunks.length)]!;
    const x = chunkOrigin(c.cx) + Math.floor(Math.random() * CHUNK_SIZE);
    const y = chunkOrigin(c.cy) + Math.floor(Math.random() * CHUNK_SIZE);
    const z = chunkOrigin(c.cz) + Math.floor(Math.random() * CHUNK_SIZE);
    positions.set([x, y, z, 0], i * 4);
    expected[i] = world.getBlock(x, y, z);
  }

  const workgroup = config.render.workgroupSize * config.render.workgroupSize;
  const module = await createShaderModule(device, 'brickmap-verify.wgsl');
  const pipeline = await device.createComputePipelineAsync({
    label: 'brickmap-verify',
    layout: 'auto',
    compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: workgroup, BRICK_BITS: config.world.brickBits } },
  });
  const posBuf = device.createBuffer({ size: positions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const resBuf = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(posBuf, 0, positions);

  const encoder = device.createCommandEncoder({ label: 'brickmap-verify' });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: posBuf } },
        { binding: 1, resource: { buffer: resBuf } },
      ],
    }),
  );
  pass.setBindGroup(1, brickmap.bindGroup(pipeline.getBindGroupLayout(1)));
  pass.dispatchWorkgroups(Math.ceil(n / workgroup));
  pass.end();
  encoder.copyBufferToBuffer(resBuf, 0, readBuf, 0, expected.byteLength);
  device.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const gpu = new Uint32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  for (const b of [posBuf, resBuf, readBuf]) b.destroy();

  const result: VerifyResult = { samples: n, mismatches: 0, solid: 0, ms: 0 };
  for (let i = 0; i < n; i++) {
    if (expected[i] !== 0) result.solid++;
    if (gpu[i] !== expected[i]) {
      result.mismatches++;
      if (!result.firstMismatch) {
        const [x, y, z] = [positions[i * 4]!, positions[i * 4 + 1]!, positions[i * 4 + 2]!];
        result.firstMismatch = { x, y, z, cpu: expected[i]!, gpu: gpu[i]! };
      }
    }
  }
  result.ms = performance.now() - start;
  return result;
}
