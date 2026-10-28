import { config } from '../config';
import { shapedIdMask } from '../world/shapes';

/**
 * Two-pipeline scheme for sub-voxel shapes (trace.wgsl MICRO_WALK): the main pipeline of a
 * tracing pass treats shaped blocks as cubes and flags the pixels whose rays stopped at
 * one in an r32uint texture; the fix-up pipeline (same shader, MICRO_WALK on) redoes only
 * the flagged pixels on the micro grid. The shaders bind the flag twice — as storage
 * (main writes) and as texture (fix-up reads) — so each pipeline gets a 1×1 dummy for the
 * binding it does not use.
 */
export interface MicroPipelines {
  main: GPUComputePipeline;
  /** null when there is nothing to fix up (no shaped blocks, or &micro=0 at startup). */
  fixup: GPUComputePipeline | null;
}

/** Whether fix-up pipelines are built (the shape mask they need is non-zero). */
export function microFixupEnabled(): boolean {
  return config.detail.microVoxels && shapedIdMask() !== 0;
}

export async function createMicroPipelines(
  device: GPUDevice,
  label: string,
  module: GPUShaderModule,
  constants: Record<string, number>,
): Promise<MicroPipelines> {
  const make = (micro: boolean) =>
    device.createComputePipelineAsync({
      label: micro ? `${label}-micro` : label,
      layout: 'auto',
      compute: { module, entryPoint: 'main', constants: { ...constants, MICRO_WALK: micro ? 1 : 0 } },
    });
  const [main, fixup] = await Promise.all([make(false), microFixupEnabled() ? make(true) : null]);
  return { main, fixup };
}

/** Flag texture of a tracing pass (see MicroPipelines). */
export function createMicroFlags(device: GPUDevice, label: string, width: number, height: number): GPUTexture {
  return device.createTexture({
    label: `${label}-micro-flags`,
    size: { width: Math.max(width, 1), height: Math.max(height, 1) },
    format: 'r32uint',
    // RENDER_ATTACHMENT: cleared with a render pass where only flagged pixels are written.
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

/** Zeroes a flag texture (a render pass clear: cheap, no shader). */
export function clearMicroFlags(encoder: GPUCommandEncoder, flags: GPUTexture): void {
  encoder
    .beginRenderPass({ label: `${flags.label}-clear`, colorAttachments: [{ view: flags.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] })
    .end();
}

const dummies = new WeakMap<GPUDevice, GPUTexture>();

/** 1×1 flag texture for the flag binding a pipeline does not use. */
export function microFlagDummy(device: GPUDevice): GPUTexture {
  let t = dummies.get(device);
  if (!t) {
    t = createMicroFlags(device, 'dummy', 1, 1);
    dummies.set(device, t);
  }
  return t;
}
