import { config } from './config';
import { Input } from './core/input';
import type { Vec3 } from './core/math';
import { parseUrlOverrides } from './core/url-params';
import { GameLoop } from './core/loop';
import { initGpu, WebGPUUnsupportedError } from './gpu/device';
import { GpuBrickmap } from './gpu/brickmap';
import { verifyBrickmap } from './gpu/brickmap-verify';
import { verifyTrace } from './gpu/trace-verify';
import { MaterialSystem } from './gpu/materials';
import { Renderer } from './gpu/renderer';
import { FlyCamera } from './player/camera';
import { DebugOverlay } from './ui/debug-overlay';
import { showErrorScreen } from './ui/error-screen';
import { TerrainGenerator } from './world/gen/terrain';
import { defaultWorkerCount, TerrainWorkerPool } from './world/gen/worker-pool';
import { ChunkStreamer } from './world/streamer';
import { World } from './world/world';

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#canvas');
  if (!canvas) throw new Error('#canvas bulunamadı');

  const overrides = parseUrlOverrides(location.search);
  if (overrides.view) config.debug.view = overrides.view;
  if (overrides.renderScale) config.render.renderScale = overrides.renderScale;
  if (overrides.workgroup) config.trace.workgroup = overrides.workgroup;
  if (overrides.prepassTile) config.trace.prepassTile = overrides.prepassTile;
  if (overrides.distanceMax) config.trace.distanceMax = overrides.distanceMax;

  const gpu = await initGpu(canvas);

  const brickmap = new GpuBrickmap(gpu.device);
  await brickmap.init();
  const materials = new MaterialSystem(gpu.device);
  await materials.init();
  const renderer = new Renderer(gpu, canvas, brickmap, materials);
  await renderer.init();

  const world = new World();
  const pool = new TerrainWorkerPool(
    config.terrain,
    config.world,
    config.generation.heightCacheColumns,
    defaultWorkerCount(config.generation.workers),
  );
  const streamer = new ChunkStreamer(world, pool);

  // Debug fly camera, spawned above the ground (or water) at the origin.
  const spawn = (): Vec3 => [
    0.5,
    Math.max(new TerrainGenerator(config.terrain, config.world).surfaceY(0, 0), config.terrain.seaLevel) +
      config.camera.spawnHeight,
    0.5,
  ];
  const pose = overrides.camera;
  let camera = pose ? new FlyCamera(pose.position, pose.pitchDeg, pose.yawDeg) : new FlyCamera(spawn());

  const regenerate = () => {
    streamer.reset();
    world.clear();
    pool.reset(config.terrain, config.generation.heightCacheColumns);
    camera = new FlyCamera(spawn());
  };

  const input = new Input(canvas);
  const debug = new DebugOverlay(renderer.profiler, {
    onResolutionChange: () => renderer.invalidateResolution(),
    onRegenerate: regenerate,
    brickmapStats: () => ({ ...brickmap.memory, ...brickmap.store.stats }),
    onVerifyBrickmap: () => verifyBrickmap(gpu.device, brickmap, world),
    onVerifyTrace: () => verifyTrace(gpu.device, brickmap, world, camera.position),
    cameraInfo: () => ({ position: camera.position, internal: renderer.internalSize }),
    onBenchmarkPrimary: () => renderer.benchmarkPrimary(config.debug.benchmarkIterations),
    onVerifyPrepass: () => renderer.verifyPrepass(),
    materialStats: () => materials.stats,
    onTextureResolution: (res) => materials.load(res),
    onSamplerChange: () => materials.createSampler(),
    worldStats: () => {
      let dense = 0;
      let chunkBytes = 0;
      for (const chunk of world.chunkValues()) {
        if (chunk.uniformBlock === null) dense++;
        chunkBytes += chunk.byteLength;
      }
      const { workers, queued, avgChunkMs } = pool.stats;
      const uniform = world.chunkCount - dense;
      return { ...streamer.stats, dense, uniform, chunkBytes, workers, queued, avgChunkMs };
    },
  });

  const loop = new GameLoop({
    update: (dt) => {
      camera.update(dt, input);
      const [x, y, z] = camera.position;
      streamer.update(x, y, z);
      input.endTick();
    },
    render: (alpha, time) => {
      debug.beginFrame();
      brickmap.sync(world, camera.position[0], camera.position[2]);
      renderer.render(time, camera, alpha);
      debug.endFrame();
    },
  });

  gpu.device.lost.then((info) => {
    loop.stop();
    if (info.reason === 'destroyed') return;
    showErrorScreen('GPU bağlantısı koptu', `${info.message || 'Bilinmeyen neden.'}\nSayfayı yenileyin.`);
  });
  loop.start();
}

main().catch((err: unknown) => {
  console.error(err);
  if (err instanceof WebGPUUnsupportedError) {
    showErrorScreen(
      'WebGPU desteklenmiyor',
      `${err.message}\n\nWebCraft için WebGPU destekli güncel bir tarayıcı gerekir ` +
        '(Chrome/Edge 113+, Safari 26+ veya Firefox 141+). Donanım hızlandırmanın açık olduğundan emin olun.',
    );
  } else {
    showErrorScreen('Başlatma hatası', err instanceof Error ? err.message : String(err));
  }
});
