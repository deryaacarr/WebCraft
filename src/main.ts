import { config } from './config';
import { Input } from './core/input';
import type { Vec3 } from './core/math';
import { TimeOfDay } from './core/time-of-day';
import { parseUrlOverrides } from './core/url-params';
import { GameLoop } from './core/loop';
import { initGpu, WebGPUUnsupportedError } from './gpu/device';
import { GpuBrickmap } from './gpu/brickmap';
import { verifyBrickmap } from './gpu/brickmap-verify';
import { verifyTrace } from './gpu/trace-verify';
import { MaterialSystem } from './gpu/materials';
import { SkySystem } from './gpu/sky';
import { Renderer } from './gpu/renderer';
import { FlyCamera } from './player/camera';
import { DebugOverlay } from './ui/debug-overlay';
import { showErrorScreen } from './ui/error-screen';
import { TerrainGenerator } from './world/gen/terrain';
import { defaultWorkerCount, TerrainWorkerPool } from './world/gen/worker-pool';
import { ChunkStreamer } from './world/streamer';
import { World } from './world/world';
import { BlockId, isSolid } from './world/blocks';
import { EmitterRegistry } from './world/emitters';

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#canvas');
  if (!canvas) throw new Error('#canvas bulunamadı');

  const overrides = parseUrlOverrides(location.search);
  if (overrides.view) config.debug.view = overrides.view;
  if (overrides.renderScale) config.render.renderScale = overrides.renderScale;
  if (overrides.cameraSpin !== undefined) config.debug.cameraSpin = overrides.cameraSpin;
  if (overrides.cameraFly !== undefined) config.debug.cameraFly = overrides.cameraFly;
  if (overrides.timeOfDay !== undefined) {
    config.sky.timeOfDay = overrides.timeOfDay;
    config.sky.paused = true;
  }
  if (overrides.workgroup) config.trace.workgroup = overrides.workgroup;
  if (overrides.gi !== undefined) config.gi.enabled = overrides.gi;
  if (overrides.prepassTile) config.trace.prepassTile = overrides.prepassTile;
  if (overrides.distanceMax) config.trace.distanceMax = overrides.distanceMax;

  const gpu = await initGpu(canvas);

  const brickmap = new GpuBrickmap(gpu.device);
  await brickmap.init();
  const materials = new MaterialSystem(gpu.device);
  await materials.init();
  const sky = new SkySystem(gpu.device);
  await sky.init();
  const clock = new TimeOfDay(config.sky.timeOfDay, config.sky.startDay);
  const emitters = new EmitterRegistry();
  const renderer = new Renderer(gpu, canvas, brickmap, materials, sky, emitters);
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
  /**
   * Debug: puts torches on the ground around the camera (lighting tests; there is no
   * block placing yet). Scans each random column down from just above the camera for the
   * first air cell resting on a solid block (works in caves too). Returns how many.
   */
  const placeTorches = (count: number, radius: number): number => {
    const [cx, cy, cz] = camera.position;
    let placed = 0;
    for (let attempt = 0; attempt < count * 20 && placed < count; attempt++) {
      const a = Math.random() * 2 * Math.PI;
      const r = radius * Math.sqrt(Math.random());
      const x = Math.floor(cx + Math.cos(a) * r);
      const z = Math.floor(cz + Math.sin(a) * r);
      for (let y = Math.floor(cy) + 4; y > cy - radius * 2; y--) {
        if (world.getBlock(x, y, z) !== BlockId.air || !isSolid(world.getBlock(x, y - 1, z))) continue;
        if (world.setBlock(x, y, z, BlockId.torch)) placed++;
        break;
      }
    }
    return placed;
  };
  let pendingTorches = overrides.torches ?? 0;

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
    onBenchmarkLighting: () => renderer.benchmarkLighting(config.debug.benchmarkIterations),
    onVerifyPrepass: () => renderer.verifyPrepass(),
    onBenchmarkFrame: () => renderer.benchmarkFrame(config.debug.benchmarkIterations),
    onPlaceTorches: () => placeTorches(config.debug.testTorches, config.debug.testTorchRadius),
    emitterCount: () => emitters.total,
    clock,
    skyState: () => sky.state,
    lightingStats: () => sky.readStats(),
    probeCenter: () => renderer.probeCenter(),
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

  // Dev builds: expose internals for automated tests and benchmarks (console / CDP).
  if (import.meta.env.DEV) Object.assign(window, {
      webcraft: {
        config, renderer, world, emitters, streamer, placeTorches,
        setCamera: (x: number, y: number, z: number, yawDeg: number, pitchDeg: number) => {
          camera = new FlyCamera([x, y, z], pitchDeg, yawDeg);
        },
      },
    });

  const loop = new GameLoop({
    update: (dt) => {
      const minutes = config.sky.dayLengthMinutes;
      if (!config.sky.paused && minutes > 0) clock.advance((dt * 86400) / (minutes * 60));
      camera.update(dt, input);
      const [x, y, z] = camera.position;
      streamer.update(x, y, z);
      input.endTick();
    },
    render: (alpha, time) => {
      debug.beginFrame();
      // &torches=N: once the chunks around the camera are loaded.
      if (pendingTorches > 0 && world.chunkCount > 0 && streamer.stats.missing === 0) {
        pendingTorches -= placeTorches(pendingTorches, config.debug.testTorchRadius);
      }
      const changes = world.takeChanges();
      emitters.apply(changes);
      brickmap.sync(world, camera.position[0], camera.position[2], changes);
      sky.update(clock.state(config.sky), camera.position[1]);
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
