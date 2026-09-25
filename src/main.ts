import { config } from './config';
import { Input } from './core/input';
import { GameLoop } from './core/loop';
import { initGpu, WebGPUUnsupportedError } from './gpu/device';
import { Renderer } from './gpu/renderer';
import { DebugOverlay } from './ui/debug-overlay';
import { showErrorScreen } from './ui/error-screen';
import { TerrainGenerator } from './world/gen/terrain';
import { defaultWorkerCount, TerrainWorkerPool } from './world/gen/worker-pool';
import { ChunkStreamer } from './world/streamer';
import { World } from './world/world';

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#canvas');
  if (!canvas) throw new Error('#canvas bulunamadı');

  const gpu = await initGpu(canvas);

  const renderer = new Renderer(gpu, canvas);
  await renderer.init();

  const world = new World();
  const pool = new TerrainWorkerPool(
    config.terrain,
    config.world,
    config.generation.heightCacheColumns,
    defaultWorkerCount(config.generation.workers),
  );
  const streamer = new ChunkStreamer(world, pool);

  // Placeholder until the player controller exists: stand above the ground (or water) at the origin.
  const spawnY = () =>
    Math.max(new TerrainGenerator(config.terrain, config.world).surfaceY(0, 0), config.terrain.seaLevel) + 2;
  const playerPosition = { x: 0, y: spawnY(), z: 0 };

  const regenerate = () => {
    streamer.reset();
    world.clear();
    pool.reset(config.terrain, config.generation.heightCacheColumns);
    playerPosition.y = spawnY();
  };

  const input = new Input(canvas);
  const debug = new DebugOverlay(renderer.profiler, {
    onResolutionChange: () => renderer.invalidateResolution(),
    onRegenerate: regenerate,
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
    update: () => {
      // Simulation systems (player, weather) will tick here.
      streamer.update(playerPosition.x, playerPosition.y, playerPosition.z);
      input.endTick();
    },
    render: (_alpha, time) => {
      debug.beginFrame();
      renderer.render(time);
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
