import { Input } from './core/input';
import { GameLoop } from './core/loop';
import { initGpu, WebGPUUnsupportedError } from './gpu/device';
import { Renderer } from './gpu/renderer';
import { DebugOverlay } from './ui/debug-overlay';
import { showErrorScreen } from './ui/error-screen';

async function main(): Promise<void> {
  const canvas = document.querySelector<HTMLCanvasElement>('#canvas');
  if (!canvas) throw new Error('#canvas bulunamadı');

  const gpu = await initGpu(canvas);

  const renderer = new Renderer(gpu, canvas);
  await renderer.init();

  const input = new Input(canvas);
  const debug = new DebugOverlay(renderer.profiler, {
    onResolutionChange: () => renderer.invalidateResolution(),
  });

  const loop = new GameLoop({
    update: () => {
      // Simulation systems (player, world, weather) will tick here.
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
