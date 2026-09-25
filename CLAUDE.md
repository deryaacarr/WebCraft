# WebCraft

Tarayıcıda çalışan, fotogerçekçi görüntüye sahip, tek oyunculu bir voxel sandbox oyunu. Hedef görünüm: path tracing shader'lı, sisli dağ ormanları, gerçekçi yağmur ve ses.

## Teknik kurallar
- Stack: Vite + TypeScript (strict), saf WebGPU + WGSL. Three.js veya başka bir render motoru KULLANMA; render pipeline'ı biz yazıyoruz.
- Render yaklaşımı: voxel ray tracing. Birincil ışınlar, gölgeler, GI ve yansımalar compute shader'da DDA (Amanatides & Woo) voxel traversal ile hesaplanır.
- Ağır işler (terrain üretimi, chunk paketleme) Web Worker'larda yapılır. Ana thread asla bloklanmaz.
- Klasör yapısı: src/core (loop, input, time), src/world (chunk, gen, blocks), src/gpu (device, buffers, passes), src/shaders (*.wgsl), src/audio, src/player, src/ui, src/weather.
- WGSL dosyaları ayrı .wgsl dosyalarında tutulur ve ?raw ile import edilir. Ortak fonksiyonlar basit bir #include preprocessor'ı ile birleştirilir.
- Her render pass ayrı bir sınıftır (init, resize, execute).
- Sihirli sayı yok: tüm parametreler tek bir config nesnesinde toplanır ve lil-gui debug panelinden canlı değiştirilebilir.
- Performans hedefi: RTX 3060 / M1 seviyesinde 1080p'de 60 FPS (dahili çözünürlük düşük olup upscale edilebilir).
- Minecraft'a ait hiçbir asset, isim veya marka kullanılmaz. Texture ve sesler CC0 kaynaklardan gelir.

## Çalışma şekli
- Her aşamadan sonra `npm run build` ve `npm test` çalıştır, hata varsa düzelt.
- WGSL hatalarını görmek için device.pushErrorScope ve getCompilationInfo kullan, sonuçları konsola yaz.
- Bir aşama bitince yaptıklarını ve nasıl test edileceğini kısaca özetle, sonra dur. Bir sonraki aşamaya kendiliğinden geçme.
- Mimari kararlar değişirse bu dosyanın "Mimari" bölümünü güncelle.

## Mimari
- **Config**: `src/config.ts` tek `config` nesnesi; lil-gui paneli (`src/ui/debug-overlay.ts`) bu nesneyi doğrudan düzenler. Render hedefi boyutunu etkileyen ayarlar `onResolutionChange` hook'u ile `Renderer.invalidateResolution()`'ı tetikler.
- **Döngü**: `FixedStepClock` (saf, test edilir) + `GameLoop` (rAF). `update(dt)` sabit adımda, `render(alpha, time)` her karede; `time = simTime + alpha * step`. Input kenarları (pressed/released, mouse delta) `Input.endTick()` ile her simülasyon adımında sıfırlanır; tuşlar `KeyboardEvent.code` ile tutulur.
- **GPU başlatma**: `initGpu()` 'timestamp-query'yi varsa açar. Desteklenmeyen durumlar `WebGPUUnsupportedError` fırlatır → `showErrorScreen`.
- **Canvas boyutu**: `CanvasSize` ResizeObserver ile ölçer, backing store'u yalnızca kare başında `apply()` ile değiştirir. DPR `config.render.maxPixelRatio` ile sınırlanır.
- **Render pass'leri**: `RenderPass` arayüzü (`init`/`resize`/`execute`), `FrameContext` ile encoder + profiler + zaman alır. Zincir: compute pass'ler dahili çözünürlükte (`canvas × renderScale`) `rgba16float` storage texture'a yazar → `BlitPass` tam ekran üçgenle canvas'a upscale eder. Şu an tek compute pass test `GradientPass`.
- **Shader'lar**: `src/shaders/*.wgsl`, `import.meta.glob(..., { query: '?raw' })` ile yüklenir; `#include "x.wgsl"` satır başında olmalı, her dosya shader başına bir kez eklenir (`preprocess.ts`). `createShaderModule()` getCompilationInfo + error scope sonuçlarını konsola yazar. Workgroup boyutu WGSL `override` sabiti olarak pipeline'a verilir.
- **Profiler**: `GpuProfiler` pass başına `timestampWrites(name)` verir, sonuçlar birkaç kare gecikmeli ve üstel ortalamalı gelir. stats-gl'in kendi GPU takibi kapalı (tek pass ölçüyor); toplam GPU süresi özel bir "GPU" paneline elle yazılır. Not: stats-gl 4.2'nin public `updatePanel()`'ı argümanları yanlış sırayla geçiriyor, kullanma.
