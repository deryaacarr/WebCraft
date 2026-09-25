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
- **Bloklar**: `src/world/blocks.ts`. `BlockId` sırası chunk verisinin ve ileride GPU/kayıt formatının parçası: sona ekle, asla yeniden sıralama. Yüz materyalleri `MATERIAL_NAMES` indeksleri (gelecekteki texture array katmanları). Sıcak döngüler için `isSolid`/`isTransparent` typed-array tablolarını kullan.
- **Chunk**: 32³ (`config.world.chunkBits`), düzen `y<<10 | z<<5 | x`. Bellek: tek tip chunk'lar (hava, derin taş) dizi tutmaz, yalnızca blok id'si; dizi ilk farklı yazımda oluşur, chunk tamamen havaya dönünce bırakılır. Yoğun dizi id'ler < 256 oldukça `Uint8Array` (32 KB), daha büyük id yazılınca `Uint16Array`'e genişler; `toArray()` her zaman `Uint16Array` döner. Dünya koordinatları int32, bit işlemleriyle floor'lanır (negatifler doğru).
- **World**: `Map<"x,y,z", Chunk>`, dikey sınır `config.world.minY` (dahil) – `maxY` (hariç), varsayılan -64..320. Sınır dışı ve yüklü olmayan chunk'ta `getBlock` hava döner, `setBlock` false döner. Dirty takibi: değişen chunk + sınır bloklarında komşu; chunk ekle/çıkar komşuları da kirletir. Tüketici `takeDirty()` ile alır ve temizler.
- **Streaming**: `ChunkStreamer` XZ'de dairesel `horizontalRadius` içindeki sütunları, dikeyde oyuncudan bağımsız olarak dünya sınırları arasındaki tüm katmanlarla yükler (sınır dışı chunk asla istenmez). İstekler oyuncuya en yakından uzağa asenkron `ChunkProvider.request()` ile gider (`maxInFlight` sınırlı); `radius + unloadMargin` dışı ve sınır dışı chunk'lar boşaltılır, bekleyen istekleri iptal edilir. Offset listesi yalnızca yarıçap/sınır veya oyuncunun dikey katmanı değişince yeniden kurulur.
- **Arazi üretimi** (`src/world/gen/`): `TerrainGenerator` (params, chunk koordinatı) için saf ve deterministik bir fonksiyon. Yükseklik = `continentalnessSpline(C) + erosionSpline(E) · peaksValleysSpline(PV)` + küçük detay; PV ridged fraktal, tüm 2D noise domain warp'lı. Spline'lar monotone cubic; girişler noise'un pratik aralığına (~±0.7) göre ayarlı. Üretici `WorldBounds` alır: sınır dışı her şey hava, en alt katman oyulmaz (mağara boşluğa açılmaz); yoğun çıktı 8-bit, worker `bytesPerBlock` bildirir. Chunk sütunu başına yükseklik ızgarası (+ ağaç erişimi kadar dolgu halkası) worker'da LRU'da tutulur ve dikey yığınla paylaşılır. Yüzey kuralları eğime ve deniz seviyesine göre (`config.terrain.surface`). Mağaralar: cheese (eşik) + spaghetti (iki ince kabuğun kesişimi), 3D noise `latticeStep` aralıklı kafeste örneklenip trilineer interpole edilir; su altında/yanında çatı korunur.
- **Ağaçlar**: aday konum `hash(seed, hücre)`, kabul `densitySpline(orman noise) × ağaç sınırı` — hepsi dünya koordinatından türetilir, komşu chunk'a ihtiyaç yok. Her chunk, gövdesi komşuda olsa bile kendine taşan ağaçları çizer; önce yapraklar (yalnızca zeminin üstündeki havaya), sonra gövdeler → sonuç sıradan bağımsız.
- **Worker havuzu**: `TerrainWorkerPool` (`ChunkProvider`), `hardwareConcurrency − 1` worker, worker başına tek iş, FIFO. Yoğun chunk verisi `postMessage(msg, [buffer])` ile taşınır (kopya yok); tek tip chunk'lar buffer'sız gelir. İş/kurulum mesajları `generation` taşır; "Regenerate" nesli artırır, eski sonuçlar yok sayılır. `WorkerHandler` worker global'lerinden bağımsız, testler sahte worker ile çalışır.
