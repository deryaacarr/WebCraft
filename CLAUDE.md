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
(Aşamalar ilerledikçe doldurulacak.)
