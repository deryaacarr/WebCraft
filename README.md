# WebCraft

WebCraft, tarayıcıda çalışan, tek oyunculu ve fotogerçekçi görünümlü bir voxel sandbox oyunudur. Vite + TypeScript ile yazılır ve harici bir render motoru kullanmadan doğrudan WebGPU + WGSL üzerine kurulur; birincil ışınlar, gölgeler, global aydınlatma ve yansımalar compute shader'larda voxel ray tracing (DDA traversal) ile hesaplanır. Hedef, sisli dağ ormanları, gerçekçi yağmur ve ses atmosferiyle RTX 3060 / M1 seviyesindeki donanımda 1080p'de 60 FPS sunan bir deneyimdir. Tüm texture ve sesler CC0 kaynaklardan gelir.

## Geliştirme

```bash
npm install
npm run textures   # ambientCG'den CC0 dokuları indirir ve paketler (bir kez; ~120 MB indirme)
npm run dev
```

Dokular paketlenmemişse oyun düz renklerle çalışır. Kaynaklar: [assets/CREDITS.md](assets/CREDITS.md).
