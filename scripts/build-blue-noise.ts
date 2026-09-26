/**
 * Generates public/blue-noise.bin: spatiotemporal blue noise (see lib/blue-noise.ts),
 * 64 × 64 pixels × 16 slices, four independent scalar channels (RGBA8), laid out as
 * [slice][y][x][channel]. Our own generator, so the asset is ours (no licence question).
 *
 *   npm run bluenoise
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spatiotemporalBlueNoise } from './lib/blue-noise.ts';

const SIZE = 64;
const DEPTH = 16;
const CHANNELS = 4;
const OUT = join(new URL('..', import.meta.url).pathname, 'public/blue-noise.bin');

const t0 = performance.now();
const count = SIZE * SIZE * DEPTH;
const out = new Uint8Array(count * CHANNELS);
for (let c = 0; c < CHANNELS; c++) {
  const rank = spatiotemporalBlueNoise(SIZE, DEPTH, 0x5eed + c * 7919);
  for (let i = 0; i < count; i++) out[i * CHANNELS + c] = Math.floor((rank[i]! * 256) / count);
  console.log(`  channel ${c} done (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
}
writeFileSync(OUT, out);
console.log(`${SIZE}×${SIZE}×${DEPTH} × ${CHANNELS} channels → ${OUT}`);
