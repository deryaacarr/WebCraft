import { describe, expect, it } from 'vitest';
import {
  CHUNK_SIZE,
  CHUNK_VOLUME,
  chunkKey,
  chunkOrigin,
  localIndex,
  parseChunkKey,
  toChunkCoord,
  toLocalCoord,
} from './coords';

const S = CHUNK_SIZE;

describe('chunk coordinates', () => {
  it('maps positive coordinates', () => {
    expect(toChunkCoord(0)).toBe(0);
    expect(toChunkCoord(S - 1)).toBe(0);
    expect(toChunkCoord(S)).toBe(1);
    expect(toLocalCoord(S + 5)).toBe(5);
  });

  it('floors negative coordinates instead of truncating toward zero', () => {
    expect(toChunkCoord(-1)).toBe(-1);
    expect(toLocalCoord(-1)).toBe(S - 1);
    expect(toChunkCoord(-S)).toBe(-1);
    expect(toLocalCoord(-S)).toBe(0);
    expect(toChunkCoord(-S - 1)).toBe(-2);
    expect(toLocalCoord(-S - 1)).toBe(S - 1);
  });

  it('floors fractional positions', () => {
    expect(toChunkCoord(-0.5)).toBe(-1);
    expect(toLocalCoord(-0.5)).toBe(S - 1);
    expect(toChunkCoord(S - 0.001)).toBe(0);
  });

  it('round-trips world → (chunk, local) → world', () => {
    for (let v = -3 * S - 7; v <= 3 * S + 7; v++) {
      const c = toChunkCoord(v);
      const l = toLocalCoord(v);
      expect(l).toBeGreaterThanOrEqual(0);
      expect(l).toBeLessThan(S);
      expect(chunkOrigin(c) + l).toBe(v);
    }
  });

  it('produces unique, in-range local indices', () => {
    const seen = new Set<number>();
    for (let y = 0; y < S; y++)
      for (let z = 0; z < S; z++)
        for (let x = 0; x < S; x++) seen.add(localIndex(x, y, z));
    expect(seen.size).toBe(CHUNK_VOLUME);
    expect(Math.min(...seen)).toBe(0);
    expect(Math.max(...seen)).toBe(CHUNK_VOLUME - 1);
  });

  it('builds and parses chunk keys', () => {
    expect(chunkKey(-1, 2, -3)).toBe('-1,2,-3');
    expect(parseChunkKey('-1,2,-3')).toEqual([-1, 2, -3]);
    expect(() => parseChunkKey('1,2')).toThrow();
  });
});
