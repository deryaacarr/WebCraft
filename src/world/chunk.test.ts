import { describe, expect, it } from 'vitest';
import { BlockId } from './blocks';
import { Chunk } from './chunk';
import { CHUNK_SIZE, CHUNK_VOLUME } from './coords';

const last = CHUNK_SIZE - 1;

describe('Chunk', () => {
  it('starts as uniform air without allocating', () => {
    const c = new Chunk(0, 0, 0);
    expect(c.isEmpty).toBe(true);
    expect(c.uniformBlock).toBe(BlockId.air);
    expect(c.get(3, 4, 5)).toBe(BlockId.air);
  });

  it('sets and gets blocks, including chunk corners', () => {
    const c = new Chunk(0, 0, 0);
    expect(c.set(0, 0, 0, BlockId.stone)).toBe(true);
    expect(c.set(last, last, last, BlockId.glass)).toBe(true);
    expect(c.get(0, 0, 0)).toBe(BlockId.stone);
    expect(c.get(last, last, last)).toBe(BlockId.glass);
    expect(c.get(1, 0, 0)).toBe(BlockId.air);
    expect(c.nonAirCount).toBe(2);
    expect(c.uniformBlock).toBeNull();
  });

  it('reports unchanged writes', () => {
    const c = new Chunk(0, 0, 0);
    expect(c.set(1, 1, 1, BlockId.air)).toBe(false);
    c.set(1, 1, 1, BlockId.dirt);
    expect(c.set(1, 1, 1, BlockId.dirt)).toBe(false);
  });

  it('drops its array when it becomes all air again', () => {
    const c = new Chunk(0, 0, 0);
    c.set(2, 2, 2, BlockId.sand);
    c.set(2, 2, 2, BlockId.air);
    expect(c.isEmpty).toBe(true);
    expect(c.uniformBlock).toBe(BlockId.air);
  });

  it('materializes a uniform solid chunk on first differing write', () => {
    const c = new Chunk(0, -5, 0, BlockId.stone);
    expect(c.nonAirCount).toBe(CHUNK_VOLUME);
    c.set(4, 4, 4, BlockId.air);
    expect(c.get(4, 4, 4)).toBe(BlockId.air);
    expect(c.get(4, 4, 5)).toBe(BlockId.stone);
    expect(c.nonAirCount).toBe(CHUNK_VOLUME - 1);
  });

  it('keeps uniform chunks array-free, whatever the block', () => {
    for (const id of [BlockId.air, BlockId.stone, BlockId.water]) {
      const c = new Chunk(0, 0, 0, id);
      expect(c.uniformBlock).toBe(id);
      expect(c.byteLength).toBe(0);
      expect(c.set(1, 2, 3, id)).toBe(false);
      expect(c.byteLength).toBe(0);
    }
  });

  it('stores dense data in one byte per block while ids fit, widening on demand', () => {
    const c = new Chunk(0, 0, 0, BlockId.stone);
    c.set(0, 0, 0, BlockId.air);
    expect(c.byteLength).toBe(CHUNK_VOLUME);
    c.set(1, 0, 0, 300);
    expect(c.byteLength).toBe(CHUNK_VOLUME * 2);
    expect(c.get(0, 0, 0)).toBe(BlockId.air);
    expect(c.get(1, 0, 0)).toBe(300);
    expect(c.get(2, 0, 0)).toBe(BlockId.stone);
  });

  it('starts wide when the first write needs it', () => {
    const c = new Chunk(0, 0, 0);
    c.set(0, 0, 0, 1000);
    expect(c.byteLength).toBe(CHUNK_VOLUME * 2);
    expect(c.get(0, 0, 0)).toBe(1000);
  });

  it('narrows wide input with small ids in fromArray', () => {
    const data = new Uint16Array(CHUNK_VOLUME);
    data[5] = BlockId.glass;
    expect(Chunk.fromArray(0, 0, 0, data).byteLength).toBe(CHUNK_VOLUME);
    data[6] = 400;
    expect(Chunk.fromArray(0, 0, 0, data).byteLength).toBe(CHUNK_VOLUME * 2);
  });

  it('round-trips through dense arrays and stays compact when uniform', () => {
    const c = new Chunk(1, 2, 3);
    c.set(5, 6, 7, BlockId.torch);
    const copy = Chunk.fromArray(1, 2, 3, c.toArray());
    expect(copy.get(5, 6, 7)).toBe(BlockId.torch);
    expect(copy.nonAirCount).toBe(1);
    const uniform = Chunk.fromArray(0, 0, 0, new Uint16Array(CHUNK_VOLUME).fill(BlockId.water));
    expect(uniform.uniformBlock).toBe(BlockId.water);
    expect(() => Chunk.fromArray(0, 0, 0, new Uint16Array(3))).toThrow();
  });
});
