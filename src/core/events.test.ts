import { describe, expect, it } from 'vitest';
import { emit, on } from './events';

describe('game events', () => {
  it('delivers the detail to listeners until they unsubscribe', () => {
    const seen: boolean[] = [];
    const off = on('underwater', (d) => seen.push(d.underwater));
    emit('underwater', { underwater: true });
    off();
    emit('underwater', { underwater: false });
    expect(seen).toEqual([true]);
  });
});
