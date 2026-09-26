import { describe, expect, it } from 'vitest';
import { parseUrlOverrides } from './url-params';

describe('parseUrlOverrides', () => {
  it('parses camera pose, view and scale', () => {
    expect(parseUrlOverrides('?cam=1,2.5,-3,90,-20&view=steps&scale=0.75')).toEqual({
      camera: { position: [1, 2.5, -3], yawDeg: 90, pitchDeg: -20 },
      view: 'steps',
      renderScale: 0.75,
    });
  });

  it('parses and wraps the time of day', () => {
    expect(parseUrlOverrides('?time=18.5').timeOfDay).toBe(18.5);
    expect(parseUrlOverrides('?time=-1').timeOfDay).toBe(23);
  });

  it('parses test torches and the GI switch', () => {
    expect(parseUrlOverrides('?torches=8&gi=0')).toEqual({ torches: 8, gi: false });
    expect(parseUrlOverrides('?gi=1').gi).toBe(true);
    expect(parseUrlOverrides('?torches=-3').torches).toBeUndefined();
  });

  it('ignores malformed values', () => {
    expect(parseUrlOverrides('?cam=1,2,x,4,5&view=nope&scale=-1')).toEqual({});
  });
});
