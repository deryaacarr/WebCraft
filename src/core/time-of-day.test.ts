import { describe, expect, it } from 'vitest';
import { bodyDirection, LUNAR_CYCLE_DAYS, TimeOfDay } from './time-of-day';

const DEG = Math.PI / 180;
const orbit = { latitude: 45, sunDeclination: 0, moonInclination: 5 };
const elevation = (d: number[]) => Math.asin(d[1]!) / DEG;

describe('bodyDirection', () => {
  it('culminates due south at 90° − latitude + declination', () => {
    const d = bodyDirection(45 * DEG, 10 * DEG, 0);
    expect(elevation(d)).toBeCloseTo(55);
    expect(d[0]).toBeCloseTo(0); // no east/west component
    expect(d[2]).toBeGreaterThan(0); // south = +Z
  });

  it('is a unit vector', () => {
    for (const h of [0, 1, 2.5, -3]) expect(Math.hypot(...bodyDirection(30 * DEG, -20 * DEG, h))).toBeCloseTo(1);
  });
});

describe('TimeOfDay', () => {
  it('puts the equinox sun east at 6h, overhead-ish at noon, west at 18h, below at midnight', () => {
    const at = (h: number) => new TimeOfDay(h).state(orbit).sunDir;
    expect(elevation(at(6))).toBeCloseTo(0, 5);
    expect(at(6)[0]).toBeCloseTo(1); // east = +X
    expect(elevation(at(12))).toBeCloseTo(45);
    expect(at(18)[0]).toBeCloseTo(-1);
    expect(elevation(at(0))).toBeCloseTo(-45);
  });

  it('wraps hours into days', () => {
    const t = new TimeOfDay(23);
    t.advance(2 * 3600);
    expect(t.hours).toBeCloseTo(1);
    expect(t.day).toBe(1);
  });

  it('runs the moon through its phases: new with the sun, full opposite it', () => {
    const newMoon = new TimeOfDay(12, 0).state(orbit);
    expect(newMoon.moonPhase).toBeCloseTo(0);
    const full = new TimeOfDay(12, 0);
    full.advance((LUNAR_CYCLE_DAYS / 2) * 24 * 3600);
    const s = full.state(orbit);
    expect(s.moonPhase).toBeCloseTo(1, 3);
    // Full moon is roughly opposite the sun (inclination only tilts it slightly).
    const dot = s.moonDir[0] * s.sunDir[0] + s.moonDir[1] * s.sunDir[1] + s.moonDir[2] * s.sunDir[2];
    expect(dot).toBeLessThan(-0.95);
  });
});
