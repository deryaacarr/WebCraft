import type { Vec3 } from './math';

const DEG = Math.PI / 180;
/** Synodic month: new moon to new moon. */
export const LUNAR_CYCLE_DAYS = 29.53;

export interface CelestialState {
  /** Unit vectors towards the sun / moon in world space (+X east, +Y up, −Z north). */
  sunDir: Vec3;
  moonDir: Vec3;
  /** Lit fraction of the moon's disk (0 = new, 1 = full). */
  moonPhase: number;
  /** Rotation of the star sphere (hour angle, radians). */
  starRotation: number;
}

export interface OrbitParams {
  /** Observer latitude in degrees (+ north). */
  latitude: number;
  /** Sun declination in degrees (season: +23.4 summer solstice, 0 equinox, −23.4 winter). */
  sunDeclination: number;
  /** Moon orbit inclination against the sun's path, degrees. */
  moonInclination: number;
}

/**
 * Direction of a body at declination `decl` and hour angle `hour` (0 = due south at
 * its highest, radians) for an observer at `latitude`: standard equatorial → horizontal
 * conversion, returned as east/up/north mapped to world +X / +Y / −Z.
 */
export function bodyDirection(latitude: number, decl: number, hour: number): Vec3 {
  const sinLat = Math.sin(latitude);
  const cosLat = Math.cos(latitude);
  const east = -Math.cos(decl) * Math.sin(hour);
  const north = cosLat * Math.sin(decl) - sinLat * Math.cos(decl) * Math.cos(hour);
  const up = sinLat * Math.sin(decl) + cosLat * Math.cos(decl) * Math.cos(hour);
  return [east, up, -north];
}

/**
 * In-game clock: hours of the day plus a day counter for the moon's phase. The moon
 * lags the sun by the phase angle (new moon rises with the sun, full moon at sunset).
 */
export class TimeOfDay {
  constructor(
    /** Hours in [0, 24). 12 = solar noon. */
    public hours: number,
    public day = 0,
  ) {}

  /** Advances by `seconds` of game time. */
  advance(seconds: number): void {
    this.hours += seconds / 3600;
    const days = Math.floor(this.hours / 24);
    this.hours -= days * 24;
    this.day += days;
  }

  state(orbit: OrbitParams): CelestialState {
    const lat = orbit.latitude * DEG;
    const sunHour = ((this.hours - 12) / 24) * 2 * Math.PI;
    // Day 0 at noon is an exact new moon (moon and sun aligned).
    const cycle = ((((this.day + (this.hours - 12) / 24) / LUNAR_CYCLE_DAYS) % 1) + 1) % 1;
    const phaseAngle = cycle * 2 * Math.PI; // 0 new, π full
    // The moon's declination swings with its orbit, tilted against the sun's path.
    const moonDecl = (orbit.sunDeclination + orbit.moonInclination * Math.sin(phaseAngle)) * DEG;
    return {
      sunDir: bodyDirection(lat, orbit.sunDeclination * DEG, sunHour),
      moonDir: bodyDirection(lat, moonDecl, sunHour - phaseAngle),
      moonPhase: (1 - Math.cos(phaseAngle)) / 2,
      starRotation: sunHour,
    };
  }
}
