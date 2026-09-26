import { DEBUG_VIEWS, type DebugView } from '../config';

/**
 * Developer overrides from the page URL, for reproducible views and benchmarks:
 *   ?cam=x,y,z,yawDeg,pitchDeg   camera pose
 *   &view=steps                  debug view
 *   &scale=1                     render scale
 *   &wg=16x8                     primary-ray workgroup size (benchmarking)
 *   &tile=4                      depth prepass tile edge (benchmarking)
 *   &dmax=8                      distance field max distance (benchmarking)
 *   &time=18.5                   time of day in hours; also pauses the clock
 *   &spin=90&fly=20              scripted camera turn (°/s) and forward flight (blocks/s)
 */
export interface UrlOverrides {
  camera?: { position: [number, number, number]; yawDeg: number; pitchDeg: number };
  view?: DebugView;
  renderScale?: number;
  workgroup?: [number, number];
  prepassTile?: number;
  distanceMax?: number;
  timeOfDay?: number;
  cameraSpin?: number;
  cameraFly?: number;
}

export function parseUrlOverrides(search: string): UrlOverrides {
  const params = new URLSearchParams(search);
  const out: UrlOverrides = {};
  const cam = params.get('cam')?.split(',').map(Number);
  if (cam && cam.length === 5 && cam.every(Number.isFinite)) {
    const [x, y, z, yaw, pitch] = cam as [number, number, number, number, number];
    out.camera = { position: [x, y, z], yawDeg: yaw, pitchDeg: pitch };
  }
  const view = params.get('view');
  if (view && (DEBUG_VIEWS as readonly string[]).includes(view)) out.view = view as DebugView;
  const scale = Number(params.get('scale'));
  if (params.has('scale') && scale > 0) out.renderScale = scale;
  const wg = params.get('wg')?.split('x').map(Number);
  if (wg && wg.length === 2 && wg.every((n) => Number.isInteger(n) && n > 0)) out.workgroup = [wg[0]!, wg[1]!];
  const positiveInt = (name: string) => {
    const v = Number(params.get(name));
    return params.has(name) && Number.isInteger(v) && v > 0 ? v : undefined;
  };
  const tile = positiveInt('tile');
  if (tile) out.prepassTile = tile;
  const time = Number(params.get('time'));
  if (params.has('time') && Number.isFinite(time)) out.timeOfDay = ((time % 24) + 24) % 24;
  const spin = Number(params.get('spin'));
  if (params.has('spin') && Number.isFinite(spin)) out.cameraSpin = spin;
  const fly = Number(params.get('fly'));
  if (params.has('fly') && Number.isFinite(fly)) out.cameraFly = fly;
  const dmax = positiveInt('dmax');
  if (dmax) out.distanceMax = Math.min(dmax, 254);

  return out;
}
