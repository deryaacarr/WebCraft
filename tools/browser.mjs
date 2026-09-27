#!/usr/bin/env node
/**
 * Dev tool: drives the app in headless Chrome over the DevTools protocol (needs the Vite
 * dev server, e.g. `npx vite --port 5199 --strictPort`, and uses `window.webcraft`, which
 * dev builds expose). Fails on any console error (WGSL errors included).
 *
 *   node tools/browser.mjs shots   OUT=dir SHOTS="name|cam|view|extra|patch;…"
 *   node tools/browser.mjs bench   CAM=x,y,z,yaw,pitch EXTRA=&time=12.5 VARIANTS='[["name",{"a.b":1}],…]'
 *
 * shots: `cam` = x,y,z,yawDeg,pitchDeg; `view` = debug view (lit, albedo, shadow, gi, …);
 *        `extra` = more URL parameters (&time=12.5&gi=0); `patch` = comma-separated
 *        config.path=JSONvalue applied after load. Waits WAIT ms (default 10000) for
 *        exposure / accumulation, then saves OUT/name.png (1440×800).
 * bench: per variant restores the config, applies the patch, and prints the minimum over
 *        RUNS (5) of Renderer.benchmarkFrame(ITER = 30) — ms per stage.
 * Env: PORT (5199), SETTLE (ms before measuring, 15000), TIMEOUT (ms, 900000).
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const mode = process.argv[2];
const env = process.env;
const PORT = env.PORT ?? '5199';
const CHROME = env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const profile = join(tmpdir(), `webcraft-chrome-${process.pid}`);
const debugPort = 9400 + (process.pid % 500);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1440,800', 'about:blank'], { stdio: 'ignore' });
const finish = (code) => {
  chrome.kill('SIGKILL');
  setTimeout(() => {
    rmSync(profile, { recursive: true, force: true, maxRetries: 5 });
    process.exit(code);
  }, 300);
};
setTimeout(() => {
  console.log('TIMEOUT');
  finish(1);
}, Number(env.TIMEOUT ?? 900000));

let ws;
for (let i = 0; i < 100 && !ws; i++) {
  try {
    const pages = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
    const page = pages.find((p) => p.type === 'page');
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch {}
  if (!ws) await sleep(200);
}
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  } else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    console.log('[console.error]', m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 600));
    finish(2);
  } else if (m.method === 'Runtime.exceptionThrown') {
    console.log('[exception]', m.params.exceptionDetails.exception?.description?.slice(0, 600));
    finish(2);
  }
});
const send = (method, params = {}) => new Promise((r) => {
  const i = ++id;
  pending.set(i, r);
  ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expression) =>
  (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;
await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 800, deviceScaleFactor: 1, mobile: false });

/** Loads a pose and waits until the app is up and the world around it has streamed in. */
async function load(query) {
  // &power=0: no frame capping / pausing (a headless window never has focus).
  await send('Page.navigate', { url: `http://localhost:${PORT}/?${query}&power=0` });
  for (let i = 0; i < 240 && !(await evaluate('!!window.webcraft')); i++) await sleep(250);
  for (let i = 0; i < 240; i++) {
    const ready = await evaluate('window.webcraft.world.chunkCount > 0 && window.webcraft.streamer.stats.missing === 0');
    if (ready) break;
    await sleep(250);
  }
}

const RESTORE = `(() => { const c = window.webcraft.config; const r = (o, b) => { for (const [k, v] of Object.entries(b)) {
  if (v && typeof v === 'object' && !Array.isArray(v)) r(o[k], v); else o[k] = v; } }; r(c, window.__base); return 1; })()`;
const patchExpr = (entries) => `(() => { const c = window.webcraft.config; for (const [k, v] of ${JSON.stringify(entries)}) {
  const p = k.split('.'); let o = c; for (const q of p.slice(0, -1)) o = o[q]; o[p.at(-1)] = v; } return 1; })()`;

if (mode === 'shots') {
  const out = env.OUT ?? join(tmpdir(), 'webcraft-shots');
  mkdirSync(out, { recursive: true });
  for (const spec of (env.SHOTS ?? '').split(';').filter(Boolean)) {
    const [name, cam, view = 'lit', extra = '', patch = ''] = spec.split('|');
    await load(`cam=${cam}&scale=1&view=${view}${extra}`);
    const entries = patch.split(',').filter(Boolean).map((p) => {
      const [k, v] = p.split('=');
      return [k, JSON.parse(v)];
    });
    if (entries.length) await evaluate(patchExpr(entries));
    await sleep(Number(env.WAIT ?? 10000));
    const shot = (await send('Page.captureScreenshot', { format: 'png' })).result.data;
    writeFileSync(join(out, `${name}.png`), Buffer.from(shot, 'base64'));
    console.log('shot', name);
  }
} else if (mode === 'bench') {
  await load(`cam=${env.CAM ?? '0.5,234,0.5,0,-20'}&scale=1${env.EXTRA ?? ''}`);
  await sleep(Number(env.SETTLE ?? 15000));
  await evaluate('window.__base = structuredClone(window.webcraft.config), 1');
  for (const [name, patch] of JSON.parse(env.VARIANTS ?? '[["base",{}]]')) {
    await evaluate(RESTORE);
    await evaluate(patchExpr(Object.entries(patch)));
    await sleep(1500);
    const results = [];
    for (let k = 0; k < Number(env.RUNS ?? 5); k++) {
      results.push(await evaluate(`window.webcraft.renderer.benchmarkFrame(${Number(env.ITER ?? 30)})`));
      await sleep(300);
    }
    const min = {};
    for (const key of Object.keys(results[0])) {
      const vals = results.map((r) => r[key]).filter((v) => v !== null);
      min[key] = vals.length ? +Math.min(...vals).toFixed(2) : null;
    }
    const total = Object.values(min).reduce((a, b) => a + (b ?? 0), 0);
    console.log(name.padEnd(20), JSON.stringify(min), 'total', total.toFixed(2));
  }
} else {
  console.log('usage: node tools/browser.mjs shots|bench (see the header)');
  finish(1);
}
finish(0);
