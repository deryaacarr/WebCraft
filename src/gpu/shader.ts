import { preprocessWgsl } from './preprocess';

// Every .wgsl under src/shaders, imported as raw text and keyed by file name.
const rawSources = import.meta.glob<string>('../shaders/**/*.wgsl', {
  query: '?raw',
  import: 'default',
  eager: true,
});

const sources = new Map<string, string>(
  Object.entries(rawSources).map(([path, src]) => [path.replace('../shaders/', ''), src]),
);

/**
 * Builds a shader module from `src/shaders/<name>`, resolving #include directives.
 * Compilation messages and validation errors are written to the console.
 */
export async function createShaderModule(device: GPUDevice, name: string): Promise<GPUShaderModule> {
  const code = preprocessWgsl(name, (n) => sources.get(n));

  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label: name, code });
  const [info, error] = await Promise.all([module.getCompilationInfo(), device.popErrorScope()]);

  const lines = code.split('\n');
  for (const msg of info.messages) {
    const src = lines[msg.lineNum - 1] ?? '';
    const text = `[WGSL ${msg.type}] ${name}:${msg.lineNum}:${msg.linePos} ${msg.message}\n  ${src}`;
    if (msg.type === 'error') console.error(text);
    else if (msg.type === 'warning') console.warn(text);
    else console.info(text);
  }
  if (error) {
    throw new Error(`Shader "${name}" failed validation: ${error.message}`);
  }
  return module;
}
