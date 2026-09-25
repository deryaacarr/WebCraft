export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  /** True when the device was created with 'timestamp-query'. */
  timestampQuery: boolean;
}

export class WebGPUUnsupportedError extends Error {
  override name = 'WebGPUUnsupportedError';
}

const OPTIONAL_FEATURES: GPUFeatureName[] = ['timestamp-query'];

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new WebGPUUnsupportedError('Bu tarayıcı WebGPU desteklemiyor (navigator.gpu yok).');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGPUUnsupportedError('Uygun bir GPU adaptörü bulunamadı.');
  }

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures });

  device.addEventListener('uncapturederror', (event) => {
    console.error('[WebGPU] Yakalanmamış hata:', event.error.message);
  });

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new WebGPUUnsupportedError('Canvas için WebGPU bağlamı oluşturulamadı.');
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const timestampQuery = device.features.has('timestamp-query');
  console.info(
    `[WebGPU] ${adapter.info.vendor || 'unknown'} ${adapter.info.architecture || ''} — ` +
      `format=${format}, timestamp-query=${timestampQuery}`,
  );

  return { adapter, device, context, format, timestampQuery };
}
