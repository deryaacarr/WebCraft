/// <reference types="vitest/config" />
import { defineConfig } from 'vite';

// SharedArrayBuffer requires a cross-origin isolated context.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  build: { target: 'es2022' },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
