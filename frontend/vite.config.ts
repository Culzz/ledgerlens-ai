import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the frontend runs on its own port and proxies /api to the
// backend, so the browser sees one origin and there is no CORS to think about.
export default defineConfig({
  plugins: [react()],
  css: {
    // The styles are hand-written CSS with no preprocessor. Pinning an empty
    // postcss config stops Vite from searching parent directories for one —
    // a checkout inside a workspace that has its own postcss/tailwind config
    // would otherwise have that config applied to these styles.
    postcss: { plugins: [] },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
