import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the frontend runs on its own port and proxies /api to the
// backend, so the browser sees one origin and there is no CORS to think about.
export default defineConfig({
  plugins: [react()],
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
