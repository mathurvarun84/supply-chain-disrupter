import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Backend URL is fixed here (not via .env) so a fresh clone works with zero
// config: `npm run dev` always proxies /api to the FastAPI server on 8173,
// matching Dockerfile.api and src/api/main.py's default CORS origin.
const BACKEND_URL = "http://127.0.0.1:8173";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
});
