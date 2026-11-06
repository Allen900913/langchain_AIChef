import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 開發時 /api 代理到本機的 FastAPI（uvicorn app.main:app --port 8741），與正式環境的 nginx 同源代理一致，
// 前端程式碼一律用相對路徑 /api/v1/...，不需要處理 CORS。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.API_URL || 'http://localhost:8741', changeOrigin: true },
    },
  },
});
