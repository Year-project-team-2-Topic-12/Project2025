import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/forward': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/forwardMultiple': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/history': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/stats': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    }
  }
})
