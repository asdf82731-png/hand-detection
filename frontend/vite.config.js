import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

// The built app goes to ../webapp, which the FastAPI server (main.py) serves at /.
export default defineConfig({
  plugins: [react()],
  build: {outDir: '../webapp', emptyOutDir: true},
  server: {proxy: {'/jobs': 'http://localhost:8000', '/recognize': 'http://localhost:8000', '/static': 'http://localhost:8000'}},
});
