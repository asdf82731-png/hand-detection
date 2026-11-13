import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

// The built app goes to ../webapp, which the FastAPI server (main.py) serves at /.
// `npm run build:demo` makes a server-less demo (sample results) in dist-demo/,
// bundled into one script so it can be inlined into a single HTML page.
export default defineConfig(({mode}) => ({
  plugins: [react()],
  base: mode === 'demo' ? './' : '/',
  build: mode === 'demo'
    ? {outDir: 'dist-demo', emptyOutDir: true, rollupOptions: {output: {inlineDynamicImports: true}}}
    : {outDir: '../webapp', emptyOutDir: true},
  server: {proxy: {'/jobs': 'http://localhost:8000', '/recognize': 'http://localhost:8000', '/static': 'http://localhost:8000'}},
}));
