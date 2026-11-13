import '@astryxdesign/core/reset.css';
import '@astryxdesign/core/astryx.css';
import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import {Theme} from '@astryxdesign/core/theme';
import {themes, defaultThemeSlug} from '../astryx-themes';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Theme theme={themes[defaultThemeSlug]}>
      <App />
    </Theme>
  </StrictMode>,
);

if ('serviceWorker' in navigator && import.meta.env.VITE_DEMO !== '1') {
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
}
