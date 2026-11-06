import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { AuthProvider } from './auth';
import { ToastProvider } from './components/Toast';
import './styles.css';

// 舊版把「目前對話」與對話清單存在 localStorage；現在由網址與伺服器（依登入身分）決定，清掉殘留
try {
  localStorage.removeItem('chef_current_thread');
  localStorage.removeItem('chef_threads');
} catch {
  /* ignore */
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
);
