import { useState } from 'react';
import { Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import { api, ApiError } from './api';
import { useAuth } from './auth';
import AuthPage from './components/AuthPage';
import ChatView from './components/ChatView';
import { useToast } from './components/Toast';
import { threadStore, useThreads } from './threads';

// 網址是當前畫面的唯一真理：/ = 新對話，/chat/:threadId = 該對話，/login、/register = 認證頁
export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage mode="login" />} />
      <Route path="/register" element={<AuthPage mode="register" />} />
      <Route path="/" element={<Layout />} />
      <Route path="/chat/:threadId" element={<Layout />} />
      <Route path="*" element={<Layout />} />
    </Routes>
  );
}

function Layout() {
  const { threadId = null } = useParams();
  const { status } = useAuth();
  // 訪客一直停在 / ：按「新對話」導到同一個網址也要清空畫面，所以 key 帶上這次導覽的 location.key
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <div id="app">
      <Sidebar currentId={threadId} open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className={`sidebar-overlay${sidebarOpen ? ' show' : ''}`} onClick={() => setSidebarOpen(false)} />
      {/* key：換對話或換身分就整個重新掛載，舊畫面的事件串流隨之關閉（任務仍在 worker 上繼續跑）。
          身分確認前不載入對話，避免用過期的 token 讀一次失敗再重來 */}
      {status === 'loading' ? (
        <main className="main-content" />
      ) : (
        <ChatView
          key={`${status}:${threadId ?? `new:${location.key}`}`}
          threadId={threadId}
          onToggleSidebar={() => setSidebarOpen((o) => !o)}
        />
      )}
    </div>
  );
}

function Sidebar({ currentId, open, onClose }: { currentId: string | null; open: boolean; onClose: () => void }) {
  const threads = useThreads();
  const { status } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  async function remove(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    try {
      await api.deleteThread(id);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        toast('這個對話還有回覆在進行中，請先停止或稍後再刪除', true);
        return;
      }
      // 404 = 伺服器上已經沒有（已刪除），照樣從清單拿掉
      if (!(err instanceof ApiError && err.status === 404)) {
        toast(`刪除失敗：${(err as Error).message}`, true);
        return;
      }
    }
    threadStore.remove(id);
    if (id === currentId) {
      const rest = threadStore.get();
      navigate(rest.length ? `/chat/${encodeURIComponent(rest[0].id)}` : '/');
    }
    toast('對話已刪除');
  }

  return (
    <aside className={`sidebar${open ? ' open' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon">🍳</div>
          <span className="logo-text">Personal Chef</span>
        </div>
        <button type="button" className="new-chat-btn" data-testid="new-chat" onClick={() => { onClose(); navigate('/'); }}>
          <span className="icon">＋</span>
          新對話
        </button>
      </div>
      <div className="thread-list">
        <div className="thread-list-title">對話記錄</div>
        {threads.length === 0 && (
          <div className="empty-threads">
            <div className="empty-icon">💬</div>
            <div>{status === 'user' ? '還沒有對話記錄' : '登入後才會保存對話記錄'}</div>
          </div>
        )}
        {threads.map((t) => (
          <div
            key={t.id}
            className={`thread-item${t.id === currentId ? ' active' : ''}`}
            data-testid="thread-item"
            onClick={() => {
              onClose();
              if (t.id !== currentId) navigate(`/chat/${encodeURIComponent(t.id)}`);
            }}
          >
            <span className="thread-icon">💬</span>
            <span className="thread-name">{t.name}</span>
            <button type="button" className="thread-delete" onClick={(e) => remove(t.id, e)} title="刪除">✕</button>
          </div>
        ))}
      </div>
      <AccountPanel onNavigate={onClose} />
    </aside>
  );
}

function AccountPanel({ onNavigate }: { onNavigate: () => void }) {
  const { status, user, logout } = useAuth();
  const navigate = useNavigate();
  const go = (path: string) => {
    onNavigate();
    navigate(path);
  };

  if (status === 'loading') return <div className="sidebar-footer" />;

  if (status === 'user' && user) {
    const name = user.display_name || user.email;
    return (
      <div className="sidebar-footer account-panel" data-testid="account-panel">
        <div className="account-avatar">{name.slice(0, 1).toUpperCase()}</div>
        <div className="account-info">
          <div className="account-name">{name}</div>
          {user.display_name && <div className="account-email">{user.email}</div>}
        </div>
        <button type="button" className="account-logout" onClick={() => void logout()} title="登出">登出</button>
      </div>
    );
  }

  return (
    <div className="sidebar-footer guest-panel" data-testid="guest-panel">
      <div className="guest-notice">訪客模式：不保存任何對話紀錄，重新整理就會消失，也不會記住你的冰箱與偏好。登入後才會保存。</div>
      <div className="guest-actions">
        <button type="button" className="guest-btn primary" onClick={() => go('/login')}>登入</button>
        <button type="button" className="guest-btn" onClick={() => go('/register')}>註冊</button>
      </div>
    </div>
  );
}
