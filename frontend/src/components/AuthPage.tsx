import { useState, type FormEvent } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth';
import { useToast } from './Toast';

/** 登入 / 註冊頁。訪客的對話不保存，所以登入後沒有東西要轉移。 */
export default function AuthPage({ mode }: { mode: 'login' | 'register' }) {
  const { status, login, register } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const isRegister = mode === 'register';

  if (status === 'user') return <Navigate to="/" replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (isRegister && password.length < 8) {
      setError('密碼至少需要 8 個字元');
      return;
    }
    setBusy(true);
    try {
      if (isRegister) await register(email, password, displayName);
      else await login(email, password);
      toast(isRegister ? '註冊成功，之後的對話都會保存在你的帳號' : '登入成功');
      navigate('/');
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <form className="auth-card" onSubmit={submit} data-testid="auth-form">
        <div className="auth-logo">🍳</div>
        <h1 className="auth-title">{isRegister ? '建立帳號' : '登入 Personal Chef'}</h1>
        <p className="auth-subtitle">
          {isRegister ? '註冊後，對話紀錄、冰箱與飲食偏好都會保存在你的帳號。' : '登入以查看你保存的對話。'}
        </p>

        {isRegister && (
          <label className="auth-field">
            <span>暱稱（選填）</span>
            <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} maxLength={50} autoComplete="nickname" />
          </label>
        )}
        <label className="auth-field">
          <span>Email</span>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required maxLength={254} autoComplete="email" autoFocus />
        </label>
        <label className="auth-field">
          <span>密碼{isRegister && '（至少 8 個字元）'}</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={isRegister ? 8 : undefined}
            maxLength={128}
            autoComplete={isRegister ? 'new-password' : 'current-password'}
          />
        </label>

        {error && <div className="auth-error" role="alert">{error}</div>}

        <button type="submit" className="auth-submit" disabled={busy}>
          {busy ? <span className="spinner-small" /> : isRegister ? '註冊' : '登入'}
        </button>

        <div className="auth-switch">
          {isRegister ? (
            <>已經有帳號？<Link to="/login">登入</Link></>
          ) : (
            <>還沒有帳號？<Link to="/register">註冊</Link></>
          )}
        </div>
        <Link to="/" className="auth-guest">先以訪客身分使用（不保存任何對話紀錄）</Link>
      </form>
    </div>
  );
}
