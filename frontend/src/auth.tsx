// 目前的身分：載入中 / 未登入（可能是訪客）/ 已登入。
// token 在 httpOnly cookie 裡，前端讀不到也不需要讀；這裡只記「伺服器說我是誰」。
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, onSessionExpired, type User } from './api';
import { useToast } from './components/Toast';
import { threadStore } from './threads';

type Status = 'loading' | 'anonymous' | 'user';

interface AuthContextValue {
  status: Status;
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>('loading');
  const [user, setUser] = useState<User | null>(null);
  const statusRef = useRef<Status>('loading');
  const navigate = useNavigate();
  const toast = useToast();

  const apply = useCallback((next: User | null) => {
    const nextStatus: Status = next ? 'user' : 'anonymous';
    statusRef.current = nextStatus;
    setUser(next);
    setStatus(nextStatus);
    // 換了身分就換一份對話清單（訪客沒有紀錄，清單是空的）
    threadStore.clear();
    void threadStore.refresh();
  }, []);

  useEffect(() => {
    api.me()
      .then((res) => apply(res.user))
      .catch(() => apply(null));
  }, [apply]);

  // refresh token 也失效了：回到未登入，請使用者重新登入
  useEffect(
    () =>
      onSessionExpired(() => {
        if (statusRef.current !== 'user') return;
        apply(null);
        toast('登入已過期，請重新登入', true);
        navigate('/login');
      }),
    [apply, navigate, toast],
  );

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.login(email, password);
    apply(res.user);
  }, [apply]);

  const register = useCallback(async (email: string, password: string, displayName: string) => {
    const res = await api.register(email, password, displayName);
    apply(res.user);
  }, [apply]);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      apply(null);
      navigate('/');
    }
  }, [apply, navigate]);

  const value = useMemo(() => ({ status, user, login, register, logout }), [status, user, login, register, logout]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth 必須在 AuthProvider 裡使用');
  return ctx;
}
