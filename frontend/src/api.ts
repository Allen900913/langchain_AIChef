// 後端 API 的薄封裝。所有路徑都是相對路徑：開發時由 Vite 代理、正式環境由 nginx 同源代理到 FastAPI。
//
// 身分放在 httpOnly cookie（JavaScript 讀不到 token），同源請求由瀏覽器自動帶上，EventSource 也一樣。
// access token 只有 15 分鐘：任何請求回 401 就先用 refresh cookie 換一張新的再重試一次；
// 同時有多個請求遇到 401 時共用同一次換發，不會一口氣送出好幾個 refresh。

export const API_BASE = '/api/v1';

export interface ActionRequest {
  name: string;
  args: Record<string, unknown>;
  description?: string;
}

export interface PendingInterrupt {
  action_requests: ActionRequest[];
}

export interface TaskSummary {
  task_id: string;
  kind: 'chat' | 'resume';
  status: string;
  user_message: string | null;
  image_url: string | null;
}

export interface ThreadView {
  thread_id: string;
  title: string;
  messages: { id: string; role: 'user' | 'assistant' | 'summary'; content: string }[];
  active_task: TaskSummary | null;
  queued_tasks: TaskSummary[];
  held: boolean;
  pending_interrupt: PendingInterrupt | null;
}

export interface ThreadSummary {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SubmitResult {
  task_id: string;
  // 訪客：null（不建立對話，只靠 task_id 拿 SSE 輸出）
  thread_id: string | null;
  ephemeral: boolean;
  status: 'pending' | 'queued';
  held: boolean;
  position: number;
}

export interface User {
  id: string;
  email: string;
  display_name: string | null;
}

export interface MeResult {
  user: User | null;
  guest: boolean;
}

export interface LoginResult {
  user: User;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// ---- 登入過期 ----

const expiredListeners = new Set<() => void>();

/** 換發也失敗（refresh token 過期或被撤銷）時通知：畫面要回到未登入狀態。 */
export function onSessionExpired(listener: () => void): () => void {
  expiredListeners.add(listener);
  return () => expiredListeners.delete(listener);
}

let refreshing: Promise<boolean> | null = null;

/** 用 refresh cookie 換新的 access token；並發呼叫共用同一次請求。回傳是否成功。 */
export function refreshSession(): Promise<boolean> {
  if (!refreshing) {
    refreshing = fetch(`${API_BASE}/auth/refresh`, { method: 'POST', credentials: 'same-origin' })
      .then((res) => res.ok)
      .catch(() => false)
      .finally(() => {
        refreshing = null;
      });
  }
  return refreshing;
}

// 這些路徑的 401 有自己的意思（帳密錯誤、本來就沒登入），不該觸發換發
const NO_REFRESH = ['/auth/login', '/auth/register', '/auth/refresh', '/auth/logout'];

async function request<T>(path: string, init?: RequestInit, retried = false): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  });
  if (res.status === 401 && !retried && !NO_REFRESH.includes(path)) {
    if (await refreshSession()) return request<T>(path, init, true);
    expiredListeners.forEach((l) => l());
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* 非 JSON 回應 */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/**
 * 代表「這一次發送」的 key：同一次發送的重送沿用，使用者發新訊息就換一個（就算文字一樣）。
 * crypto.randomUUID 只有安全來源（HTTPS / localhost）才有；用區網 IP 走 http 測試時退回 getRandomValues。
 */
export function newIdempotencyKey(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
}

export const api = {
  // ---- 認證 ----
  me: () => request<MeResult>('/auth/me'),

  login: (email: string, password: string) =>
    request<LoginResult>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),

  register: (email: string, password: string, displayName: string) =>
    request<LoginResult>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, display_name: displayName || null }),
    }),

  logout: () => request<void>('/auth/logout', { method: 'POST' }),

  // ---- 對話 ----
  listThreads: () => request<{ threads: ThreadSummary[] }>('/chat/threads'),

  getThread: (threadId: string) => request<ThreadView>(`/chat/threads/${encodeURIComponent(threadId)}`),

  deleteThread: (threadId: string) =>
    request<unknown>(`/chat/threads/${encodeURIComponent(threadId)}`, { method: 'DELETE' }),

  sendMessage: (
    body: { message: string; image_url: string | null; thread_id: string | null },
    idempotencyKey: string,
  ) =>
    request<SubmitResult>('/chat/tasks', {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'Idempotency-Key': idempotencyKey },
    }),

  resume: (threadId: string, decisions: { type: 'approve' | 'reject'; message?: string }[]) =>
    request<{ task_id: string }>('/chat/tasks/resume', {
      method: 'POST',
      body: JSON.stringify({ thread_id: threadId, decisions }),
    }),

  cancel: (taskId: string) =>
    request<{ status: string }>(`/chat/tasks/${encodeURIComponent(taskId)}/cancel`, { method: 'POST' }),

  presign: (filename: string) =>
    request<{ uploadUrl: string; accessUrl: string; contentType: string; uploadHeaders: Record<string, string> }>(
      `/gcs/presign?filename=${encodeURIComponent(filename)}`,
    ),

  /** 訪客用：圖片轉成 Base64 Data URL 直接放進訊息，不上傳到雲端（訪客不留任何檔案）。 */
  async imageToDataUrl(file: File): Promise<string> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/chat/upload-image`, { method: 'POST', body: form, credentials: 'same-origin' });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || body.error) throw new ApiError(res.status, body.error || body.detail || `HTTP ${res.status}`);
    return body.image_url as string;
  },
};

/** 事件串流網址。重建連線時帶上最後收到的事件 id，伺服器從斷點續傳。 */
export const eventsUrl = (taskId: string, lastEventId?: string | null) => {
  const base = `${API_BASE}/chat/tasks/${encodeURIComponent(taskId)}/events`;
  return lastEventId ? `${base}?last_event_id=${encodeURIComponent(lastEventId)}` : base;
};
