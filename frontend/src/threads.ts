// 側邊欄的對話清單：以伺服器為準（GET /chat/threads 只回傳登入使用者自己的對話；訪客沒有紀錄）。
// 這裡只是記憶體快取，讓多個元件共用同一份清單；登入、登出、送出訊息、刪除後重新抓一次。
import { useSyncExternalStore } from 'react';
import { api, type ThreadSummary } from './api';

export interface ThreadMeta {
  id: string;
  name: string;
  updatedAt: string;
}

const listeners = new Set<() => void>();
let threads: ThreadMeta[] = [];
// 多次 refresh 的回應可能亂序抵達：只採用最後一次發出的
let requestSeq = 0;

function write(next: ThreadMeta[]) {
  threads = next;
  listeners.forEach((l) => l());
}

const fromSummary = (t: ThreadSummary): ThreadMeta => ({
  id: t.thread_id,
  name: t.title,
  updatedAt: t.updated_at,
});

export const threadStore = {
  async refresh() {
    const seq = ++requestSeq;
    try {
      const res = await api.listThreads();
      if (seq === requestSeq) write(res.threads.map(fromSummary));
    } catch {
      /* 抓不到就保留目前的清單；登入過期由 api 層統一處理 */
    }
  },
  /** 新對話剛建立：先放進清單頂端，不等 refresh 回來，側邊欄才不會閃一下才出現。 */
  upsert(id: string, name: string) {
    if (threads.some((t) => t.id === id)) return;
    write([{ id, name, updatedAt: new Date().toISOString() }, ...threads]);
  },
  remove(id: string) {
    write(threads.filter((t) => t.id !== id));
  },
  clear() {
    requestSeq++;
    write([]);
  },
  get: () => threads,
};

export function threadName(text: string): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length > 20 ? `${clean.slice(0, 20)}...` : clean;
}

export function useThreads(): ThreadMeta[] {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => threads,
  );
}
