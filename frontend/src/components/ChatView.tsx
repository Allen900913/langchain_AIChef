import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError, eventsUrl, newIdempotencyKey, refreshSession } from '../api';
import { useAuth } from '../auth';
import {
  newKey, nextToStream, pendingCardKey, reducer,
  type InterruptCard, type Item,
} from '../chatState';
import { threadName, threadStore, useThreads } from '../threads';
import { useToast } from './Toast';
import { AssistantMessage, InterruptCardView, SummaryNote, TaskReply, UserMessage } from './Messages';
import Composer from './Composer';

const STREAM_EVENTS = ['started', 'token', 'tool_start', 'tool_end', 'interrupt', 'status', 'error', 'done'];

const QUICK_ACTIONS = [
  { icon: '🥦', text: '用現有食材推薦料理', prompt: '我手邊有雞胸肉、花椰菜和蒜頭，可以做什麼料理？' },
  { icon: '🍛', text: '學習一道新菜的做法', prompt: '請教我做日式咖哩飯的步驟' },
  { icon: '📖', text: '推薦新手友善食譜', prompt: '推薦一些適合新手的家常菜食譜' },
  { icon: '⏰', text: '快速做出一桌料理', prompt: '如何在30分鐘內做出一桌三菜一湯？' },
];

/**
 * 一個對話的畫面（以 key={threadId} 掛載：切換對話 = 卸載舊畫面、關閉它的事件串流，
 * 任務照樣在 worker 上跑，切回來時由 GET /chat/threads/{id} 接續）。
 * threadId 為 null 是「新對話」首頁：送出第一則訊息後由伺服器配發 thread_id 並跳轉。
 */
export default function ChatView({ threadId, onToggleSidebar }: { threadId: string | null; onToggleSidebar: () => void }) {
  const [state, dispatch] = useReducer(reducer, { loading: !!threadId, items: [] });
  const [lightbox, setLightbox] = useState<string | null>(null);
  const [serverTitle, setServerTitle] = useState<string | null>(null);
  const { status } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();
  const threads = useThreads();
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // ---- 載入歷史 ----
  useEffect(() => {
    if (!threadId) return;
    let alive = true;
    api.getThread(threadId)
      .then((view) => {
        if (!alive) return;
        dispatch({ type: 'loaded', view });
        setServerTitle(view.title);
      })
      .catch((e) => {
        if (!alive) return;
        dispatch({ type: 'reset' });
        // 404：不是自己的或已刪除（伺服器不區分，不洩漏對話是否存在）。
        // 未登入時的 401 也一樣處理；已登入時的 401 = 登入過期，交給 AuthProvider 導去登入頁
        const notFound = e instanceof ApiError && (e.status === 404 || (e.status === 401 && status !== 'user'));
        if (notFound) {
          threadStore.remove(threadId);
          toast('找不到這個對話（可能已刪除，或需要先登入）', true);
          navigate('/', { replace: true });
        } else if (!(e instanceof ApiError && e.status === 401)) {
          toast('載入對話失敗', true);
        }
      });
    return () => {
      alive = false;
    };
  }, [threadId, toast, navigate, status]);

  // ---- 事件串流：永遠只接「第一個還沒結束的任務」，結束後 reducer 讓下一個浮上來 ----
  const target = nextToStream(state.items);
  const targetId = target?.taskId ?? null;
  // 自己重建連線時要從哪個事件接續、已經為了換發 token 重試過幾次（換任務就歸零）
  const streamCursor = useRef<{ taskId: string | null; lastId: string | null; authRetried: boolean }>(
    { taskId: null, lastId: null, authRetried: false },
  );
  const [reconnectSeq, setReconnectSeq] = useState(0);
  useEffect(() => {
    if (!targetId) return;
    const cursor = streamCursor.current;
    if (cursor.taskId !== targetId) streamCursor.current = { taskId: targetId, lastId: null, authRetried: false };
    // 斷線時瀏覽器自動重連並帶上 Last-Event-ID，伺服器從斷點續傳，畫面不會重複或缺漏
    const es = new EventSource(eventsUrl(targetId, streamCursor.current.lastId));
    let finished = false;
    let alive = true;
    for (const type of STREAM_EVENTS) {
      es.addEventListener(type, (ev) => {
        const data = (ev as MessageEvent).data;
        // 'error' 同時是 EventSource 連線錯誤的事件名稱：只有伺服器送來的才帶 data
        if (type === 'error' && typeof data !== 'string') {
          if (es.readyState !== EventSource.CLOSED || finished) return;
          // 連線被拒（多半是重連時 access token 剛好過期 → 401）：EventSource 不會再自己重試，
          // 也看不到狀態碼。換發一次 token 後從最後收到的事件重建連線；仍失敗才判定串流遺失
          if (streamCursor.current.authRetried) {
            dispatch({ type: 'streamLost', taskId: targetId });
            return;
          }
          streamCursor.current.authRetried = true;
          void refreshSession().then((ok) => {
            if (!alive) return;
            if (ok) setReconnectSeq((n) => n + 1);
            else dispatch({ type: 'streamLost', taskId: targetId });
          });
          return;
        }
        const eventId = (ev as MessageEvent).lastEventId;
        if (eventId) streamCursor.current.lastId = eventId;
        streamCursor.current.authRetried = false;
        let payload: unknown = {};
        try {
          payload = JSON.parse(data);
        } catch {
          /* 忽略格式錯誤的事件 */
        }
        dispatch({ type: 'event', taskId: targetId, event: type, data: payload });
        if (type === 'done' || type === 'interrupt') {
          finished = true;
          es.close();
        }
      });
    }
    return () => {
      alive = false;
      es.close();
    };
  }, [targetId, reconnectSeq]);

  // ---- 自動捲動：使用者往上捲去看舊訊息時不打擾 ----
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [state.items]);
  const onScroll = () => {
    const el = scrollRef.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  // ---- 送出：回覆進行中也能送，後端會排隊（enqueue）----
  const send = useCallback(async (
    text: string,
    imageUrl: string | null,
    idempotencyKey: string = newIdempotencyKey(),
  ): Promise<boolean> => {
    const message = text || '請描述這張圖片';
    const key = newKey('task');
    const skippedCard = pendingCardKey(state.items);
    stickToBottom.current = true;
    dispatch({ type: 'addTask', key, taskKind: 'chat', user: { text: message, imageUrl } });
    try {
      const res = await api.sendMessage({ message, image_url: imageUrl, thread_id: threadId }, idempotencyKey);
      if (!res.ephemeral && !threadId && res.thread_id) {
        // 登入使用者的新對話：伺服器配發 thread_id 並記下擁有者，網址驅動跳轉，新畫面從 GET /chat/threads/{id} 接上任務
        threadStore.upsert(res.thread_id, threadName(message));
        void threadStore.refresh();
        navigate(`/chat/${encodeURIComponent(res.thread_id)}`);
        return true;
      }
      // 訪客：不跳轉、不建對話，畫面只靠 task_id 開 SSE 拿輸出；登入使用者：最近使用的對話排到最上面
      if (!res.ephemeral) void threadStore.refresh();
      dispatch({ type: 'submitted', key, taskId: res.task_id, status: res.status, held: res.held });
      // 直接開始執行（沒有排隊）而畫面上還有沒審核的卡片：後端把它當成略過審核
      if (res.status === 'pending' && skippedCard) dispatch({ type: 'card', key: skippedCard, state: 'skipped' });
      if (res.status === 'queued') {
        toast(res.held ? '已排隊：請先處理上方的授權確認' : `已排隊，前一則回覆完成後自動送出（第 ${res.position} 則）`);
      }
      return true;
    } catch (e) {
      dispatch({ type: 'removeTask', key });
      toast(`送出失敗：${(e as Error).message}`, true);
      return false;
    }
  }, [state.items, threadId, navigate, toast, status]);

  // ---- 停止目前的回覆 ----
  const streaming = target && (target.phase === 'waiting' || target.phase === 'running') ? target : null;
  const stop = useCallback(async () => {
    if (!streaming?.taskId) return;
    dispatch({ type: 'cancelling', taskId: streaming.taskId });
    try {
      await api.cancel(streaming.taskId);
    } catch (e) {
      toast(`停止失敗：${(e as Error).message}`, true);
    }
  }, [streaming, toast]);

  // ---- 撤回排隊中的訊息 ----
  const withdraw = useCallback(async (key: string, taskId: string) => {
    try {
      const res = await api.cancel(taskId);
      if (res.status === 'cancelled') {
        dispatch({ type: 'removeTask', key });
        toast('已撤回這則訊息');
      } else if (res.status === 'cancelling') {
        toast('這則訊息已開始處理，正在停止');
      }
    } catch (e) {
      toast(`撤回失敗：${(e as Error).message}`, true);
    }
  }, [toast]);

  // ---- HITL 審核 ----
  const decide = useCallback(async (itemKey: string, card: InterruptCard, decision: 'approve' | 'reject') => {
    if (!threadId || card.state !== 'pending') return;
    dispatch({ type: 'card', key: itemKey, state: 'submitting' });
    const decisions = card.actionRequests.map(() =>
      decision === 'approve' ? { type: 'approve' as const } : { type: 'reject' as const, message: '使用者拒絕執行此工具' },
    );
    try {
      const res = await api.resume(threadId, decisions);
      dispatch({ type: 'card', key: itemKey, state: decision === 'approve' ? 'approved' : 'rejected' });
      const key = newKey('resume');
      dispatch({ type: 'addTask', key, taskKind: 'resume', after: itemKey });
      dispatch({ type: 'submitted', key, taskId: res.task_id, status: 'pending', held: false });
      dispatch({ type: 'unhold' });
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        dispatch({ type: 'card', key: itemKey, state: 'expired' });
        toast(e.message, true);
      } else {
        dispatch({ type: 'card', key: itemKey, state: 'pending' });
        toast(`送出審核失敗：${(e as Error).message}`, true);
      }
    }
  }, [threadId, toast]);

  // ---- 刪除對話（歷史與擁有權紀錄一起刪，刪完回到新對話）----
  const clearChat = useCallback(async () => {
    if (!threadId || !confirm('確定要刪除這個對話嗎？所有歷史訊息都會清除。')) return;
    try {
      await api.deleteThread(threadId);
      threadStore.remove(threadId);
      toast('對話已刪除');
      navigate('/');
    } catch (e) {
      toast(e instanceof ApiError && e.status === 409 ? '回覆進行中，請先停止或稍後再刪除' : `刪除失敗：${(e as Error).message}`, true);
    }
  }, [threadId, toast, navigate]);

  const title = threadId
    ? threads.find((t) => t.id === threadId)?.name || serverTitle || '對話'
    : status === 'user' ? '新對話' : '訪客對話（不保存）';
  const queuedCount = state.items.filter((it) => it.kind === 'task' && (it.phase === 'queued' || it.phase === 'held')).length;

  return (
    <main className="main-content">
      <header className="chat-header">
        <div className="chat-header-left">
          <button type="button" className="sidebar-toggle" onClick={onToggleSidebar}>☰</button>
          <div>
            <div className="chat-title" data-testid="chat-title">{title}</div>
            <div className="chat-status">
              <span className="status-dot" />
              <span>
                {streaming ? (streaming.cancelling ? '停止中…' : '回覆中…') : '線上'}
                {queuedCount > 0 && ` · ${queuedCount} 則排隊中`}
              </span>
            </div>
          </div>
        </div>
        <div className="chat-header-right">
          {threadId && (
            <button type="button" className="header-btn" onClick={clearChat} title="刪除對話">🗑️</button>
          )}
        </div>
      </header>

      <div className="messages-container" ref={scrollRef} onScroll={onScroll}>
        <div className="messages-wrapper" data-testid="messages">
          {!state.loading && state.items.length === 0 && (
            <div className="welcome-screen">
              <div className="welcome-icon">👨‍🍳</div>
              <h1 className="welcome-title">您的私人廚師已就緒</h1>
              <p className="welcome-subtitle">告訴我您手邊有什麼食材，或想做什麼料理？</p>
              <div className="quick-actions">
                {QUICK_ACTIONS.map((q) => (
                  <div key={q.text} className="quick-action" onClick={() => send(q.prompt, null)}>
                    <div className="quick-action-icon">{q.icon}</div>
                    <div className="quick-action-text">{q.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {state.items.map((it) => renderItem(it))}
        </div>
      </div>

      <Composer onSend={send} onStop={stop} streaming={!!streaming} stopping={!!streaming?.cancelling} />

      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <img className="lightbox-image" src={lightbox} alt="image" />
        </div>
      )}
    </main>
  );

  function renderItem(it: Item) {
    if (it.kind === 'history') {
      if (it.role === 'summary') return <SummaryNote key={it.key} content={it.content} />;
      return it.role === 'user'
        ? <UserMessage key={it.key} text={it.content} />
        : <AssistantMessage key={it.key} content={it.content} />;
    }
    if (it.kind === 'card') {
      return (
        <div key={it.key} className="message assistant" data-testid="assistant-message">
          <div className="message-avatar">🍳</div>
          <div className="message-body">
            <div className="message-sender">Chef AI</div>
            <InterruptCardView card={it.card} onDecide={(d) => decide(it.key, it.card, d)} />
          </div>
        </div>
      );
    }
    const badge =
      it.phase === 'sending' ? '傳送中…'
      : it.phase === 'queued' ? '排隊中，前一則回覆完成後自動送出'
      : it.phase === 'held' ? '等待你先完成上方的授權確認'
      : undefined;
    const canWithdraw = (it.phase === 'queued' || it.phase === 'held') && it.taskId;
    return (
      <div key={it.key} className="task-turn" data-testid="task-turn" data-phase={it.phase}>
        {it.user && (
          <UserMessage
            text={it.user.text}
            imageUrl={it.user.imageUrl}
            badge={badge}
            onWithdraw={canWithdraw ? () => withdraw(it.key, it.taskId!) : undefined}
            onImageClick={setLightbox}
          />
        )}
        {!badge && (
          <TaskReply task={it} onDecide={(d) => it.reply.card && decide(it.key, it.reply.card, d)} />
        )}
      </div>
    );
  }
}
