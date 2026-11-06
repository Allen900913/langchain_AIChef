// 單一對話畫面的狀態：歷史訊息 + 一串「任務」（每則送出的訊息 / 每次審核各是一個任務）。
//
// 任務依送出順序排在列表裡，後端也依同樣順序在同一個 thread 上逐一執行（enqueue），
// 所以前端只要永遠接上「列表中第一個還沒結束的任務」的事件串流，跑完再換下一個即可。

import type { ActionRequest, PendingInterrupt, ThreadView } from './api';

export type Phase =
  | 'sending'      // POST 還沒回來
  | 'queued'       // 排在前一則後面
  | 'held'         // 排隊中，但前面停在 HITL 審核，要等使用者先審核
  | 'waiting'      // 輪到它了，等 worker 開始
  | 'running'
  | 'done'
  | 'interrupted'  // 停在 HITL 審核
  | 'cancelled'
  | 'failed';

export const UNFINISHED: Phase[] = ['queued', 'waiting', 'running'];

export interface ToolCall {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  // awaiting = 等使用者審核；skipped = 審核被拒絕或略過，沒有執行
  status: 'running' | 'awaiting' | 'success' | 'error' | 'skipped';
  content?: string;
}

export interface Banner {
  level: 'info' | 'warning' | 'error';
  text: string;
}

export type CardState = 'pending' | 'submitting' | 'approved' | 'rejected' | 'skipped' | 'expired';

export interface InterruptCard {
  actionRequests: ActionRequest[];
  state: CardState;
}

export interface Reply {
  text: string;
  tools: ToolCall[];
  banners: Banner[];
  card?: InterruptCard;
}

export type Item =
  | { kind: 'history'; key: string; role: 'user' | 'assistant' | 'summary'; content: string }
  // 重新載入時 checkpointer 裡待審核的 HITL（不屬於任何還在 Redis 的任務）
  | { kind: 'card'; key: string; card: InterruptCard }
  | {
      kind: 'task';
      key: string;
      taskId: string | null;
      taskKind: 'chat' | 'resume';
      user?: { text: string; imageUrl?: string | null };
      phase: Phase;
      cancelling?: boolean;
      reply: Reply;
    };

export type TaskItem = Extract<Item, { kind: 'task' }>;

export interface ChatState {
  loading: boolean;
  items: Item[];
}

const emptyReply = (): Reply => ({ text: '', tools: [], banners: [] });
let keySeq = 0;
export const newKey = (prefix: string) => `${prefix}-${Date.now()}-${keySeq++}`;

export type Action =
  | { type: 'loaded'; view: ThreadView }
  | { type: 'reset' }
  | { type: 'addTask'; key: string; taskKind: 'chat' | 'resume'; user?: TaskItem['user']; after?: string }
  | { type: 'submitted'; key: string; taskId: string; status: 'pending' | 'queued'; held: boolean }
  | { type: 'removeTask'; key: string }
  | { type: 'event'; taskId: string; event: string; data: any }
  | { type: 'streamLost'; taskId: string }
  | { type: 'cancelling'; taskId: string }
  | { type: 'card'; key: string; state: CardState }
  | { type: 'unhold' };

function cardFromInterrupt(pi: PendingInterrupt): InterruptCard {
  return { actionRequests: pi.action_requests, state: 'pending' };
}

export function fromView(view: ThreadView): Item[] {
  const items: Item[] = view.messages.map((m) => ({
    kind: 'history',
    key: `h-${m.id}`,
    role: m.role,
    content: m.content,
  }));
  if (view.pending_interrupt) {
    items.push({ kind: 'card', key: newKey('card'), card: cardFromInterrupt(view.pending_interrupt) });
  }
  const toTask = (t: ThreadView['queued_tasks'][number], phase: Phase): TaskItem => ({
    kind: 'task',
    key: `t-${t.task_id}`,
    taskId: t.task_id,
    taskKind: t.kind,
    user: t.kind === 'chat' ? { text: t.user_message || '', imageUrl: t.image_url } : undefined,
    phase,
    reply: emptyReply(),
  });
  if (view.active_task) {
    items.push(toTask(view.active_task, view.active_task.status === 'processing' ? 'running' : 'waiting'));
  }
  for (const t of view.queued_tasks) items.push(toTask(t, view.held ? 'held' : 'queued'));
  return items;
}

/** 目前畫面上還沒處理的審核卡片（送出新訊息時，若直接開始執行就代表使用者略過了它）。 */
export function pendingCardKey(items: Item[]): string | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const it = items[i];
    const card = it.kind === 'card' ? it.card : it.kind === 'task' ? it.reply.card : undefined;
    if (card && (card.state === 'pending' || card.state === 'submitting')) return it.key;
  }
  return null;
}

/** 應該接上事件串流的任務：列表中第一個還沒結束、且不是在等審核的任務。 */
export function nextToStream(items: Item[]): TaskItem | null {
  for (const it of items) {
    if (it.kind !== 'task' || !it.taskId) continue;
    if (it.phase === 'held') return null;
    if (UNFINISHED.includes(it.phase)) return it;
  }
  return null;
}

function mapTask(items: Item[], taskId: string, fn: (t: TaskItem) => TaskItem): Item[] {
  return items.map((it) => (it.kind === 'task' && it.taskId === taskId ? fn(it) : it));
}

function setToolStatus(t: TaskItem, from: ToolCall['status'][], to: ToolCall['status']): TaskItem {
  if (!t.reply.tools.some((x) => from.includes(x.status))) return t;
  return { ...t, reply: { ...t.reply, tools: t.reply.tools.map((x) => (from.includes(x.status) ? { ...x, status: to } : x)) } };
}

/** 審核通過後，工具在 resume 任務裡才真正執行：把結果回填到發出呼叫的那則訊息的工具卡。 */
function fillToolElsewhere(items: Item[], taskId: string, data: any): Item[] | null {
  const own = items.find((it) => it.kind === 'task' && it.taskId === taskId) as TaskItem | undefined;
  if (!data.id || own?.reply.tools.some((x) => x.id === data.id)) return null;
  const owner = items.find((it) => it.kind === 'task' && it.reply.tools.some((x) => x.id === data.id));
  if (!owner) return null;
  const status: ToolCall['status'] = data.status === 'error' ? 'error' : 'success';
  return items.map((it) =>
    it === owner && it.kind === 'task'
      ? { ...it, reply: { ...it.reply, tools: it.reply.tools.map((x) => (x.id === data.id ? { ...x, status, content: data.content } : x)) } }
      : it,
  );
}

function applyEvent(t: TaskItem, event: string, data: any): TaskItem {
  const reply = { ...t.reply };
  switch (event) {
    case 'started':
      return { ...t, phase: 'running' };
    case 'token':
      reply.text += data.delta || '';
      break;
    case 'tool_start':
      // 工具開始代表還沒進入最終回答，清掉 tool-call 階段的草稿碎片
      reply.text = '';
      reply.tools = [...reply.tools, { id: data.id || newKey('tool'), tool: data.tool, args: data.args || {}, status: 'running' }];
      break;
    case 'tool_end': {
      const tools = [...reply.tools];
      let idx = tools.findIndex((x) => x.id === data.id);
      if (idx < 0) idx = tools.map((x) => x.status).lastIndexOf('running');
      if (idx >= 0) tools[idx] = { ...tools[idx], status: data.status === 'error' ? 'error' : 'success', content: data.content };
      reply.tools = tools;
      break;
    }
    case 'status':
      reply.banners = [...reply.banners, { level: data.level || 'warning', text: data.message }];
      break;
    case 'error':
      reply.banners = [...reply.banners, { level: 'error', text: data.message }];
      break;
    case 'interrupt':
      reply.text = '';
      reply.card = { actionRequests: data.action_requests || [], state: 'pending' };
      // 待審核的工具還沒執行：從「執行中」改成「等待授權」
      reply.tools = reply.tools.map((x) => (x.status === 'running' ? { ...x, status: 'awaiting' } : x));
      return { ...t, phase: 'interrupted', reply };
    case 'done': {
      const phase: Phase = data.status === 'cancelled' ? 'cancelled' : data.status === 'failed' ? 'failed' : 'done';
      return { ...t, phase, cancelling: false, reply };
    }
  }
  return { ...t, phase: t.phase === 'waiting' || t.phase === 'queued' ? 'running' : t.phase, reply };
}

export function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'reset':
      return { loading: false, items: [] };

    case 'loaded':
      return { loading: false, items: fromView(action.view) };

    case 'addTask': {
      const item: TaskItem = {
        kind: 'task', key: action.key, taskId: null, taskKind: action.taskKind,
        user: action.user, phase: 'sending', reply: emptyReply(),
      };
      const items = [...state.items];
      // 審核任務插在卡片正後方：它要先於排隊中的訊息執行，畫面順序也要跟執行順序一致
      const at = action.after ? items.findIndex((it) => it.key === action.after) : -1;
      if (at >= 0) items.splice(at + 1, 0, item);
      else items.push(item);
      return { ...state, items };
    }

    case 'submitted':
      return {
        ...state,
        items: state.items.map((it) =>
          it.kind === 'task' && it.key === action.key
            ? { ...it, taskId: action.taskId, phase: action.status === 'pending' ? 'waiting' : action.held ? 'held' : 'queued' }
            : it,
        ),
      };

    case 'removeTask':
      return { ...state, items: state.items.filter((it) => it.key !== action.key) };

    case 'event': {
      if (action.event === 'tool_end') {
        const filled = fillToolElsewhere(state.items, action.taskId, action.data);
        if (filled) return { ...state, items: filled };
      }
      let items = mapTask(state.items, action.taskId, (t) => applyEvent(t, action.event, action.data));
      if (action.event === 'interrupt') {
        // 後端在 HITL 待審核時暫停該對話的佇列：排在後面的訊息改顯示「等待審核」
        items = items.map((it) => (it.kind === 'task' && it.phase === 'queued' ? { ...it, phase: 'held' } : it));
      }
      return { ...state, items };
    }

    case 'streamLost':
      return {
        ...state,
        items: mapTask(state.items, action.taskId, (t) => ({
          ...t,
          phase: 'failed',
          reply: { ...t.reply, banners: [...t.reply.banners, { level: 'error', text: '無法取得這則訊息的處理結果（可能已過期）。' }] },
        })),
      };

    case 'cancelling':
      return { ...state, items: mapTask(state.items, action.taskId, (t) => ({ ...t, cancelling: true })) };

    case 'card':
      return {
        ...state,
        items: state.items.map((it) => {
          if (it.key !== action.key) return it;
          if (it.kind === 'card') return { ...it, card: { ...it.card, state: action.state } };
          if (it.kind === 'task' && it.reply.card) {
            let t: TaskItem = { ...it, reply: { ...it.reply, card: { ...it.reply.card, state: action.state } } };
            if (action.state === 'approved') t = setToolStatus(t, ['awaiting'], 'running');
            if (action.state === 'rejected' || action.state === 'skipped' || action.state === 'expired') {
              t = setToolStatus(t, ['awaiting', 'running'], 'skipped');
            }
            return t;
          }
          return it;
        }),
      };

    case 'unhold':
      return {
        ...state,
        items: state.items.map((it) => (it.kind === 'task' && it.phase === 'held' ? { ...it, phase: 'queued' } : it)),
      };
  }
}
