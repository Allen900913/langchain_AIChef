import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Banner, InterruptCard, Reply, TaskItem, ToolCall } from '../chatState';

// 用 action 參數區分動作的工具（inventory、step_tracker）以「工具:action」查，查不到再用工具名
const TOOL_META: Record<string, { name: string; icon: string }> = {
  web_search: { name: '搜尋食譜與烹飪技巧', icon: '🔍' },
  nutrition_lookup: { name: '查詢食材營養成分', icon: '🥗' },
  inventory: { name: '管理冰箱庫存', icon: '🧊' },
  'inventory:get': { name: '查詢冰箱現有食材', icon: '🧊' },
  'inventory:add': { name: '新增食材至冰箱庫存', icon: '📥' },
  'inventory:remove': { name: '從冰箱庫存移除食材', icon: '📤' },
  shopping_list_generate: { name: '比對缺料並產生採購清單', icon: '🛒' },
  profiles_get: { name: '確認個人飲食偏好與設備', icon: '👤' },
  diet_profile_manage: { name: '更新飲食限制與過敏原', icon: '📝' },
  kitchen_profile_manage: { name: '更新廚房設備設定', icon: '🍳' },
  household_profile_manage: { name: '更新家庭用餐偏好', icon: '🏠' },
  set_goal: { name: '記錄當前烹飪目標', icon: '🎯' },
  step_tracker: { name: '步驟引導', icon: '⏱️' },
  'step_tracker:start': { name: '啟動步驟引導模式', icon: '⏱️' },
  'step_tracker:next': { name: '推進至下一個步驟', icon: '⏭️' },
  'step_tracker:current': { name: '確認當前步驟進度', icon: '📌' },
  conversation_search: { name: '回想過去的對話', icon: '🗂️' },
  run_python_in_sandbox: { name: '在沙箱中計算', icon: '🧮' },
};

// react-markdown 預設不渲染原始 HTML，模型輸出裡夾帶的 <script> 等標籤不會被執行
export function Markdown({ text }: { text: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>;
}

export function UserMessage({
  text, imageUrl, badge, onWithdraw, onImageClick,
}: {
  text: string;
  imageUrl?: string | null;
  badge?: string;
  onWithdraw?: () => void;
  onImageClick?: (src: string) => void;
}) {
  return (
    <div className={`message user${badge ? ' is-queued' : ''}`} data-testid="user-message">
      <div className="message-avatar">👤</div>
      <div className="message-body">
        <div className="message-sender">你</div>
        {imageUrl && (
          <img className="message-image" src={imageUrl} alt="uploaded" onClick={() => onImageClick?.(imageUrl)} />
        )}
        <div className="message-content">
          <Markdown text={text} />
        </div>
        {badge && (
          <div className="queue-badge" data-testid="queue-badge">
            <span className="spinner-small" /> {badge}
            {onWithdraw && (
              <button type="button" className="queue-withdraw" onClick={onWithdraw} title="撤回這則訊息">
                撤回
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function AssistantShell({ children, testId }: { children: React.ReactNode; testId?: string }) {
  return (
    <div className="message assistant" data-testid={testId}>
      <div className="message-avatar">🍳</div>
      <div className="message-body">
        <div className="message-sender">Chef AI</div>
        {children}
      </div>
    </div>
  );
}

export function AssistantMessage({ content }: { content: string }) {
  return (
    <AssistantShell testId="assistant-message">
      <div className="message-content">
        <Markdown text={content} />
      </div>
    </AssistantShell>
  );
}

// 較早的對話被摘要壓縮後，原始訊息已經不在 state 裡，這張卡片顯示壓縮成的摘要
export function SummaryNote({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <div className={`summary-note tool-activity-card${expanded ? ' expanded' : ''}`} data-testid="summary-note">
      <div className="tool-card-header" onClick={() => setExpanded(!expanded)}>
        <div className="tool-card-title">
          <span className="tool-card-label">🗜️ 較早的對話已整理成摘要</span>
        </div>
        <button type="button" className="tool-card-toggle">{expanded ? '收合 ▴' : '展開 ▾'}</button>
      </div>
      <div className="tool-card-body">
        <div className="message-content summary-content">
          <Markdown text={content} />
        </div>
      </div>
    </div>
  );
}

export function Typing() {
  return (
    <AssistantShell testId="typing">
      <div className="typing-indicator">
        <span /><span /><span />
      </div>
    </AssistantShell>
  );
}

function ToolCard({ tools }: { tools: ToolCall[] }) {
  const [expanded, setExpanded] = useState(true);
  const running = tools.some((t) => t.status === 'running');
  const awaiting = tools.some((t) => t.status === 'awaiting');
  return (
    <div className={`tool-activity-card${expanded ? ' expanded' : ''}`}>
      <div className="tool-card-header" onClick={() => setExpanded(!expanded)}>
        <div className="tool-card-title">
          <span className={`tool-status-icon ${running || awaiting ? 'pulse-dot' : 'check-dot'}`} />
          <span className="tool-card-label">
            {running ? `正在調用工具... (${tools.length})`
              : awaiting ? `等待你授權工具調用 (${tools.length})`
              : `⚡ 已處理 ${tools.length} 項工具調用`}
          </span>
        </div>
        <button type="button" className="tool-card-toggle">{expanded ? '收合 ▴' : '展開 ▾'}</button>
      </div>
      <div className="tool-card-body">
        {tools.map((t) => {
          const meta = TOOL_META[`${t.tool}:${t.args?.action}`] || TOOL_META[t.tool] || { name: t.tool, icon: '🔧' };
          const args = Object.keys(t.args || {}).length ? JSON.stringify(t.args, null, 2) : '';
          return (
            <div key={t.id} className={`tool-item status-${t.status}`}>
              <div className="tool-item-header">
                <div className="tool-item-info">
                  <span className="tool-item-icon">{meta.icon}</span>
                  <span className="tool-item-name">{meta.name}</span>
                </div>
                {t.status === 'running' && (
                  <span className="tool-item-badge badge-running"><span className="spinner-small" /> 執行中...</span>
                )}
                {t.status === 'success' && <span className="tool-item-badge badge-success">✓ 已完成</span>}
                {t.status === 'error' && <span className="tool-item-badge badge-error">✕ 執行失敗</span>}
                {t.status === 'awaiting' && <span className="tool-item-badge badge-awaiting">🔐 等待授權</span>}
                {t.status === 'skipped' && <span className="tool-item-badge badge-skipped">— 未執行</span>}
              </div>
              {args && (
                <div className="tool-item-args"><span className="args-label">參數：</span><code>{args}</code></div>
              )}
              {t.content && (
                <div className="tool-item-result">
                  <span className="result-label">結果摘要：</span>
                  <pre><code>{t.content}</code></pre>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Banners({ banners }: { banners: Banner[] }) {
  return (
    <>
      {banners.map((b, i) => (
        <div key={i} className={`status-banner banner-${b.level}`} data-testid="status-banner">
          <span className="banner-icon">{b.level === 'error' ? '💥' : b.level === 'info' ? '⏹️' : '⚠️'}</span>
          <span className="banner-text">{b.text}</span>
        </div>
      ))}
    </>
  );
}

const CARD_RESULT: Record<string, string> = {
  approved: '✓ 已同意執行',
  rejected: '✕ 已拒絕操作',
  skipped: '↷ 已略過（你直接送出了新訊息，這個操作沒有執行）',
  expired: '這個審核已失效',
};

export function InterruptCardView({
  card, onDecide,
}: {
  card: InterruptCard;
  onDecide?: (decision: 'approve' | 'reject') => void;
}) {
  const open = card.state === 'pending';
  const cls = card.state === 'approved' ? ' hitl-completed' : card.state === 'pending' || card.state === 'submitting' ? '' : ' hitl-closed';
  return (
    <div className={`hitl-card${cls}`} data-testid="hitl-card" data-state={card.state}>
      <div className="hitl-card-header">
        <span className="hitl-card-icon">🔐</span>
        <span className="hitl-card-title">需要您的授權確認</span>
      </div>
      <div className="hitl-card-body">
        <p className="hitl-card-desc">AI 廚師即將執行以下操作，請確認是否允許：</p>
        <div className="hitl-tool-list">
          {card.actionRequests.map((a, i) => (
            <div key={i} className="hitl-tool" style={{ whiteSpace: 'pre-wrap' }}>
              🔧 {a.description || `${a.name}(${JSON.stringify(a.args)})`}
            </div>
          ))}
        </div>
        {open || card.state === 'submitting' ? (
          <div className="hitl-actions">
            <button type="button" className="hitl-btn hitl-approve" disabled={!open} onClick={() => onDecide?.('approve')}>
              <span className="btn-icon">✓</span> 同意執行
            </button>
            <button type="button" className="hitl-btn hitl-reject" disabled={!open} onClick={() => onDecide?.('reject')}>
              <span className="btn-icon">✕</span> 拒絕操作
            </button>
          </div>
        ) : (
          <div className="hitl-result">{CARD_RESULT[card.state]}</div>
        )}
        {open && <p className="hitl-hint">也可以直接輸入新訊息，等同略過這個操作。</p>}
      </div>
    </div>
  );
}

/** 一個任務的助理回覆：工具卡 → 審核卡 → 文字 → 狀態提示。 */
export function TaskReply({
  task, onDecide,
}: {
  task: TaskItem;
  onDecide: (decision: 'approve' | 'reject') => void;
}) {
  const r: Reply = task.reply;
  const hasContent = r.text || r.tools.length || r.banners.length || r.card;
  const active = task.phase === 'waiting' || task.phase === 'running';
  if (!hasContent) {
    if (active) return <Typing />;
    if (task.phase === 'done') return <AssistantMessage content="（無回應）" />;
    return null;
  }
  return (
    <AssistantShell testId="assistant-message">
      {r.tools.length > 0 && <ToolCard tools={r.tools} />}
      {r.card && <InterruptCardView card={r.card} onDecide={onDecide} />}
      {r.text && (
        <div className="message-content">
          <Markdown text={r.text} />
        </div>
      )}
      {/* 狀態提示（已停止、步數上限、錯誤…）都發生在回答之後，放在文字下方 */}
      <Banners banners={r.banners} />

      {active && !r.text && (
        <div className="typing-indicator inline"><span /><span /><span /></div>
      )}
    </AssistantShell>
  );
}
