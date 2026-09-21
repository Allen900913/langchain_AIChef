/* ============================================
   Personal Chef AI - Application Logic
   ============================================ */

// --- State ---
const state = {
  currentThreadId: null,
  threads: [],            // [{id, name, createdAt}]
  isStreaming: false,
  currentImageUrl: null,  // base64 data URL for pending image
};

const API_BASE = '/api/v1';

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  loadThreads();
  updateSendButton();

  // Auto-focus input
  document.getElementById('messageInput').focus();
});


// =============================================
//  Thread Management
// =============================================

function generateThreadId() {
  return 'thread_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
}

function createNewThread() {
  const id = generateThreadId();
  const thread = {
    id,
    name: '新對話',
    createdAt: new Date().toISOString(),
  };

  state.threads.unshift(thread);
  state.currentThreadId = id;
  saveThreads();
  renderThreadList();
  clearMessages();
  showWelcomeScreen();
  updateChatTitle('新對話');
  document.getElementById('messageInput').focus();

  // Close sidebar on mobile
  closeSidebar();
}

function switchThread(threadId) {
  if (state.isStreaming) return;

  state.currentThreadId = threadId;
  saveThreads();
  renderThreadList();
  loadChatHistory(threadId);
  closeSidebar();
}

async function deleteThread(threadId, event) {
  event.stopPropagation();

  try {
    await fetch(`${API_BASE}/chat/messages?thread_id=${encodeURIComponent(threadId)}`, {
      method: 'DELETE',
    });
  } catch (e) {
    console.warn('Failed to delete server thread:', e);
  }

  state.threads = state.threads.filter(t => t.id !== threadId);

  if (state.currentThreadId === threadId) {
    if (state.threads.length > 0) {
      switchThread(state.threads[0].id);
    } else {
      state.currentThreadId = null;
      clearMessages();
      showWelcomeScreen();
      updateChatTitle('新對話');
    }
  }

  saveThreads();
  renderThreadList();
  showToast('對話已刪除');
}

async function clearCurrentChat() {
  if (!state.currentThreadId) return;
  if (!confirm('確定要清除此對話的所有歷史訊息嗎？')) return;

  try {
    await fetch(`${API_BASE}/chat/messages?thread_id=${encodeURIComponent(state.currentThreadId)}`, {
      method: 'DELETE',
    });
    clearMessages();
    showWelcomeScreen();
    showToast('對話已清除');
  } catch (e) {
    showToast('清除失敗: ' + e.message, true);
  }
}

function saveThreads() {
  localStorage.setItem('chef_threads', JSON.stringify(state.threads));
  localStorage.setItem('chef_current_thread', state.currentThreadId || '');
}

function loadThreads() {
  try {
    const saved = localStorage.getItem('chef_threads');
    state.threads = saved ? JSON.parse(saved) : [];
    state.currentThreadId = localStorage.getItem('chef_current_thread') || null;
  } catch {
    state.threads = [];
    state.currentThreadId = null;
  }

  renderThreadList();

  if (state.currentThreadId) {
    loadChatHistory(state.currentThreadId);
  }
}

function renderThreadList() {
  const container = document.getElementById('threadList');
  const emptyEl = document.getElementById('emptyThreads');

  // Clear all thread items (keep title and empty state)
  container.querySelectorAll('.thread-item').forEach(el => el.remove());

  if (state.threads.length === 0) {
    emptyEl.style.display = 'block';
    return;
  }

  emptyEl.style.display = 'none';

  state.threads.forEach(thread => {
    const el = document.createElement('div');
    el.className = `thread-item${thread.id === state.currentThreadId ? ' active' : ''}`;
    el.onclick = () => switchThread(thread.id);
    el.innerHTML = `
      <span class="thread-icon">💬</span>
      <span class="thread-name">${escapeHtml(thread.name)}</span>
      <button class="thread-delete" onclick="deleteThread('${thread.id}', event)" title="刪除">✕</button>
    `;
    container.appendChild(el);
  });
}


// =============================================
//  Chat History
// =============================================

async function loadChatHistory(threadId) {
  clearMessages();

  try {
    const res = await fetch(`${API_BASE}/chat/messages?thread_id=${encodeURIComponent(threadId)}`);
    const messages = await res.json();

    if (!messages || messages.length === 0) {
      showWelcomeScreen();
      return;
    }

    hideWelcomeScreen();

    messages.forEach(msg => {
      appendMessage(msg.role, msg.content, false);
    });

    // Update thread name from first user message
    const firstUser = messages.find(m => m.role === 'user');
    if (firstUser) {
      const name = getThreadName(typeof firstUser.content === 'string' ? firstUser.content : '對話');
      updateChatTitle(name);
    }

    scrollToBottom();
  } catch (e) {
    console.warn('Failed to load history:', e);
    showWelcomeScreen();
  }
}


// =============================================
//  Messaging
// =============================================

async function sendMessage() {
  const input = document.getElementById('messageInput');
  const text = input.value.trim();

  if (!text && !state.currentImageUrl) return;
  if (state.isStreaming) return;

  // Ensure thread exists
  if (!state.currentThreadId) {
    createNewThread();
  }

  hideWelcomeScreen();

  // Build user message display
  let userDisplayContent = text;

  // Show user message
  appendMessage('user', userDisplayContent, true, state.currentImageUrl);

  // Update thread name
  if (text) {
    const threadName = getThreadName(text);
    updateThreadName(state.currentThreadId, threadName);
    updateChatTitle(threadName);
  }

  // Clear input
  input.value = '';
  input.style.height = 'auto';
  updateSendButton();

  const imageUrl = state.currentImageUrl;
  removeImage();

  // Show typing indicator
  const typingEl = showTypingIndicator();

  // Start streaming
  state.isStreaming = true;
  updateSendButton();

  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text || '請描述這張圖片',
        image_url: imageUrl || null,
        thread_id: state.currentThreadId,
      }),
    });

    // Remove typing indicator
    typingEl.remove();

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    // Read Typed SSE stream
    await consumeTypedStream(response);

  } catch (error) {
    typingEl.remove();
    appendMessage('assistant', `⚠️ 發生錯誤: ${error.message}`, true);
    showToast('連線失敗，請確認伺服器是否啟動', true);
  } finally {
    state.isStreaming = false;
    updateSendButton();
    scrollToBottom();
  }
}

// =============================================
//  Tool Metadata & Friendly Display
// =============================================

const TOOL_META = {
  web_search: { name: '搜尋食譜與烹飪技巧', icon: '🔍' },
  nutrition_lookup: { name: '查詢食材營養成分', icon: '🥗' },
  inventory_get: { name: '查詢冰箱現有食材', icon: '🧊' },
  inventory_add: { name: '新增食材至冰箱庫存', icon: '📥' },
  inventory_remove: { name: '從冰箱庫存移除食材', icon: '📤' },
  shopping_list_generate: { name: '比對缺料並產生採購清單', icon: '🛒' },
  profiles_get: { name: '確認個人飲食偏好與設備', icon: '👤' },
  diet_profile_manage: { name: '更新飲食限制與過敏原', icon: '📝' },
  kitchen_profile_manage: { name: '更新廚房設備設定', icon: '🍳' },
  household_profile_manage: { name: '更新家庭用餐偏好', icon: '🏠' },
  set_goal: { name: '記錄當前烹飪目標', icon: '🎯' },
  step_tracker_start: { name: '啟動步驟引導模式', icon: '⏱️' },
  step_tracker_next: { name: '推進至下一個步驟', icon: '⏭️' },
  step_tracker_current: { name: '確認當前步驟進度', icon: '📌' },
};


// =============================================
//  Typed SSE Stream Consumer
// =============================================

async function consumeTypedStream(response, existingMessageEl = null) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let assistantText = '';
  let messageEl = existingMessageEl;

  function ensureMessageEl() {
    if (!messageEl) {
      messageEl = appendMessage('assistant', '', true);
    }
    return messageEl;
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';

    for (const part of parts) {
      if (!part.trim()) continue;
      let eventType = 'token';
      let dataText = '';

      for (const line of part.split('\n')) {
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          const content = line.slice(5).trim();
          dataText += (dataText ? '\n' : '') + content;
        }
      }

      if (!dataText) continue;

      let payload;
      try {
        payload = JSON.parse(dataText);
      } catch {
        payload = { delta: dataText };
      }

      switch (eventType) {
        case 'token': {
          const delta = typeof payload === 'string' ? payload : (payload.delta || '');
          if (delta) {
            const el = ensureMessageEl();
            assistantText += delta;
            updateMessageContent(el, assistantText);
            scrollToBottom();
          }
          break;
        }

        case 'tool_start': {
          const el = ensureMessageEl();
          addToolStart(el, payload);
          // 工具開始執行代表尚未進入最終回答階段，清空 tool-call 階段產生的草稿碎片
          if (assistantText) {
            assistantText = '';
            updateMessageContent(el, '');
          }
          scrollToBottom();
          break;
        }

        case 'tool_end': {
          const el = ensureMessageEl();
          updateToolEnd(el, payload);
          scrollToBottom();
          break;
        }

        case 'interrupt': {
          const el = ensureMessageEl();
          renderInterruptCard(el, payload.action_requests);
          // 中斷授權時清空 tool-call 階段產生的未授權草稿文字
          if (assistantText) {
            assistantText = '';
            updateMessageContent(el, '');
          }
          scrollToBottom();
          break;
        }

        case 'status': {
          const el = ensureMessageEl();
          renderStatusBanner(el, payload.message, payload.level || 'warning');
          scrollToBottom();
          break;
        }

        case 'error': {
          const el = ensureMessageEl();
          renderStatusBanner(el, payload.message, 'error');
          scrollToBottom();
          break;
        }

        case 'done': {
          break;
        }

        default: {
          if (payload && payload.delta) {
            const el = ensureMessageEl();
            assistantText += payload.delta;
            updateMessageContent(el, assistantText);
          }
          break;
        }
      }
    }
  }

  // 處理殘留 buffer
  if (buffer.trim()) {
    let eventType = 'token';
    let dataText = '';
    for (const line of buffer.split('\n')) {
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataText += (dataText ? '\n' : '') + line.slice(5).trim();
      }
    }
    if (dataText && eventType === 'token') {
      try {
        const p = JSON.parse(dataText);
        if (p.delta) {
          const el = ensureMessageEl();
          assistantText += p.delta;
          updateMessageContent(el, assistantText);
        }
      } catch {
        const el = ensureMessageEl();
        assistantText += dataText;
        updateMessageContent(el, assistantText);
      }
    }
  }

  if (!assistantText.trim() && messageEl && !messageEl.querySelector('.tool-activity-card') && !messageEl.querySelector('.hitl-card')) {
    updateMessageContent(messageEl, '（無回應）');
  }

  return { messageEl, assistantText };
}


// =============================================
//  Interactive Tool Activity Card
// =============================================

function getOrCreateToolCard(messageEl) {
  const body = messageEl.querySelector('.message-body');
  let card = body.querySelector('.tool-activity-card');
  if (!card) {
    card = document.createElement('div');
    card.className = 'tool-activity-card expanded';
    card.innerHTML = `
      <div class="tool-card-header" onclick="toggleToolCard(this)">
        <div class="tool-card-title">
          <span class="tool-status-icon pulse-dot"></span>
          <span class="tool-card-label">正在調用工具...</span>
        </div>
        <button type="button" class="tool-card-toggle">收合 ▴</button>
      </div>
      <div class="tool-card-body"></div>
    `;
    const contentEl = body.querySelector('.message-content');
    body.insertBefore(card, contentEl);
  }
  return card;
}

function toggleToolCard(headerEl) {
  const card = headerEl.closest('.tool-activity-card');
  if (!card) return;
  card.classList.toggle('expanded');
  const toggleBtn = card.querySelector('.tool-card-toggle');
  if (toggleBtn) {
    toggleBtn.textContent = card.classList.contains('expanded') ? '收合 ▴' : '展開 ▾';
  }
}

function addToolStart(messageEl, toolData) {
  const card = getOrCreateToolCard(messageEl);
  const body = card.querySelector('.tool-card-body');
  const meta = TOOL_META[toolData.tool] || { name: toolData.tool, icon: '🔧' };

  const toolItem = document.createElement('div');
  toolItem.className = 'tool-item status-running';
  const toolId = toolData.id || `tool-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  toolItem.id = `tool-item-${toolId}`;

  let argsStr = '';
  if (toolData.args && Object.keys(toolData.args).length > 0) {
    try {
      argsStr = JSON.stringify(toolData.args, null, 2);
    } catch {
      argsStr = String(toolData.args);
    }
  }

  toolItem.innerHTML = `
    <div class="tool-item-header">
      <div class="tool-item-info">
        <span class="tool-item-icon">${meta.icon}</span>
        <span class="tool-item-name">${escapeHtml(meta.name)}</span>
      </div>
      <span class="tool-item-badge badge-running">
        <span class="spinner-small"></span> 執行中...
      </span>
    </div>
    ${argsStr ? `<div class="tool-item-args"><span class="args-label">參數：</span><code>${escapeHtml(argsStr)}</code></div>` : ''}
  `;
  body.appendChild(toolItem);
  updateToolCardSummary(card);
}

function updateToolEnd(messageEl, toolData) {
  const card = getOrCreateToolCard(messageEl);
  const body = card.querySelector('.tool-card-body');
  let toolItem = toolData.id ? document.getElementById(`tool-item-${toolData.id}`) : null;
  if (!toolItem) {
    const items = body.querySelectorAll('.tool-item.status-running');
    if (items.length > 0) toolItem = items[items.length - 1];
  }

  if (toolItem) {
    toolItem.classList.remove('status-running');
    const isError = toolData.status === 'error';
    toolItem.classList.add(isError ? 'status-error' : 'status-success');

    const badge = toolItem.querySelector('.tool-item-badge');
    if (badge) {
      if (isError) {
        badge.className = 'tool-item-badge badge-error';
        badge.innerHTML = '✕ 執行失敗';
      } else {
        badge.className = 'tool-item-badge badge-success';
        badge.innerHTML = '✓ 已完成';
      }
    }

    if (toolData.content) {
      const resultDiv = document.createElement('div');
      resultDiv.className = 'tool-item-result';
      resultDiv.innerHTML = `
        <span class="result-label">結果摘要：</span>
        <pre><code>${escapeHtml(toolData.content)}</code></pre>
      `;
      toolItem.appendChild(resultDiv);
    }
  }
  updateToolCardSummary(card);
}

function updateToolCardSummary(card) {
  const allItems = card.querySelectorAll('.tool-item');
  const runningItems = card.querySelectorAll('.tool-item.status-running');
  const label = card.querySelector('.tool-card-label');
  const icon = card.querySelector('.tool-status-icon');

  if (runningItems.length > 0) {
    icon.className = 'tool-status-icon pulse-dot';
    label.textContent = `正在調用工具... (${allItems.length})`;
  } else {
    icon.className = 'tool-status-icon check-dot';
    label.textContent = `⚡ 已完成 ${allItems.length} 項工具調用`;
  }
}


// =============================================
//  HITL & Status Cards
// =============================================

function renderInterruptCard(messageEl, actionRequests) {
  const body = messageEl.querySelector('.message-body');
  const existingCard = body.querySelector('.hitl-card');
  if (existingCard) existingCard.remove();

  const card = document.createElement('div');
  card.className = 'hitl-card';

  const list = actionRequests.map(a =>
    a.description
      ? `<div class="hitl-tool" style="white-space:pre-wrap">🔧 ${escapeHtml(a.description)}</div>`
      : `<div class="hitl-tool">🔧 <strong>${escapeHtml(a.name)}</strong>(${escapeHtml(JSON.stringify(a.args))})</div>`
  ).join('');

  card.innerHTML = `
    <div class="hitl-card-header">
      <span class="hitl-card-icon">🔐</span>
      <span class="hitl-card-title">需要您的授權確認</span>
    </div>
    <div class="hitl-card-body">
      <p class="hitl-card-desc">AI 廚師即將執行以下狀態修改操作，請確認是否允許：</p>
      <div class="hitl-tool-list">${list}</div>
      <div class="hitl-actions">
        <button class="hitl-btn hitl-approve" onclick="resolveInterrupt(this, 'approve', ${actionRequests.length})">
          <span class="btn-icon">✓</span> 同意執行
        </button>
        <button class="hitl-btn hitl-reject" onclick="resolveInterrupt(this, 'reject', ${actionRequests.length})">
          <span class="btn-icon">✕</span> 拒絕操作
        </button>
      </div>
    </div>
  `;

  const contentEl = body.querySelector('.message-content');
  body.insertBefore(card, contentEl);
}

function renderStatusBanner(messageEl, text, level = 'warning') {
  const body = messageEl.querySelector('.message-body');
  const banner = document.createElement('div');
  banner.className = `status-banner banner-${level}`;
  const icon = level === 'error' ? '💥' : '⚠️';
  banner.innerHTML = `
    <span class="banner-icon">${icon}</span>
    <span class="banner-text">${escapeHtml(text)}</span>
  `;
  const contentEl = body.querySelector('.message-content');
  body.insertBefore(banner, contentEl);
}

async function resolveInterrupt(btn, decision, count) {
  if (state.isStreaming) return;

  const card = btn.closest('.hitl-card');
  const actionsEl = card.querySelector('.hitl-actions');
  actionsEl.querySelectorAll('button').forEach(b => b.disabled = true);

  if (decision === 'approve') {
    btn.innerHTML = `<span class="spinner-small"></span> 已同意，執行中...`;
    btn.classList.add('active');
  } else {
    btn.innerHTML = `✕ 已拒絕操作`;
    btn.classList.add('active-reject');
  }

  const decisions = [];
  for (let i = 0; i < count; i++) {
    decisions.push(
      decision === 'approve'
        ? { type: 'approve' }
        : { type: 'reject', message: '使用者拒絕執行此工具' }
    );
  }

  state.isStreaming = true;
  updateSendButton();
  const typingEl = showTypingIndicator();
  const messageEl = card.closest('.message');

  try {
    const response = await fetch(`${API_BASE}/chat/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        thread_id: state.currentThreadId,
        decisions,
      }),
    });

    await consumeTypedStream(response, messageEl);

    // 執行完畢後更新按鈕狀態，移除旋轉中圖示
    if (decision === 'approve') {
      btn.innerHTML = `<span class="btn-icon">✓</span> 已授權並執行完成`;
      card.classList.add('hitl-completed');
    } else {
      btn.innerHTML = `<span class="btn-icon">✕</span> 已拒絕操作`;
      card.classList.add('hitl-rejected');
    }
  } catch (error) {
    typingEl.remove();
    appendMessage('assistant', `⚠️ 發生錯誤: ${error.message}`, true);
    showToast('繼續執行失敗', true);
    btn.innerHTML = `⚠️ 執行失敗`;
  } finally {
    state.isStreaming = false;
    updateSendButton();
    scrollToBottom();
  }
}

function sendQuickAction(text) {
  document.getElementById('messageInput').value = text;
  updateSendButton();
  sendMessage();
}


// =============================================
//  Image Upload
// =============================================

function triggerImageUpload() {
  document.getElementById('imageFileInput').click();
}

async function handleImageSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Validate file type
  if (!file.type.startsWith('image/')) {
    showToast('請上傳圖片檔案', true);
    return;
  }

  // Validate size (10MB)
  if (file.size > 10 * 1024 * 1024) {
    showToast('圖片大小不可超過 10MB', true);
    return;
  }

  try {
    // 產生唯一檔名，避免覆蓋
    const ext = file.name.split('.').pop() || 'jpg';
    const uniqueName = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}.${ext}`;

    // Step 1: 向後端取得 GCS 預簽名 URL
    const presignRes = await fetch(`${API_BASE}/gcs/presign?filename=${encodeURIComponent(uniqueName)}`);
    if (!presignRes.ok) throw new Error('無法取得上傳簽名');
    const presignData = await presignRes.json();

    console.log("準備上傳！", presignData.uploadUrl);
    console.log("後端給的 Content-Type 是：", presignData.contentType); // 確認這裡有沒有印出 "image/jpeg"

    // Step 2: 透過預簽名 URL 直接上傳到 GCS
    showToast('正在上傳圖片至雲端...');
    const uploadRes = await fetch(presignData.uploadUrl, {
      method: 'PUT',
      headers: { 'Content-Type': presignData.contentType },
      body: file,
    });

    if (!uploadRes.ok) {
      throw new Error(`GCS 上傳失敗: HTTP ${uploadRes.status}`);
    }

    // Step 3: 儲存帶簽名的讀取 URL，讓 LLM 可以存取圖片
    state.currentImageUrl = presignData.accessUrl;

    // 使用本地檔案做即時預覽（快速，不需額外網路請求）
    const preview = document.getElementById('imagePreview');
    const container = document.getElementById('imagePreviewContainer');
    preview.src = URL.createObjectURL(file);
    container.classList.add('has-image');

    updateSendButton();
    showToast('圖片已上傳至雲端 ☁️');
  } catch (e) {
    showToast('圖片上傳失敗: ' + e.message, true);
  }

  // Reset file input
  event.target.value = '';
}

function removeImage() {
  state.currentImageUrl = null;
  const container = document.getElementById('imagePreviewContainer');
  container.classList.remove('has-image');
  document.getElementById('imagePreview').src = '';
  updateSendButton();
}


// =============================================
//  DOM Helpers
// =============================================

function appendMessage(role, content, animate = true, imageUrl = null) {
  const wrapper = document.getElementById('messagesWrapper');

  const msgEl = document.createElement('div');
  msgEl.className = `message ${role}`;
  if (animate) msgEl.style.animationDuration = '0.3s';

  const avatarEmoji = role === 'user' ? '👤' : '🍳';
  const senderName = role === 'user' ? '你' : 'Chef AI';

  // Process content - handle both string and array content (multimodal)
  let displayContent = '';
  if (typeof content === 'string') {
    displayContent = content;
  } else if (Array.isArray(content)) {
    // Multimodal content from history
    for (const part of content) {
      if (part.type === 'text') {
        displayContent += part.text;
      } else if (part.type === 'image_url') {
        imageUrl = part.image_url.url;
      }
    }
  }

  let imageHtml = '';
  if (imageUrl) {
    imageHtml = `<img class="message-image" src="${escapeHtml(imageUrl)}" alt="uploaded" onclick="showLightbox(this.src)">`;
  }

  msgEl.innerHTML = `
    <div class="message-avatar">${avatarEmoji}</div>
    <div class="message-body">
      <div class="message-sender">${senderName}</div>
      ${imageHtml}
      <div class="message-content">${formatMessage(displayContent)}</div>
    </div>
  `;

  wrapper.appendChild(msgEl);
  return msgEl;
}

function updateMessageContent(msgEl, content) {
  const contentEl = msgEl.querySelector('.message-content');
  if (contentEl) {
    contentEl.innerHTML = formatMessage(content);
  }
}

function formatMessage(text) {
  if (!text) return '';
  // 使用 marked.js 來解析 Markdown（加上 breaks: true 支援一般換行）
  try {
    return marked.parse(text, { breaks: true });
  } catch (e) {
    console.error("Markdown parsing error", e);
    return escapeHtml(text).replace(/\n/g, '<br>');
  }
}

function escapeHtml(text) {
  if (typeof text !== 'string') return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showTypingIndicator() {
  const wrapper = document.getElementById('messagesWrapper');
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="message-avatar">🍳</div>
    <div class="message-body">
      <div class="message-sender">Chef AI</div>
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  wrapper.appendChild(el);
  scrollToBottom();
  return el;
}

function clearMessages() {
  const wrapper = document.getElementById('messagesWrapper');
  wrapper.innerHTML = '';
}

function showWelcomeScreen() {
  let welcome = document.getElementById('welcomeScreen');
  if (!welcome) {
    const wrapper = document.getElementById('messagesWrapper');
    wrapper.innerHTML = `
      <div class="welcome-screen" id="welcomeScreen">
        <div class="welcome-icon">👨‍🍳</div>
        <h1 class="welcome-title">您的私人廚師已就緒</h1>
        <p class="welcome-subtitle">告訴我您手邊有什麼食材，或想做什麼料理？</p>
        <div class="quick-actions">
          <div class="quick-action" onclick="sendQuickAction('我手邊有雞胸肉、花椰菜和蒜頭，可以做什麼料理？')">
            <div class="quick-action-icon">🥦</div>
            <div class="quick-action-text">用現有食材推薦料理</div>
          </div>
          <div class="quick-action" onclick="sendQuickAction('請教我做日式咖哩飯的步驟')">
            <div class="quick-action-icon">🍛</div>
            <div class="quick-action-text">學習一道新菜的做法</div>
          </div>
          <div class="quick-action" onclick="sendQuickAction('推薦一些適合新手的家常菜食譜')">
            <div class="quick-action-icon">📖</div>
            <div class="quick-action-text">推薦新手友善食譜</div>
          </div>
          <div class="quick-action" onclick="sendQuickAction('如何在30分鐘內做出一桌三菜一湯？')">
            <div class="quick-action-icon">⏰</div>
            <div class="quick-action-text">快速做出一桌料理</div>
          </div>
        </div>
      </div>
    `;
  }
}

function hideWelcomeScreen() {
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.remove();
}

function scrollToBottom() {
  const container = document.getElementById('messagesContainer');
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
}

function updateChatTitle(title) {
  document.getElementById('chatTitle').textContent = title;
}

function getThreadName(text) {
  if (typeof text !== 'string') return '對話';
  const clean = text.replace(/\n/g, ' ').trim();
  return clean.length > 20 ? clean.slice(0, 20) + '...' : clean;
}

function updateThreadName(threadId, name) {
  const thread = state.threads.find(t => t.id === threadId);
  if (thread && thread.name === '新對話') {
    thread.name = name;
    saveThreads();
    renderThreadList();
  }
}


// =============================================
//  UI Interactions
// =============================================

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
  updateSendButton();
}

function handleKeyDown(event) {
  if (event.key === 'Enter' && event.ctrlKey) {
    event.preventDefault();
    sendMessage();
  }
}

function updateSendButton() {
  const input = document.getElementById('messageInput');
  const btn = document.getElementById('sendBtn');
  const hasContent = input.value.trim().length > 0 || state.currentImageUrl;
  btn.disabled = !hasContent || state.isStreaming;
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  sidebar.classList.toggle('open');
  overlay.classList.toggle('show');
}

function closeSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  sidebar.classList.remove('open');
  overlay.classList.remove('show');
}


// =============================================
//  Lightbox
// =============================================

function showLightbox(src) {
  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.onclick = () => overlay.remove();
  overlay.innerHTML = `<img class="lightbox-image" src="${src}" alt="image">`;
  document.body.appendChild(overlay);
}


// =============================================
//  Toast Notifications
// =============================================

function showToast(message, isError = false) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast${isError ? ' error' : ''}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3000);
}
