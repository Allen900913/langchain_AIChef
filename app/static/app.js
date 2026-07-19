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

    // Read stream（含 HITL 中斷處理）
    const { assistantContent } = await consumeStream(response);
    const { text: responseText, interrupt } = splitInterrupt(assistantContent);

    if (interrupt) {
      renderInterrupt(interrupt.action_requests);
    } else if (!responseText.trim()) {
      appendMessage('assistant', '（無回應）', true);
    }

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

// 讀取串流回應，邊收邊顯示文字（自動隱藏結尾的 interrupt JSON），回傳完整內容
async function consumeStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let assistantContent = '';
  let messageEl = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    if (!chunk) continue;
    assistantContent += chunk;

    // 顯示時把 interrupt JSON 那段藏起來，只秀純文字
    const { text } = splitInterrupt(assistantContent);
    if (!messageEl) {
      messageEl = appendMessage('assistant', text, true);
    } else {
      updateMessageContent(messageEl, text);
    }
    scrollToBottom();
  }

  return { assistantContent, messageEl };
}

// 後端在 HITL 時會 yield 一段 {"type": "interrupt", ...} 的 JSON。
// 把它從串流文字中切出來：text = 前面的純文字，interrupt = 解析後的物件（或 null）
function splitInterrupt(fullText) {
  const idx = fullText.indexOf('{"type": "interrupt"');
  if (idx === -1) return { text: fullText, interrupt: null };

  let interrupt = null;
  try {
    interrupt = JSON.parse(fullText.slice(idx));
  } catch {
    // JSON 還沒收完，先當作沒有，等下一個 chunk
  }
  return { text: fullText.slice(0, idx), interrupt };
}

// 把待審核的工具呼叫渲染成「同意 / 拒絕」卡片
function renderInterrupt(actionRequests) {
  const wrapper = document.getElementById('messagesWrapper');
  const el = document.createElement('div');
  el.className = 'message assistant';

  const list = actionRequests.map(a =>
    `<div class="hitl-tool">🔧 <strong>${escapeHtml(a.name)}</strong>(${escapeHtml(JSON.stringify(a.args))})</div>`
  ).join('');

  el.innerHTML = `
    <div class="message-avatar">🔐</div>
    <div class="message-body">
      <div class="message-sender">需要你的核准</div>
      <div class="message-content">
        <p>AI 想執行以下工具，請選擇是否允許：</p>
        ${list}
        <div class="hitl-actions">
          <button class="hitl-btn hitl-approve" onclick="resolveInterrupt(this, 'approve', ${actionRequests.length})">✅ 同意</button>
          <button class="hitl-btn hitl-reject" onclick="resolveInterrupt(this, 'reject', ${actionRequests.length})">❌ 拒絕</button>
        </div>
      </div>
    </div>`;
  wrapper.appendChild(el);
  scrollToBottom();
}

// 使用者點同意/拒絕後，呼叫 /chat/resume 繼續執行（並處理可能的下一個 interrupt）
async function resolveInterrupt(btn, decision, count) {
  if (state.isStreaming) return;

  const actionsEl = btn.closest('.hitl-actions');
  actionsEl.querySelectorAll('button').forEach(b => b.disabled = true);
  btn.textContent = decision === 'approve' ? '已同意，執行中…' : '已拒絕';

  // 每個待審核工具都要對應一個 decision
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

  try {
    const response = await fetch(`${API_BASE}/chat/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        thread_id: state.currentThreadId,
        decisions,
      }),
    });

    typingEl.remove();
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const { assistantContent } = await consumeStream(response);
    const { interrupt } = splitInterrupt(assistantContent);

    // 可能還有下一個待審核的工具
    if (interrupt) {
      renderInterrupt(interrupt.action_requests);
    }
  } catch (error) {
    typingEl.remove();
    appendMessage('assistant', `⚠️ 發生錯誤: ${error.message}`, true);
    showToast('繼續執行失敗', true);
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

  // Simple markdown-like formatting
  let html = escapeHtml(text);

  // Code blocks (```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Links
  html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Line breaks
  html = html.replace(/\n/g, '<br>');

  return html;
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
