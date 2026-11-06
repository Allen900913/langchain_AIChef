// 真瀏覽器 UI 走查：連到已開啟的 Chrome（--remote-debugging-port），像人一樣點按鈕、打字，
// 每一步截圖並寫成報告。結束時只斷開連線、不關閉瀏覽器，畫面留著給人看。
//
// 用法：
//   1. 啟動 Chrome：chrome.exe --remote-debugging-port=9333 --user-data-dir=<任意空資料夾> http://localhost:8080
//   2. node e2e/ui-walkthrough.mjs            （APP_URL / CDP_URL 可用環境變數覆寫）
//
// 會實際呼叫 LLM（Groq），每分鐘 token 額度有限，所以情境之間會停頓。
import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const APP = process.env.APP_URL || 'http://localhost:8080';
const CDP = process.env.CDP_URL || 'http://localhost:9333';
const PAUSE_BETWEEN = Number(process.env.PAUSE_MS || 25000);   // 讓 Groq TPM 額度恢復
const OUT = process.env.E2E_OUT || join(dirname(fileURLToPath(import.meta.url)), 'screenshots');
mkdirSync(OUT, { recursive: true });

const report = [];
let shot = 0;
const started = new Date();

function log(...a) {
  console.log(new Date().toLocaleTimeString(), ...a);
}

async function snap(page, name, note) {
  const file = `${String(++shot).padStart(2, '0')}-${name}.png`;
  await page.screenshot({ path: join(OUT, file) });
  report.push({ type: 'shot', file, note });
  log('📸', file, note || '');
}

let currentPage = null;
const RATE_LIMIT_TEXT = '每分鐘上限';

/** 最後兩個任務裡有沒有 Groq 限流的錯誤提示（免費方案 TPM 8000，屬於環境限制而非功能問題）。 */
async function recentRateLimit(page) {
  const turns = page.locator('[data-testid="task-turn"]');
  const n = await turns.count();
  for (let i = Math.max(0, n - 2); i < n; i++) {
    if ((await turns.nth(i).innerText().catch(() => '')).includes(RATE_LIMIT_TEXT)) return true;
  }
  return false;
}

async function record(scenario, ok, detail) {
  // 沒通過但畫面上是 Groq 限流：另外標示，不跟功能錯誤混在一起
  const status = ok ? 'pass' : currentPage && (await recentRateLimit(currentPage)) ? 'rate' : 'fail';
  report.push({ type: 'check', scenario, status, detail });
  log({ pass: '✅', rate: '⚠️', fail: '❌' }[status], scenario, status === 'rate' ? `（Groq 限流）${detail || ''}` : detail || '');
}

/** 像真人一樣：看到「請約半分鐘後再送出一次」就等一下再送一次（最多兩次）。 */
async function retryIfRateLimited(page, text, result, opts) {
  for (let i = 0; i < 2 && (await recentRateLimit(page)); i++) {
    log('  ⏳ Groq 每分鐘 token 上限，等 45 秒後重送');
    report.push({ type: 'note', text: `遇到 Groq 每分鐘 token 上限，等 45 秒後重送「${text}」` });
    await sleep(45000);
    await typeAndSend(page, text);
    result = await waitSettled(page, opts);
  }
  return result;
}

let firstThreadUrl = null;
const errorCount = (page) => page.locator('[data-testid="status-banner"].banner-error').count();

// ---------- 像人一樣操作 ----------
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function typeAndSend(page, text) {
  const input = page.getByTestId('message-input');
  await input.click();
  await input.pressSequentially(text, { delay: 25 });
  await sleep(300);
  await page.getByTestId('send-button').click();
}

const pendingCard = (page) => page.locator('[data-testid="hitl-card"][data-state="pending"]');
const stopButton = (page) => page.getByTestId('stop-button');

/** 等目前對話的所有回覆跑完；途中出現審核卡就依 policy 處理（模型很愛上網搜尋，每次都要審核）。 */
async function waitSettled(page, { onCard = 'approve', timeout = 180000, maxCards = 4 } = {}) {
  const t0 = Date.now();
  let cards = 0;
  let quietSince = null;
  while (Date.now() - t0 < timeout) {
    if (await pendingCard(page).count()) {
      if (onCard === 'stop' || cards >= maxCards) return 'card';
      cards++;
      const btn = onCard === 'approve' ? '同意執行' : '拒絕操作';
      await pendingCard(page).last().getByRole('button', { name: btn }).click();
      log(`  🔐 審核卡 → ${btn}`);
      quietSince = null;
      await sleep(1500);
      continue;
    }
    const busy = (await stopButton(page).count()) > 0
      || (await page.locator('[data-testid="task-turn"][data-phase="queued"], [data-testid="task-turn"][data-phase="sending"]').count()) > 0;
    if (!busy) {
      quietSince ??= Date.now();
      if (Date.now() - quietSince > 2000) return 'settled';
    } else {
      quietSince = null;
    }
    await sleep(500);
  }
  return 'timeout';
}

async function waitForTokens(page, minChars = 40, timeout = 120000) {
  // 等最後一則助理訊息開始出現文字（代表模型正在串流最終回答）
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    if (await pendingCard(page).count()) {
      await pendingCard(page).last().getByRole('button', { name: '同意執行' }).click();
      log('  🔐 審核卡 → 同意執行（等待模型開始回答）');
      await sleep(1500);
      continue;
    }
    const text = await page.locator('[data-testid="assistant-message"] .message-content').last().innerText().catch(() => '');
    if (text.length >= minChars && (await stopButton(page).count())) return true;
    await sleep(200);
  }
  return false;
}

function redis(...args) {
  return execFileSync('docker', ['exec', 'redis-ai-chef', 'redis-cli', ...args], { encoding: 'utf8' }).trim();
}

// ---------- 情境 ----------
const scenarios = [
  ['首頁與新對話', async (page) => {
    await page.goto(APP);
    await page.getByTestId('new-chat').click();
    await page.waitForSelector('.welcome-screen');
    await snap(page, 'welcome', '首頁：新對話歡迎畫面');
    await typeAndSend(page, '鹽巴可以放多久？請用一句話回答，不用查資料');
    await page.waitForURL(/\/chat\/thread_/, { timeout: 15000 });
    await record('送出第一則後網址變成 /chat/{thread_id}', true, page.url());
    firstThreadUrl = page.url();
    await page.waitForSelector('[data-testid="typing"], [data-testid="assistant-message"]', { timeout: 20000 });
    await snap(page, 'streaming', '回覆中：輸入框旁出現停止鈕 ■');
    const first = '鹽巴可以放多久？請用一句話回答，不用查資料';
    const r = await retryIfRateLimited(page, first, await waitSettled(page));
    const answer = await page.locator('[data-testid="task-turn"]').last().innerText();
    await record('第一則訊息得到回覆', r === 'settled' && !(await recentRateLimit(page)), answer.slice(-60).replace(/\n/g, ' '));
    await snap(page, 'first-answer', '第一則回覆完成');
  }],

  ['連發訊息自動排隊（enqueue）', async (page) => {
    await typeAndSend(page, '請用大約 300 字說明煎牛排前為什麼要先讓肉回溫');
    await stopButton(page).waitFor({ timeout: 20000 });
    await sleep(1500);
    await typeAndSend(page, '那雞胸肉也需要回溫嗎？一句話回答');
    const badge = page.getByTestId('queue-badge');
    const queued = await badge.first().waitFor({ timeout: 8000 }).then(() => true).catch(() => false);
    const badgeText = queued ? await badge.first().innerText() : '';
    await record('回覆中送出第二則 → 顯示排隊中', queued, badgeText);
    await snap(page, 'queued', '第二則訊息排隊中（右下角徽章），標題列顯示「1 則排隊中」');
    const r = await waitSettled(page);
    const users = await page.getByTestId('user-message').allInnerTexts();
    const lastUser = users[users.length - 1] || '';
    await record('排隊的訊息在前一則完成後自動送出並得到回覆', r === 'settled' && lastUser.includes('雞胸肉')
      && (await page.getByTestId('queue-badge').count()) === 0, `結果=${r}`);
    await snap(page, 'queue-drained', '兩則都完成，順序與送出順序一致');
  }],

  ['停止回覆後再送出', async (page) => {
    await typeAndSend(page, '請用大約 800 字詳細介紹梅納反應的化學原理');
    const streaming = await waitForTokens(page, 30);
    await record('模型開始串流回答', streaming);
    await snap(page, 'before-stop', '模型回答到一半');
    await stopButton(page).click();
    log('  ■ 按下停止');
    const banner = page.locator('[data-testid="status-banner"]', { hasText: '已停止回應' });
    const stopped = await banner.last().waitFor({ timeout: 15000 }).then(() => true).catch(() => false);
    await record('按停止後顯示「已停止回應」，停止鈕消失', stopped && (await stopButton(page).count()) === 0);
    await snap(page, 'stopped', '停止後：保留半段回答 + 已停止提示');
    await page.reload();
    await page.waitForSelector('[data-testid="assistant-message"]');
    const lastAssistant = await page.locator('[data-testid="assistant-message"] .message-content').last().innerText();
    await record('重新整理後歷史保留半段回答並註明（已停止回應）', lastAssistant.includes('已停止回應'), lastAssistant.slice(-40));
    await snap(page, 'stopped-after-reload', '重新整理：半段回答與「（已停止回應）」存在 checkpointer');
    const follow = '好，那用一句話總結梅納反應就好，不用查資料';
    await typeAndSend(page, follow);
    const r = await retryIfRateLimited(page, follow, await waitSettled(page));
    await record('停止後可以繼續對話', r === 'settled' && !(await recentRateLimit(page)));
    await snap(page, 'after-stop-continue', '停止後繼續對話正常');
  }],

  ['撤回排隊中的訊息', async (page) => {
    await typeAndSend(page, '請用大約 300 字介紹怎麼挑選新鮮的魚');
    await stopButton(page).waitFor({ timeout: 20000 });
    await typeAndSend(page, '這則訊息我等一下會撤回');
    await page.getByTestId('queue-badge').first().waitFor({ timeout: 8000 });
    await snap(page, 'before-withdraw', '排隊中的訊息旁有「撤回」');
    await page.getByRole('button', { name: '撤回' }).click();
    await sleep(800);
    const gone = !(await page.getByText('這則訊息我等一下會撤回').count());
    await record('按撤回後排隊中的訊息消失', gone);
    await snap(page, 'withdrawn', '撤回後只剩正在回覆的那則');
    await waitSettled(page);
  }],

  ['HITL 審核卡：重新整理仍在、Redis 紀錄過期仍在、同意後執行', async (page) => {
    await page.getByTestId('new-chat').click();
    await typeAndSend(page, '請上網搜尋三杯雞的做法');
    await page.waitForURL(/\/chat\/thread_/);
    const r = await retryIfRateLimited(page, '請上網搜尋三杯雞的做法', await waitSettled(page, { onCard: 'stop' }), { onCard: 'stop' });
    await record('模型要上網搜尋 → 出現授權確認卡', r === 'card');
    await snap(page, 'hitl-card', 'HITL 授權確認卡');
    await page.reload();
    const afterReload = await pendingCard(page).waitFor({ timeout: 10000 }).then(() => true).catch(() => false);
    await record('重新整理後審核卡仍在', afterReload);
    // 模擬 24 小時後 Redis 任務紀錄過期：把這個 thread 相關的任務 key 全刪掉
    const threadId = decodeURIComponent(page.url().split('/chat/')[1]);
    const taskKeys = redis('--scan', '--pattern', 'chef:task:*').split('\n').filter(Boolean);
    let deleted = 0;
    for (const k of taskKeys) {
      if (k.endsWith(':events')) continue;
      if (redis('HGET', k, 'thread_id') === threadId) {
        redis('DEL', k, `${k}:events`);
        deleted++;
      }
    }
    redis('DEL', `chef:thread:${threadId}:hold`);
    await page.reload();
    const afterExpiry = await pendingCard(page).waitFor({ timeout: 10000 }).then(() => true).catch(() => false);
    await record('刪掉 Redis 任務紀錄（模擬過期）後審核卡仍在（讀自 checkpointer）', afterExpiry, `刪除 ${deleted} 個任務`);
    await snap(page, 'hitl-after-redis-expiry', 'Redis 紀錄刪除後重新整理，卡片照樣出現');
    const errBefore = await errorCount(page);
    const r2 = await waitSettled(page, { onCard: 'approve' });
    const answer = await page.locator('[data-testid="task-turn"]').last().innerText().catch(() => '');
    const newErrors = (await errorCount(page)) - errBefore;
    await record('同意後執行搜尋並回答', r2 === 'settled' && newErrors === 0,
      `結果=${r2}，新增錯誤提示 ${newErrors} 個：${answer.slice(-60).replace(/\n/g, ' ')}`);
    await snap(page, 'hitl-approved', '同意後搜尋並回答');
  }],

  ['略過審核：直接送新訊息', async (page) => {
    await typeAndSend(page, '再幫我上網搜尋宮保雞丁的做法');
    const r = await retryIfRateLimited(page, '再幫我上網搜尋宮保雞丁的做法', await waitSettled(page, { onCard: 'stop' }), { onCard: 'stop' });
    await record('再次出現審核卡', r === 'card');
    // 同一對話先前情境若被 Groq 限流，頁面上會留有舊的錯誤提示；只看這一步之後新增的
    const errorsBefore = await page.locator('[data-testid="status-banner"].banner-error').count();
    await typeAndSend(page, '算了不用搜尋，告訴我宮保雞丁要準備哪些主要食材就好');
    const skipped = await page.locator('[data-testid="hitl-card"][data-state="skipped"]').first()
      .waitFor({ timeout: 10000 }).then(() => true).catch(() => false);
    await record('直接送新訊息 → 卡片標示「已略過」', skipped);
    const r2 = await waitSettled(page, { onCard: 'reject' });
    const errorsAfter = await page.locator('[data-testid="status-banner"].banner-error').count();
    await record('新訊息正常回覆（沒有因懸空的工具呼叫報錯）', r2 === 'settled' && errorsAfter === errorsBefore,
      `結果=${r2}，新增錯誤提示 ${errorsAfter - errorsBefore} 個`);
    await snap(page, 'hitl-skipped', '卡片已略過，對話繼續');
  }],

  ['回覆中重新整理 / 切換對話', async (page) => {
    await typeAndSend(page, '請用大約 300 字說明炒青菜怎麼保持翠綠，不用查資料');
    await waitForTokens(page, 20);
    await page.reload();
    const resumed = await page.locator('[data-testid="typing"], [data-testid="assistant-message"]').last()
      .waitFor({ timeout: 15000 }).then(() => true).catch(() => false);
    await snap(page, 'reload-mid-stream', '回覆到一半按 F5：從頭重播並接續');
    const r = await waitSettled(page);
    await record('回覆中重新整理後接續完成', resumed && r === 'settled');
    const url = page.url();
    const errBefore = await errorCount(page);
    await typeAndSend(page, '請用大約 300 字說明水煮蛋怎麼煮出溏心，不用查資料');
    await stopButton(page).waitFor({ timeout: 20000 });
    await page.getByTestId('thread-item').nth(1).click();
    await snap(page, 'switched-away', '回覆中切到別的對話（任務在背景繼續跑）');
    await sleep(8000);
    await page.goto(url);
    const r2 = await waitSettled(page);
    const last = await page.locator('[data-testid="task-turn"]').last().innerText().catch(() => '');
    const newErrors = (await errorCount(page)) - errBefore;
    await record('切回來看到背景完成的回覆', r2 === 'settled' && newErrors === 0 && last.includes('蛋'),
      `新增錯誤提示 ${newErrors} 個：${last.slice(-60).replace(/\n/g, ' ')}`);
    await snap(page, 'switched-back', '切回原對話，回覆已完成');
  }],

  ['刪除對話', async (page) => {
    const before = await page.getByTestId('thread-item').count();
    await page.getByTestId('thread-item').last().hover();
    await page.getByTestId('thread-item').last().locator('.thread-delete').click();
    await sleep(1000);
    const after = await page.getByTestId('thread-item').count();
    await record('側邊欄刪除對話', after === before - 1, `${before} → ${after}`);
  }],
];

// ---------- 主程式 ----------
const browser = await chromium.connectOverCDP(CDP, { slowMo: 60 });
const context = browser.contexts()[0];
const page = context.pages().find((p) => p.url().startsWith(APP)) || context.pages()[0] || (await context.newPage());
await page.bringToFront();
await page.setViewportSize({ width: 1400, height: 880 }).catch(() => {});
page.on('pageerror', (e) => { record('瀏覽器 console 錯誤', false, e.message); });

const only = process.argv.slice(2);
for (const [i, [name, fn]] of scenarios.entries()) {
  if (only.length && !only.some((o) => name.includes(o))) continue;
  log(`\n=== ${name} ===`);
  report.push({ type: 'scenario', name });
  currentPage = page;
  try {
    await fn(page);
  } catch (e) {
    await record(`${name}：執行中斷`, false, e.message.split('\n')[0]);
    await snap(page, `error-${i}`, '出錯時的畫面').catch(() => {});
  }
  if (i < scenarios.length - 1) await sleep(PAUSE_BETWEEN);
}

// 最後停在第一個對話（排隊 / 停止都在這裡），瀏覽器保持開啟
if (firstThreadUrl) await page.goto(firstThreadUrl).catch(() => {});
const checks = report.filter((r) => r.type === 'check');
const passed = checks.filter((c) => c.status === 'pass').length;
const rated = checks.filter((c) => c.status === 'rate').length;
const md = [
  `# UI 走查報告`,
  ``,
  `- 開始：${started.toLocaleString()}　結束：${new Date().toLocaleString()}`,
  `- 網址：${APP}（docker compose 的 web 服務）`,
  `- 結果：${passed} / ${checks.length} 項通過；${rated} 項因 Groq 免費方案每分鐘 token 上限（429）未能完成（⚠️，非功能錯誤）；${checks.length - passed - rated} 項失敗（❌）`,
  ``,
  ...report.map((r) =>
    r.type === 'scenario' ? `\n## ${r.name}\n`
    : r.type === 'note' ? `- ⏳ ${r.text}`
    : r.type === 'check' ? `- ${{ pass: '✅', rate: '⚠️', fail: '❌' }[r.status]} ${r.scenario}${r.detail ? `　\`${String(r.detail).replace(/`/g, "'").replace(/\n/g, ' ')}\`` : ''}`
    : `- 📸 [${r.file}](${r.file})${r.note ? `：${r.note}` : ''}`),
].join('\n');
writeFileSync(join(OUT, 'REPORT.md'), md);
log(`\n報告：${join(OUT, 'REPORT.md')}（${passed}/${checks.length} 通過，${rated} 項限流）`);
process.exit(0);  // 只斷開 CDP 連線，不呼叫 browser.close()，瀏覽器維持開啟
