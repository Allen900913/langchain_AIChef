"""Auditor Agent：Run 結束後**有條件地**背景執行，分析 Agent 的行為模式問題，
並透過 save_error_rule 工具把高價值行為準則寫進資料庫（AsyncPostgresStore）。

觸發條件（只在以下情況才啟動，非每次 run 都跑）：
1. 認知失敗：_stream_agent 偵測到 COGNITIVE_FAILURE（重複迴圈、步數耗盡等）
2. 外部觸發：未來擴充使用者負評等信號

不觸發的情況：
- 技術故障（API timeout、404 等）：由 ToolException + handle_tool_error 即時處理
- 正常結束的對話：無需浪費 LLM 成本做稽核

設計原則：
- 存入的是「可重複利用的行為準則」，不是系統 log 或技術故障記錄
- 工具化儲存：Auditor Agent 透過 save_error_rule tool 決定存什麼
- 底層使用 LangGraph AsyncPostgresStore，namespace=("error_lessons", user_id)
"""

import asyncio
import hashlib
import json

# ──────────────────────────────────────────────────────────────
# 背景任務工具（從 lessons.py 搬遷）
# ──────────────────────────────────────────────────────────────

# 持有 task 參照，避免 fire-and-forget 的背景任務在完成前被 GC 回收
_background_tasks: set[asyncio.Task] = set()


def fire_and_forget(coro) -> None:
    """將 coroutine 排入背景執行，不阻塞主流程。失敗不影響使用者請求。"""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ──────────────────────────────────────────────────────────────
# 執行期由 init_agent_infra 注入的全域依賴（與 app.py 的模式一致）
# ──────────────────────────────────────────────────────────────
store = None           # AsyncPostgresStore
summary_model = None   # 驅動 Auditor Agent 的小模型（Groq 8B）

# 在工具執行期用 contextvars 傳遞 user_id，避免全域變數競爭
import contextvars
_current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_user_id", default="default_user"
)

# 防呆規則門檻設定（對齊原 lessons.py 的設計，數值可依觀察調整）
_ERROR_RULE_MIN_HITS = 2    # 同一情境重複發生這麼多次才注入（避免偶發事件）
_ERROR_RULE_MAX_INJECT = 5  # 每輪最多注入幾條規則（成本封頂）
_ERROR_RULE_TTL_DAYS = 30   # 超過這麼久沒再觸發就不再注入
_DEBOUNCE_SECONDS = 120     # 同一 run 內同一條規則最多累計一次（對齊 lessons.py）


# ──────────────────────────────────────────────────────────────
# Auditor Agent 的工具
# ──────────────────────────────────────────────────────────────

@tool
async def save_error_rule(trigger_condition: str, corrective_action: str) -> str:
    """將高價值的行為準則存入資料庫，供未來 Agent 參考避免重蹈覆轍。

    只記錄「可重複利用的行為模式修正」，不記錄一次性技術故障（如 API timeout）。

    Args:
        trigger_condition: 觸發情境的一句話描述，聚焦行為模式而非技術細節，例如
            「Agent 對同一個搜尋 query 反覆呼叫 web_search 且不改變關鍵字」
            「使用者要求特定料理但 Agent 偏離主題去推薦其他菜式」
        corrective_action: 下次應採取的正確做法，用「請…」開頭，例如
            「請在第一次搜尋結果不理想時，改變關鍵字策略或改用其他工具」
            「請始終以使用者最新的明確請求為優先，避免自行發散」
    """
    if store is None:
        return "❌ Store 尚未初始化，無法儲存規則"

    user_id = _current_user_id.get()
    ns = ("error_lessons", user_id)
    # 用 trigger_condition 的 MD5 作為 key，避免完全相同情境重複存入
    key = hashlib.md5(trigger_condition.encode("utf-8")).hexdigest()[:12]
    now = datetime.now(timezone.utc)

    try:
        existing = await store.aget(ns, key)
        current_hits = existing.value.get("hits", 0) if existing else 0

        # Debounce：同一個 run（時間窗口內）同一條規則只累計一次
        if existing:
            try:
                last = datetime.fromisoformat(existing.value["last_hit"])
                if (now - last).total_seconds() < _DEBOUNCE_SECONDS:
                    return f"⏭️ Debounce 跳過（距上次記錄 < {_DEBOUNCE_SECONDS}s）"
            except (KeyError, ValueError):
                pass

        new_hits = current_hits + 1
        await store.aput(ns, key, {
            "trigger_condition": trigger_condition,
            "corrective_action": corrective_action,
            "hits": new_hits,
            "last_hit": now.isoformat(),
        })

        threshold_note = "（已達門檻，下輪將注入 System Prompt）" if new_hits == _ERROR_RULE_MIN_HITS else ""
        msg = f"✅ 規則已儲存至資料庫 | Key={key} | Hits={new_hits} {threshold_note}"
        print(f"📓 [AUDITOR] {trigger_condition[:60]}... → hits={new_hits}")
        return msg

    except Exception as exc:
        print(f"⚠️ [AUDITOR] save_error_rule 失敗：{exc}")
        return f"❌ 儲存失敗：{exc}"


# ──────────────────────────────────────────────────────────────
# Auditor Agent 提示詞
# ──────────────────────────────────────────────────────────────

_AUDITOR_SYSTEM = """你是一位 AI 助理的行為稽核員。你的工作是閱讀另一個 AI 的任務執行軌跡，\
找出它的「行為模式問題」，並透過工具把可重複利用的行為準則寫進資料庫。

━━━━━━━━━━━━━━━━━━━━━━━━
你的稽核範圍（只關注以下兩類問題）
━━━━━━━━━━━━━━━━━━━━━━━━

① 重複行為迴圈
   辨識方法：軌跡中出現 [COGNITIVE_FAILURE:REPEAT:*] 標籤。
   代表意義：Agent 以完全相同的參數反覆呼叫同一個工具，陷入死迴圈。
   你需要提煉：是什麼任務情境導致了這個迴圈？Agent 應該如何提早辨識
   並切換策略（例如換關鍵字、改用其他工具、或直接用現有資訊作答）？

② 隱性行為偏差（僅限有強烈證據時）
   辨識方法：軌跡中無技術報錯，但最終回覆與任務目標「嚴重且明顯」不符。
   注意：以下情況「不算」錯誤，請直接 PASS：
   - 助理用不同方式（如直接說明而非啟動 step_tracker）完成了任務 → 正常
   - 助理在回覆末尾主動提供額外建議或延伸資訊 → 正常
   - 搜尋結果不夠完美但助理已盡力 → 正常
   只有以下情況才算隱性錯誤：
   - 使用者明確提到過敏原/限制，但最終推薦完全忽略
   - 在無任何工具佐證的情況下，編造了具體的數字或步驟
   - 完全答非所問（使用者問 A，助理答 B，且 B 與 A 無關）

━━━━━━━━━━━━━━━━━━━━━━━━
不在你的稽核範圍（請忽略）
━━━━━━━━━━━━━━━━━━━━━━━━

- 技術故障（API timeout、搜尋失敗等）：這些是暫態問題，Agent 已透過錯誤訊息
  即時處理，不需要你記錄為行為準則。
- 步數耗盡（STEP_LIMIT）或整體逾時（TIMEOUT）：這些是系統保護機制觸發，
  不代表 Agent 有行為問題。

━━━━━━━━━━━━━━━━━━━━━━━━
你的行動指南
━━━━━━━━━━━━━━━━━━━━━━━━

1. 仔細閱讀下方的觸發原因、任務目標與執行軌跡。
2. 若助理行為合理（即使結果不完美），直接回答 "PASS"，不要呼叫工具。
   寧可漏判也不要誤判——錯誤的規則比沒有規則更有害。
3. 若發現明確的行為模式問題：呼叫 save_error_rule 工具，一次記錄一條準則。
   - trigger_condition：描述「行為模式」而非技術細節。
   - corrective_action：具體說明「應該怎麼做」，用「請…」開頭。
   - 同一份軌跡最多呼叫 2 次。
4. 只記錄你有充分軌跡證據的問題，不要臆測。有疑問時一律 PASS。"""

# executor 在 init_auditor 時建立（需要 summary_model 已初始化）
_auditor_executor = None


def init_auditor(sm, st) -> None:
    """由 app.py 的 init_agent_infra() 呼叫，注入模型與 store。"""
    global store, summary_model, _auditor_executor
    store = st
    summary_model = sm

    _auditor_executor = create_react_agent(
        summary_model,
        tools=[save_error_rule],
        prompt=_AUDITOR_SYSTEM,
    )


# ──────────────────────────────────────────────────────────────
# 軌跡工具函式
# ──────────────────────────────────────────────────────────────

def _is_summary_msg(msg) -> bool:
    """判斷是否為摘要壓縮注入的假 HumanMessage（不算本輪起點）。"""
    return (
        isinstance(msg, HumanMessage)
        and msg.additional_kwargs.get("lc_source") == "summarization"
    )


def _this_run_messages(messages: list) -> list:
    """從訊息歷史中切出「本輪 run」的部分（最後一則真實 HumanMessage 到結尾）。"""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and not _is_summary_msg(messages[i]):
            start = i
            break
    return messages[start:]


def format_trajectory(messages: list) -> str:
    """將本輪訊息格式化為易讀的軌跡文字，供 Auditor Agent 分析。"""
    lines = []
    for m in _this_run_messages(messages):
        if isinstance(m, HumanMessage):
            content = str(m.content)[:600]
            lines.append(f"[使用者] {content}")
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)[:400]
                    lines.append(f"[AI→工具呼叫] {tc['name']}({args_str})")
            if m.content:
                text = m.content if isinstance(m.content, str) else str(m.content)
                # 只顯示前 600 字，避免稽核上下文過長
                lines.append(f"[AI回覆] {text[:600]}")
        elif isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            tool_name = getattr(m, "name", "?")
            lines.append(f"[工具回傳 {tool_name}] {content[:400]}")
    return "\n".join(lines)


def _should_audit(messages: list, goal: str, trigger_reason: str | None = None) -> bool:
    """判斷本輪是否值得啟動稽核。

    觸發條件（任一滿足即觸發）：
    1. 外部明確觸發（trigger_reason 非 None）：如 COGNITIVE_FAILURE 或未來的使用者負評
    2. 未來擴充：可加入抽樣率等機制

    不觸發的情況：
    - 正常結束的對話（無 trigger_reason）
    - 技術故障（API timeout 等，由 ToolException 即時處理）
    """
    # 有明確觸發原因就跑
    if trigger_reason:
        return bool(goal)
    # 沒有觸發原因 = 正常結束，不跑 Auditor
    return False


# ──────────────────────────────────────────────────────────────
# 主入口：run_auditor
# ──────────────────────────────────────────────────────────────

async def run_auditor(
    messages: list,
    goal: str,
    user_id: str,
    trigger_reason: str | None = None,
) -> None:
    """背景執行的 Auditor Agent 主入口。

    由 app.py 的 _stream_agent finally 區塊透過 fire_and_forget 呼叫。
    只在有明確觸發原因時才啟動稽核，正常結束的對話不觸發。
    失敗時只打 print，不影響主流程。

    Args:
        messages:       從 checkpointer 讀取的完整訊息歷史
        goal:           本輪任務的 original_goal（來自 ChefState）
        user_id:        用於 Store namespace 定位
        trigger_reason: 觸發原因（如 'COGNITIVE_FAILURE:REPEAT:web_search'）。
                        None 表示正常結束，不觸發稽核。
    """
    if _auditor_executor is None:
        print("⚠️ [AUDITOR] Executor 尚未初始化，跳過稽核")
        return

    if not _should_audit(messages, goal, trigger_reason):
        return  # 正常結束，不需要稽核

    trajectory = format_trajectory(messages)
    if not trajectory.strip():
        return

    # 透過 contextvars 傳遞 user_id 給 save_error_rule tool
    token = _current_user_id.set(user_id)
    try:
        goal_str = goal or "（使用者未設定明確任務目標）"
        reason_str = trigger_reason or "未知"
        print(f"📓 [AUDITOR] 啟動稽核（原因：{reason_str}）")
        await _auditor_executor.ainvoke({
            "messages": [
                HumanMessage(content=(
                    f"【觸發原因】\n{reason_str}\n\n"
                    f"【任務目標】\n{goal_str}\n\n"
                    f"【執行軌跡】\n{trajectory}\n\n"
                    "請開始稽核。若發現行為模式問題，呼叫 save_error_rule 工具記錄準則。"
                    "若行為合理，直接回答 PASS。"
                ))
            ]
        })
    except Exception as exc:
        print(f"⚠️ [AUDITOR] 稽核執行失敗：{type(exc).__name__}: {exc}")
    finally:
        _current_user_id.reset(token)


# ──────────────────────────────────────────────────────────────
# 讀取已生效的防呆規則（供 ContextShaping 注入 System Prompt）
# ──────────────────────────────────────────────────────────────

async def active_error_rules(user_id: str) -> list[str]:
    """讀取已達門檻（hits >= N）且未過期的防呆規則，供每輪注入 system prompt。

    Returns:
        list of "當…時，請…" 格式的規則字串，最多 _ERROR_RULE_MAX_INJECT 條。
    """
    if store is None:
        return []

    ns = ("error_lessons", user_id)
    try:
        items = await store.asearch(ns)
    except Exception as exc:
        print(f"⚠️ [AUDITOR] 讀取規則失敗：{exc}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=_ERROR_RULE_TTL_DAYS)
    valid = []
    for it in items:
        v = it.value
        if v.get("hits", 0) < _ERROR_RULE_MIN_HITS:
            continue
        try:
            if datetime.fromisoformat(v["last_hit"]) < cutoff:
                continue
        except (KeyError, ValueError):
            continue
        valid.append(v)

    # 按 hits 由高到低排序，取前 N 條最常發生的
    valid.sort(key=lambda v: v.get("hits", 0), reverse=True)
    return [
        f"當「{v['trigger_condition']}」時，{v['corrective_action']}"
        for v in valid[:_ERROR_RULE_MAX_INJECT]
    ]
