"""Auditor Agent：Run 結束後背景執行，讀取對話軌跡，找出三大類錯誤，
並呼叫 save_error_rule 工具把防呆規則寫進資料庫（AsyncPostgresStore）。

三大類錯誤：
1. 顯性錯誤 (Explicit)  ：ToolMessage 中含有 [System_Error:*] 標籤
2. 認知失敗 (Cognitive) ：messages 末端含有 [COGNITIVE_FAILURE:*] 標籤（由 _stream_agent 寫入）
3. 隱性錯誤 (Implicit) ：軌跡完整且無報錯，但邏輯目標未達成（由 LLM 稽核）

設計原則：
- 工具化儲存：Auditor Agent 透過 save_error_rule tool 決定什麼時候、存什麼，
  而非由程式寫死判斷邏輯。
- 不自建表：底層使用 LangGraph AsyncPostgresStore.aput，namespace=("error_lessons", user_id)，
  LangGraph 統一管理底層 schema，我們不寫 SQL。
- 完全取代 lessons.py：原本所有的 record_lesson 呼叫統一由 Auditor Agent 接管。
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
    """當你在軌跡中發現 Agent 有系統性錯誤時，呼叫此工具把防呆規則存入資料庫。

    Args:
        trigger_condition: 觸發情境的一句話描述，例如
            「呼叫 web_search 時發生 TimeoutError」
            「Agent 在達到步數上限時仍在重複呼叫相同的工具」
        corrective_action: 下次應採取的正確做法，例如
            「搜尋逾時時請換一組更精確的關鍵字，不要重複相同的搜尋」
            「若第一次呼叫結果不理想，請改變策略或直接用現有資訊作答」
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

_AUDITOR_SYSTEM = """你是一位 AI 助理的資深稽核員。你的工作是閱讀另一個 AI 的任務執行軌跡，\
找出它犯的錯誤，並透過工具把防呆規則寫進資料庫。

━━━━━━━━━━━━━━━━━━━━━━━━
錯誤分類與辨識方式
━━━━━━━━━━━━━━━━━━━━━━━━

① 顯性錯誤（Explicit Error）
   辨識方法：ToolMessage 中含有 [System_Error:*] 標籤。
   代表意義：工具呼叫發生了系統層面的失敗（逾時、API 錯誤、參數不合法等）。
   你需要記錄：什麼情境觸發了這個錯誤？下次 Agent 應該如何避免或修正？

② 認知失敗（Cognitive Failure）
   辨識方法：軌跡末尾的 AIMessage 含有 [COGNITIVE_FAILURE:*] 標籤，例如
   [COGNITIVE_FAILURE:TIMEOUT]、[COGNITIVE_FAILURE:STEP_LIMIT]、[COGNITIVE_FAILURE:REPEAT]。
   代表意義：Agent 陷入死結、步數用盡、或無限迴圈。
   你需要記錄：是什麼任務模式導致了這個失敗？如何提早識別並放棄？

③ 隱性錯誤（Implicit Error）
   辨識方法：程式沒有報錯（無 [System_Error] 也無 [COGNITIVE_FAILURE]），
   但最終回覆與任務目標明顯不符。常見模式：
   - 空值誤判：工具回傳空結果，但助理就此斷定「沒有資料」而非確認
   - 目標漂移：最終回覆沒有真正解決使用者的原始問題
   - 虛構資訊：在無工具佐證的情況下，編造了具體的烹飪時間或步驟
   - 遺漏約束：使用者提到的限制（過敏原、不辣等）在最終推薦中被忽略

━━━━━━━━━━━━━━━━━━━━━━━━
你的行動指南
━━━━━━━━━━━━━━━━━━━━━━━━

1. 仔細閱讀下方的任務目標與執行軌跡。
2. 若助理表現完美，沒有以上任何一類錯誤：直接回答 "PASS"，不要呼叫工具。
3. 若發現錯誤：呼叫 save_error_rule 工具，一次記錄一條規則。
   - trigger_condition 要精確描述「是什麼情境」，讓未來的 Agent 知道這條規則何時適用。
   - corrective_action 要具體說明「應該怎麼做」，用「請…」開頭，避免「不要…」。
   - 同一份軌跡最多呼叫 3 次（避免過度提煉偶發事件）。
4. 只記錄你有充分軌跡證據的錯誤，不要臆測。"""

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


def _should_audit(messages: list, goal: str) -> bool:
    """判斷本輪是否值得啟動稽核（成本控制：閒聊不稽核）。

    條件：
    1. 本輪有工具呼叫（純閒聊對話不存在可稽核的行為）
    2. 有任務目標 original_goal（無目標就無法判斷「漂移」）
    3. 工具呼叫次數 >= 2（單次查詢偶發錯誤不值得提煉為模式）
    """
    run_msgs = _this_run_messages(messages)
    tool_call_rounds = sum(
        1 for m in run_msgs
        if isinstance(m, AIMessage) and m.tool_calls
    )
    return bool(goal) and tool_call_rounds >= 2


# ──────────────────────────────────────────────────────────────
# 主入口：run_auditor
# ──────────────────────────────────────────────────────────────

async def run_auditor(messages: list, goal: str, user_id: str) -> None:
    """背景執行的 Auditor Agent 主入口。

    由 app.py 的 _stream_agent finally 區塊透過 fire_and_forget 呼叫。
    失敗時只打 print，不影響主流程。

    Args:
        messages: 從 checkpointer 讀取的完整訊息歷史
        goal:     本輪任務的 original_goal（來自 ChefState）
        user_id:  用於 Store namespace 定位
    """
    if _auditor_executor is None:
        print("⚠️ [AUDITOR] Executor 尚未初始化，跳過稽核")
        return

    if not _should_audit(messages, goal):
        return  # 靜默跳過，不打 log（閒聊太頻繁會洗版）

    trajectory = format_trajectory(messages)
    if not trajectory.strip():
        return

    # 透過 contextvars 傳遞 user_id 給 save_error_rule tool
    token = _current_user_id.set(user_id)
    try:
        goal_str = goal or "（使用者未設定明確任務目標）"
        await _auditor_executor.ainvoke({
            "messages": [
                HumanMessage(content=(
                    f"【任務目標】\n{goal_str}\n\n"
                    f"【執行軌跡】\n{trajectory}\n\n"
                    "請開始稽核。若有錯誤，呼叫 save_error_rule 工具記錄。若沒有，直接回答 PASS。"
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
