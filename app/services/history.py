"""API 端的對話歷史讀取：直接讀 LangGraph checkpointer（Postgres），不經過 agent。

API 伺服器只需要「讀歷史 / 刪 thread」，不該為此載入模型與向量庫，所以只建 checkpointer
物件，只依賴 langgraph 的 postgres checkpointer 與 langchain_core 的訊息型別。
資料表的 setup()（migration）由 worker 的 init_agent_infra 負責，API 不做，
避免兩邊同時啟動時併發跑 migration 互相衝突。
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.errors import UndefinedTable

from app.common.messages import clean_user_text, is_summary_msg, summary_text
from app.services import db

# LangGraph 存放 HITL 中斷的 pending write channel 名稱（langgraph.constants.INTERRUPT 已改為私有）
_INTERRUPT_CHANNEL = "__interrupt__"

_checkpointer: AsyncPostgresSaver | None = None


async def init_history() -> None:
    """建立 checkpointer 讀寫物件（共用 db 模組的連線池）。"""
    global _checkpointer
    await db.init_pool()
    _checkpointer = AsyncPostgresSaver(db.get_pool())


async def close_history() -> None:
    global _checkpointer
    _checkpointer = None


async def get_thread_state(thread_id: str) -> tuple[list, dict | None]:
    """讀 thread 最新的 checkpoint，回傳 (messages, pending_interrupt)。

    pending_interrupt：HITL 中斷時，LangGraph 把 Interrupt 物件存成最新 checkpoint 的
    pending write（channel = "__interrupt__"）。只要使用者還沒審核（也沒有改送新訊息讓
    graph 往前走），它就一直在 checkpointer 裡——沒有期限，不依賴 Redis 的任務紀錄。
    """
    try:
        tup = await _checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    except UndefinedTable:
        # worker 還沒啟動過、checkpointer 資料表尚未建立 → 視同空 thread
        return [], None
    if not tup:
        return [], None
    messages = (tup.checkpoint.get("channel_values") or {}).get("messages") or []

    pending_interrupt = None
    for _task_id, channel, value in tup.pending_writes or []:
        if channel != _INTERRUPT_CHANNEL:
            continue
        for intr in value if isinstance(value, (list, tuple)) else [value]:
            payload = getattr(intr, "value", None)
            if isinstance(payload, dict) and payload.get("action_requests"):
                pending_interrupt = {
                    "action_requests": payload["action_requests"],
                    "review_configs": payload.get("review_configs", []),
                }
    return messages, pending_interrupt


async def get_raw_messages(thread_id: str) -> list:
    """回傳 thread 目前 state 中的完整 messages（含工具呼叫/摘要），不存在時回傳空 list。"""
    messages, _ = await get_thread_state(thread_id)
    return messages


def last_message_id(messages: list) -> str | None:
    return messages[-1].id if messages else None


def slice_until(messages: list, boundary_id: str | None) -> list:
    """取到 boundary_id（含）為止的訊息，作為「任務開始前」的歷史。

    boundary_id 為 None（任務送出時 thread 還是空的）或找不到（任務執行中觸發摘要壓縮，
    界線之前的訊息已被摘要取代）時回傳空 list——兩種情況下界線之前都沒有可渲染的原始訊息。
    """
    if boundary_id is None:
        return []
    for i, m in enumerate(messages):
        if m.id == boundary_id:
            return messages[: i + 1]
    return []


def render_messages(messages: list) -> list[dict]:
    """轉成前端要渲染的對話氣泡：使用者原話、有文字的助理回覆、摘要。

    摘要訊息（role="summary"）：較早的對話被 SummarizationMiddleware 壓縮掉了，畫面上用摘要卡片
    告訴使用者前面還有內容、大概聊了什麼，不然對話會像是從半途（常常是一則 AI 回答）開始。
    略過：純 tool_calls 的 AIMessage、ToolMessage。
    使用者訊息去掉 <user_input> 包裝與 vision 附加的 <tool_output> 區塊。
    """
    rendered = []
    for m in messages:
        if isinstance(m, HumanMessage):
            if is_summary_msg(m):
                text = summary_text(m)
                if text:
                    rendered.append({"id": m.id, "role": "summary", "content": text})
                continue
            if not isinstance(m.content, str):
                continue
            text = clean_user_text(m.content)
            if text:
                rendered.append({"id": m.id, "role": "user", "content": text})
        elif isinstance(m, AIMessage):
            if isinstance(m.content, str) and m.content.strip():
                rendered.append({"id": m.id, "role": "assistant", "content": m.content})
    return rendered


async def delete_thread(thread_id: str) -> None:
    try:
        await _checkpointer.adelete_thread(thread_id)
    except UndefinedTable:
        pass

