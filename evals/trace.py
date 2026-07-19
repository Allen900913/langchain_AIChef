"""跑真實 agent 並抽出評測所需的軌跡。

對每個 case：
1. 視需要把 diet profile / inventory 種子寫進 store（隔離在該 case 的 user_id 下）
2. agent.ainvoke 跑到結束（eval 用 enable_hitl=False 建的 agent，不會觸發中斷）
3. 從最終 state 的 messages 抽出：
   - answer            : 最後一則有文字的 AIMessage
   - tool_calls        : 整輪所有 AIMessage.tool_calls 的工具名（依序、含重複）
   - retrieved_contexts: 所有 web_search ToolMessage 的內容（給 Faithfulness / ContextRelevance）
   - tokens / latency   : 供成本與延遲指標使用
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.agents.app as app_module
from app.agents.app import MAX_STEPS, init_agent_infra

_initialized = False


async def ensure_infra() -> None:
    global _initialized
    if not _initialized:
        # eval 跑無人值守批次：關 HITL（不需人工審核）、關 reflection（其 worker 是非 daemon
        # 執行緒會卡住行程 exit，且 eval 走 ainvoke 本就不觸發反思）。正式環境預設皆 True。
        await init_agent_infra(enable_hitl=False, enable_reflection=False)
        _initialized = True


async def _seed(case: dict[str, Any]) -> None:
    """把 case 指定的 diet profile / inventory 寫入 store（對齊 tools.py 的讀取格式）。"""
    store = app_module.store
    user_id = case["user_id"]

    if case.get("seed_diet") is not None:
        # profiles_get 讀的是 ("memories", user_id, domain) 下每筆 value["content"]
        await store.adelete(("memories", user_id, "diet"), "seed")
        await store.aput(("memories", user_id, "diet"), "seed",
                         {"content": case["seed_diet"]})

    if case.get("seed_inventory") is not None:
        # inventory_get 讀的是 ("inventory", user_id) 的 "items" → value["items"]
        await store.aput(("inventory", user_id), "items", {
            "items": sorted(set(case["seed_inventory"])),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })


def _text_of(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""


def _strip_tags(text: str) -> str:
    """去掉 web_search 回傳外層的 <tool_output> 標籤，只留實質內容。"""
    return text.replace("<tool_output>", "").replace("</tool_output>", "").strip()


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    await ensure_infra()
    await _seed(case)

    agent = app_module.agent
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {"thread_id": thread_id, "user_id": case["user_id"]},
        "recursion_limit": MAX_STEPS * 2 + 1,
    }
    msg = HumanMessage(content=f"<user_input>\n{case['query']}\n</user_input>")

    start = time.perf_counter()
    final_state = await agent.ainvoke({"messages": [msg]}, config=config)
    latency_seconds = time.perf_counter() - start

    messages = final_state["messages"]

    tool_calls: list[str] = []
    retrieved_contexts: list[str] = []
    answer = ""
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                tool_calls.append(tc["name"])
            text = _text_of(m)
            if text.strip():
                answer = text.strip()  # 保留最後一則有文字的 AI 訊息
            usage = m.usage_metadata or {}
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
            model_calls += 1
        elif isinstance(m, ToolMessage) and m.name == "web_search":
            ctx = _strip_tags(_text_of(m))
            if ctx:
                retrieved_contexts.append(ctx)

    return {
        "id": case["id"],
        "query": case["query"],
        "answer": answer,
        "tool_calls": tool_calls,
        "retrieved_contexts": retrieved_contexts,
        "expected_tools": case.get("expected_tools", []),
        "allergens": case.get("allergens", []),
        "latency_seconds": latency_seconds,
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
