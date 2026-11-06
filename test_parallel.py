import asyncio
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langfuse.langchain import CallbackHandler
import sys
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ==============================================================================
# 1. 定義兩個「很慢」的工具
# ==============================================================================
@tool
async def fetch_weather(city: str) -> str:
    """查詢天氣"""
    print(f"🌧️ [工具 1] 開始查詢 {city} 天氣...")
    await asyncio.sleep(3)  # 模擬耗時操作
    print(f"🌧️ [工具 1] 查詢 {city} 完成！")
    return f"{city} 天氣晴朗"

@tool
async def fetch_news(city: str) -> str:
    """查詢新聞"""
    print(f"📰 [工具 2] 開始查詢 {city} 新聞...")
    await asyncio.sleep(3)  # 模擬耗時操作
    print(f"📰 [工具 2] 查詢 {city} 完成！")
    return f"{city} 無重大新聞"


# ==============================================================================
# 2. 定義 LangGraph 狀態與假模型
# ==============================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def mock_model_node(state: State, config):
    """這是一個假模型，用來強制觸發平行處理"""
    messages = state["messages"]
    
    # 如果已經有 ToolMessage 回傳，代表工具跑完了，輸出最終答案
    if messages and messages[-1].type == "tool":
        return {"messages": [AIMessage(content="兩個工具都平行跑完啦！總耗時應該只有約 3 秒，而不是 6 秒。")]}
    
    # 否則，強制一次吐出「兩個」 tool_call 給下一個節點
    print("🤖 [假模型] 決定同時呼叫 2 個工具...")
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "fetch_weather", "args": {"city": "台北"}, "id": "call_w"},
            {"name": "fetch_news", "args": {"city": "台北"}, "id": "call_n"},
        ]
    )
    return {"messages": [msg]}

def should_continue(state: State):
    """判斷要進入工具節點還是結束"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ==============================================================================
# 3. 組合 Graph 並執行
# ==============================================================================
async def main():
    # 使用 LangGraph 內建的 ToolNode，它遇到多個 tool_call 時預設就是非同步平行執行
    tools = [fetch_weather, fetch_news]
    tool_node = ToolNode(tools)

    # 建立工作流
    workflow = StateGraph(State)
    workflow.add_node("model", mock_model_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "model")
    workflow.add_conditional_edges("model", should_continue)
    workflow.add_edge("tools", "model")

    app = workflow.compile()

    # 設定 Langfuse 監聽器
    lf_handler = CallbackHandler()
    
    print("🚀 開始執行 (請觀察終端機，兩個工具會同時印出開始)")
    print("-" * 50)
    
    # 執行
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="開始測試平行處理")]},
        config={
            "callbacks": [lf_handler], 
            "run_name": "Parallel Tools Demo"  # 讓你在 Langfuse 好找
        }
    )
    
    print("-" * 50)
    print("✅ 最終結果:", result["messages"][-1].content)
    
    # 等待 Langfuse 背景上傳完成（Python SDK 會在腳本結束時自動 flush）
    # lf_handler.flush()
    print("📊 資料已傳送，請至 Langfuse 儀表板查看名為 'Parallel Tools Demo' 的 Trace！")

if __name__ == "__main__":
    asyncio.run(main())
