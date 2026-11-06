"""離線測試：摘要模型出錯時不能把前文換成錯誤訊息（app/agents/app.py 的 SafeSummarization）。

不呼叫任何模型 API：主模型與摘要模型都是假的。

測試項目：
1. 對照組：LangChain 原本的 SummarizationMiddleware 遇到摘要模型出錯，前文被「Error generating summary」取代
2. SafeSummarization：摘要模型出錯時保留全部訊息，主模型照常回答
3. SafeSummarization：摘要模型正常時照常壓縮
4. 中文 token 估算：一字約一個 token（預設估法只算 1/4）
5. 摘要一直失敗、上下文超過門檻：送給模型的請求只留最近的訊息，state 不動

執行方式：python app/tests/test_safe_summarization.py
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.app import SafeSummarization, cjk_token_count

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  {detail}")


class _Chef(BaseChatModel):
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-chef"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _Chef.seen = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("好的" * 50))])


class _Summarizer(BaseChatModel):
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake-summarizer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail:
            raise RuntimeError("Error code: 429 - rate_limit_exceeded（模擬每日額度用完）")
        return ChatResult(generations=[ChatGeneration(message=AIMessage("（摘要）使用者第一句說對蝦過敏"))])


async def run(middleware_cls, summarizer: _Summarizer, **kwargs) -> list:
    agent = create_agent(
        _Chef(), tools=[], checkpointer=InMemorySaver(),
        middleware=[middleware_cls(model=summarizer, trigger=("messages", 6), keep=("messages", 2), **kwargs)],
    )
    cfg = {"configurable": {"thread_id": "t"}}
    await agent.ainvoke({"messages": [HumanMessage("我對蝦過敏")]}, cfg)
    for i in range(4):
        await agent.ainvoke({"messages": [HumanMessage(f"第 {i + 2} 句")]}, cfg)
    return (await agent.aget_state(cfg)).values["messages"]


def text_of(messages) -> str:
    return "\n".join(str(m.content) for m in messages)


async def main():
    print("1. 對照組：原本的 SummarizationMiddleware")
    msgs = await run(SummarizationMiddleware, _Summarizer(fail=True))
    check("摘要模型出錯 → 前文被錯誤訊息取代（重現問題）",
          "Error generating summary" in text_of(msgs) and "我對蝦過敏" not in text_of(msgs))

    print("2. SafeSummarization：摘要模型出錯")
    msgs = await run(SafeSummarization, _Summarizer(fail=True))
    check("沒有錯誤訊息混進對話", "Error generating summary" not in text_of(msgs))
    check("第一句還在", "我對蝦過敏" in text_of(msgs))
    check("每一輪都有回答（5 問 5 答）", len(msgs) == 10, f"len={len(msgs)}")

    print("3. SafeSummarization：摘要模型正常")
    msgs = await run(SafeSummarization, _Summarizer(fail=False))
    check("照常壓縮成摘要", "（摘要）" in text_of(msgs) and len(msgs) < 10, f"len={len(msgs)}")

    print("4. 中文 token 估算")
    n = cjk_token_count([HumanMessage("雞" * 1000)])
    check("1000 個中文字約 1000 token", 950 <= n <= 1100, f"n={n}")
    n_en = cjk_token_count([HumanMessage("a" * 1000)])
    check("英文維持字元數 ÷ 4", 240 <= n_en <= 270, f"n={n_en}")

    print("5. 摘要一直失敗、上下文超過門檻")
    msgs = await run(SafeSummarization, _Summarizer(fail=True), fallback_tokens=250)
    check("state 裡前文都還在", "我對蝦過敏" in text_of(msgs) and len(msgs) == 10, f"len={len(msgs)}")
    sent = [m for m in _Chef.seen if not isinstance(m, SystemMessage)]
    check("送給模型的只有最近幾則、從使用者訊息開始", 0 < len(sent) < 9 and isinstance(sent[0], HumanMessage),
          f"sent={len(sent)}")
    check("最後一則使用者訊息一定在", "第 5 句" in text_of(sent))

    print(f"\n通過 {passed}，失敗 {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
