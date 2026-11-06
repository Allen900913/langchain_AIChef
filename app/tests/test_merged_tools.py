"""離線測試：合併後的 inventory / step_tracker 工具，以及歷史紀錄顯示摘要。

不呼叫任何模型 API：工具直接呼叫底層函式，store 用記憶體版。

測試項目：
1. inventory：get / add（去重排序）/ remove，沒給 items 時回錯誤
2. step_tracker：start / next / current / 做完清空，沒有進行中食譜時提示先 start
3. HITL：只有 inventory 的 remove 一次刪 3 樣以上才攔，get / add 不攔
4. 工具清單、system prompt 不再出現舊工具名稱
5. 歷史紀錄：摘要訊息以 role="summary" 顯示、去掉 LangChain 加的英文開頭
6. 走真的 agent（ToolNode + ChefState）：新對話就能 start / next / current，參數注入與 schema 驗證都會經過
   （2026-11-01 瀏覽器實測：只測底層函式時全過，實際上 ChefState 必填欄位讓第一次 start 就驗證失敗）

執行方式：python app/tests/test_merged_tools.py
"""

import asyncio
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agents import tools
from app.agents.app import _hitl_interrupt_on
from app.agents.prompt_manager import DEFAULT_SYSTEM_PROMPT
from app.services.history import render_messages

passed = 0
failed = 0

OLD_NAMES = ("inventory_get", "inventory_add", "inventory_remove",
             "step_tracker_start", "step_tracker_next", "step_tracker_current")


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  {detail}")


def inventory(action: str, items=None, store=None):
    cfg = {"configurable": {"user_id": "u1"}}
    return tools.inventory.func(action=action, config=cfg, store=store, items=items)


def step(action: str, state: dict, **kwargs):
    cmd = tools.step_tracker.func(action=action, tool_call_id="tc", state=state, **kwargs)
    update = dict(cmd.update)
    text = update.pop("messages")[0].content
    return text, update


class _Scripted(BaseChatModel):
    """假模型：收到使用者訊息就依 calls 呼叫下一個工具，收到工具結果就結束這一輪。"""
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            msg = AIMessage("好")
        else:
            name, args = self.calls.pop(0)
            msg = AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{len(messages)}", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def agent_flow():
    print("6. 走真的 agent")
    model = _Scripted(calls=[
        ("step_tracker", {"action": "next"}),
        ("step_tracker", {"action": "start", "recipe_name": "番茄炒蛋", "steps": ["切番茄", "炒蛋"]}),
        ("step_tracker", {"action": "next"}),
        ("step_tracker", {"action": "current"}),
        ("inventory", {"action": "add", "items": ["雞蛋"]}),
        ("inventory", {"action": "get"}),
    ])
    agent = create_agent(model, tools=[tools.step_tracker, tools.inventory], state_schema=tools.ChefState,
                         checkpointer=InMemorySaver(), store=InMemoryStore())
    cfg = {"configurable": {"thread_id": "t", "user_id": "u1"}}

    async def turn() -> tuple[ToolMessage, dict]:
        result = await agent.ainvoke({"messages": [HumanMessage("繼續")]}, cfg)
        return [m for m in result["messages"] if isinstance(m, ToolMessage)][-1], result

    tm, _ = await turn()
    check("新對話 next → 提示先 start（不是驗證錯誤）", tm.status == "success" and "start" in tm.content, tm.content[:200])
    tm, r = await turn()
    check("新對話 start 成功並寫入 state",
          tm.status == "success" and r.get("cooking_steps") == ["切番茄", "炒蛋"], tm.content[:200])
    tm, r = await turn()
    check("next 前進到第 2 步",
          tm.status == "success" and "第 2/2 步：炒蛋" in tm.content and r.get("current_step") == 1, tm.content[:200])
    tm, r = await turn()
    check("current 回報第 2 步、不動 state", "第 2/2 步：炒蛋" in tm.content and r.get("current_step") == 1, tm.content[:200])
    tm, _ = await turn()
    check("inventory add（store / config 注入）", tm.status == "success" and "雞蛋" in tm.content, tm.content[:200])
    tm, _ = await turn()
    check("inventory get 讀到剛加的", tm.status == "success" and "雞蛋" in tm.content, tm.content[:200])


def main():
    print("1. inventory")
    store = InMemoryStore()
    check("空冰箱 get → 空清單", inventory("get", store=store)["items"] == [])
    r = inventory("add", ["雞蛋", "番茄", " 雞蛋 "], store=store)
    check("add 去重、排序", r["items"] == sorted({"雞蛋", "番茄"}) and r["added"], f"{r}")
    inventory("add", ["高麗菜"], store=store)
    r = inventory("remove", ["番茄"], store=store)
    check("remove 拿掉指定食材", r["items"] == sorted({"雞蛋", "高麗菜"}) and r["removed"] == ["番茄"], f"{r}")
    check("get 讀到最新清單", inventory("get", store=store)["items"] == sorted({"雞蛋", "高麗菜"}))
    check("add 沒給 items → 回錯誤、不改資料",
          "error" in inventory("add", [], store=store)
          and inventory("get", store=store)["items"] == sorted({"雞蛋", "高麗菜"}))

    print("2. step_tracker")
    state: dict = {}
    text, _ = step("next", state)
    check("沒有進行中食譜 → 提示先 start", "start" in text, text)
    text, upd = step("start", state, recipe_name="番茄炒蛋", steps=["切番茄", "炒蛋", "合炒"])
    check("start 回報第 1 步並寫入 state", "第 1 步：切番茄" in text and upd["current_step"] == 0, text)
    state.update(upd)
    text, upd = step("next", state)
    check("next → 第 2/3 步", "第 2/3 步：炒蛋" in text and upd["current_step"] == 1, text)
    state.update(upd)
    text, upd = step("current", state)
    check("current 不改 state", "第 2/3 步：炒蛋" in text and upd == {}, f"{text} {upd}")
    state.update(step("next", state)[1])
    text, upd = step("next", state)
    check("最後一步之後 → 完成並清空", "已完成" in text and upd["cooking_steps"] == [], text)
    text, _ = step("start", {}, recipe_name="x", steps=[])
    check("start 沒給 steps → 錯誤", "不可為空" in text, text)

    print("3. HITL")
    cfg = _hitl_interrupt_on()["inventory"]
    req = lambda args: SimpleNamespace(tool_call={"name": "inventory", "args": args})
    check("remove 3 樣 → 攔", cfg["when"](req({"action": "remove", "items": ["a", "b", "c"]})))
    check("remove 2 樣 → 不攔", not cfg["when"](req({"action": "remove", "items": ["a", "b"]})))
    check("add 5 樣 → 不攔", not cfg["when"](req({"action": "add", "items": list("abcde")})))
    check("get → 不攔", not cfg["when"](req({"action": "get"})))

    print("4. 舊工具名稱")
    names = {t.name for t in tools.ALL_TOOLS}
    check("ALL_TOOLS 有 inventory / step_tracker、沒有舊名稱",
          {"inventory", "step_tracker"} <= names and not names & set(OLD_NAMES), f"{sorted(names)}")
    check("訪客可用 step_tracker、不能用 inventory",
          {t.name for t in tools.GUEST_TOOLS} >= {"step_tracker"}
          and "inventory" not in {t.name for t in tools.GUEST_TOOLS})
    check("system prompt 沒有舊名稱", not any(n in DEFAULT_SYSTEM_PROMPT for n in OLD_NAMES))

    print("5. 歷史紀錄顯示摘要")
    summary = HumanMessage(
        "Here is a summary of the conversation to date:\n\n使用者對蝦過敏，正在學番茄炒蛋",
        additional_kwargs={"lc_source": "summarization"}, id="s1")
    rendered = render_messages([
        summary,
        AIMessage("", tool_calls=[{"name": "step_tracker", "args": {"action": "next"}, "id": "c1"}], id="a0"),
        AIMessage("第 2 步：炒蛋", id="a1"),
        HumanMessage("<user_input>下一步</user_input>", id="h1"),
    ])
    check("摘要排第一則、role=summary",
          rendered[0] == {"id": "s1", "role": "summary", "content": "使用者對蝦過敏，正在學番茄炒蛋"}, f"{rendered[:1]}")
    check("其餘照舊：純 tool_calls 不顯示、使用者訊息去掉標籤",
          [(m["role"], m["content"]) for m in rendered[1:]] == [("assistant", "第 2 步：炒蛋"), ("user", "下一步")],
          f"{rendered[1:]}")

    asyncio.run(agent_flow())

    print(f"\n通過 {passed}，失敗 {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
