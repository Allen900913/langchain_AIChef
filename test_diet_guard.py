"""驗證 DietarySafetyGuard（零信任飲食安全驗證迴圈）是否真的會擋下過敏原推薦。

不連真實 Postgres：用 InMemoryStore 模擬 store，用假的 handler 模擬「模型這輪回了什麼」，
只測 app.py 裡 DietarySafetyGuard 這顆 middleware 本身的邏輯是否正確。

執行：uv run python test_diet_guard.py
"""
import asyncio

from dotenv import load_dotenv
load_dotenv()

from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import var_child_runnable_config
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.store.memory import InMemoryStore

from app.agents.app import DietarySafetyGuard, _diet_allergies, model

GREEN, RED, YEL, RST = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def ok(cond):
    return f"{GREEN}PASS{RST}" if cond else f"{RED}FAIL{RST}"


TEST_USER_ID = "test_diet_guard_user"


def seed_store(allergies: list[str]) -> InMemoryStore:
    store = InMemoryStore()
    store.put(("memories", TEST_USER_ID, "diet"), "diet_1", {"content": {"allergies": allergies}})
    return store


def make_request(store) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[],
        runtime=SimpleNamespace(store=store),
        state={},
    )


def with_user_config(user_id: str):
    """模擬 langgraph 執行期會設好的 contextvar，讓 get_config() 在測試裡也能讀到。"""
    return var_child_runnable_config.set({"configurable": {"user_id": user_id}})


# ==============================================================================
# Part A：_find_hit 純邏輯比對
# ==============================================================================
print("=" * 70)
print("Part A — _find_hit 命中判斷")
print("=" * 70)

a1_result = [AIMessage(content="你可以試試蝦仁炒飯，做法簡單又快速。")]
a1_hit = DietarySafetyGuard._find_hit(a1_result, ["蝦"])
print(f"[A1] {ok(a1_hit == '蝦')} 純文字推薦命中過敏原 → hit={a1_hit!r}")

a2_result = [AIMessage(content="", tool_calls=[
    {"name": "step_tracker_start", "args": {"recipe_name": "番茄炒蛋", "steps": ["切番茄", "打蛋", "下鍋炒"]}, "id": "1"},
])]
a2_hit = DietarySafetyGuard._find_hit(a2_result, ["蝦"])
print(f"[A2] {ok(a2_hit is None)} 無過敏原食譜不誤判 → hit={a2_hit!r}")

a3_result = [AIMessage(content="", tool_calls=[
    {"name": "step_tracker_start", "args": {"recipe_name": "蒜蝦義大利麵", "steps": ["蝦去殼", "煮麵", "拌炒"]}, "id": "2"},
])]
a3_hit = DietarySafetyGuard._find_hit(a3_result, ["蝦"])
print(f"[A3] {ok(a3_hit == '蝦')} step_tracker_start 工具參數裡的過敏原也抓得到 → hit={a3_hit!r}")


# ==============================================================================
# Part B：awrap_model_call 完整迴圈行為
# ==============================================================================
print("\n" + "=" * 70)
print("Part B — awrap_model_call 重試/攔截行為")
print("=" * 70)


async def run_b1_retry_then_clean():
    """模型第一次違規，重試後第二次乾淨 → 應該回傳乾淨版本，且只重試 1 次。"""
    store = seed_store(["蝦"])
    guard = DietarySafetyGuard()
    calls = {"n": 0}

    async def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(result=[AIMessage(content="推薦你做蝦仁炒飯！")])
        return ModelResponse(result=[AIMessage(content="推薦你做番茄炒蛋！")])

    token = with_user_config(TEST_USER_ID)
    try:
        resp = await guard.awrap_model_call(make_request(store), handler)
    finally:
        var_child_runnable_config.reset(token)

    final_text = resp.result[-1].content
    cond = calls["n"] == 2 and "番茄炒蛋" in final_text and "蝦" not in final_text
    print(f"[B1] {ok(cond)} 第一次違規→自動重試→回傳乾淨內容 (calls={calls['n']}, final={final_text!r})")


async def run_b2_persistent_violation():
    """模型每次都違規 → 重試 MAX_RETRY 次後，強制攔截改寫成安全提示。"""
    store = seed_store(["蝦"])
    guard = DietarySafetyGuard()
    calls = {"n": 0}

    async def handler(req):
        calls["n"] += 1
        return ModelResponse(result=[AIMessage(content="就是要推薦蝦仁炒飯！")])

    token = with_user_config(TEST_USER_ID)
    try:
        resp = await guard.awrap_model_call(make_request(store), handler)
    finally:
        var_child_runnable_config.reset(token)

    final_text = resp.result[-1].content
    cond = calls["n"] == guard.MAX_RETRY + 1 and "蝦仁炒飯" not in final_text and "過敏原" in final_text
    print(f"[B2] {ok(cond)} 持續違規→重試 {guard.MAX_RETRY} 次後強制攔截 (calls={calls['n']}, final={final_text!r})")


async def run_b3_no_allergy_skips_check():
    """使用者沒有過敏原記錄 → 不檢查、不重試，直接回傳 handler 的結果。"""
    store = seed_store([])  # 空清單，等同沒有過敏原
    guard = DietarySafetyGuard()
    calls = {"n": 0}

    async def handler(req):
        calls["n"] += 1
        return ModelResponse(result=[AIMessage(content="推薦你做蝦仁炒飯！")])

    token = with_user_config(TEST_USER_ID)
    try:
        resp = await guard.awrap_model_call(make_request(store), handler)
    finally:
        var_child_runnable_config.reset(token)

    cond = calls["n"] == 1 and "蝦仁炒飯" in resp.result[-1].content
    print(f"[B3] {ok(cond)} 無過敏原記錄時直接放行、不額外檢查 (calls={calls['n']})")


async def main():
    await run_b1_retry_then_clean()
    await run_b2_persistent_violation()
    await run_b3_no_allergy_skips_check()


asyncio.run(main())
