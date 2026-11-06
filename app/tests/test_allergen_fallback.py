"""離線測試：過敏原判定器（LLM）失敗時改用關鍵字比對（app/agents/app.py）。不呼叫任何模型 API。

測試項目：
1. _keyword_allergen_hit：直接寫出過敏原 → 命中；「不含 / 避開 / 代替」緊接在前、或後面接「過敏」→ 不算
2. 不是緊接的否定詞（「不要錯過蝦仁」）、列舉的後項，照樣算命中：寧可多擋
3. _find_recommended_allergen：判定器拋錯時走關鍵字比對，不再一律回傳第一個過敏原

執行方式：python app/tests/test_allergen_fallback.py
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import app.agents.app as agent_app  # noqa: E402
from app.agents.app import _find_recommended_allergen, _keyword_allergen_hit  # noqa: E402

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


def test_keyword():
    print("\n[1] 關鍵字比對")
    allergens = ["蝦", "花生"]
    cases = [
        ("推薦你做蒜香蝦仁義大利麵", "蝦"),
        ("最後撒上花生碎增加口感", "花生"),
        ("這道菜不含蝦，可以放心吃", None),
        ("我幫你避開花生，改用腰果", None),
        ("用雞胸肉代替蝦仁，一樣很下飯", None),
        ("記得你對蝦過敏，所以這次推薦雞肉料理", None),
        ("完全沒有使用花生", None),
        ("今晚吃番茄炒蛋吧", None),
        # 同一則回覆裡先說不含、後面又推薦：後面那次要抓到
        ("這道不加蝦。另外推薦一道鹽酥蝦當配菜", "蝦"),
    ]
    for content, expected in cases:
        got = _keyword_allergen_hit(allergens, content)
        check(f"{content} → {expected}", got == expected, f"得到 {got}")


def test_not_adjacent():
    print("\n[2] 否定詞沒有緊接過敏原 → 照樣算命中")
    for content in ("不要錯過這道蝦仁炒蛋", "去掉蝦頭後下鍋", "不用多久蝦就熟了"):
        check(content, _keyword_allergen_hit(["蝦"], content) == "蝦")
    # 已知限制：列舉時只有第一個認得出否定。多擋的代價是模型被要求重寫一次，可以接受
    check("「沒有使用花生或花生油」→ 後面的花生油被當成命中（已知限制，偏向多擋）",
          _keyword_allergen_hit(["花生"], "完全沒有使用花生或花生油") == "花生")


async def test_judge_failure():
    print("\n[3] 判定器失敗時改用關鍵字比對")

    class BrokenModel:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("429 rate_limit_exceeded（模擬額度用完）")

    original = agent_app.summary_model
    agent_app.summary_model = BrokenModel()
    try:
        safe = await _find_recommended_allergen(["蝦"], "今晚做番茄炒蛋，十分鐘就好")
        check("沒提到過敏原 → 放行（以前會一律當成違規、收回回答）", safe is None, f"得到 {safe}")
        hit = await _find_recommended_allergen(["蝦"], "推薦你做胡椒蝦")
        check("直接寫出過敏原 → 仍然攔下", hit == "蝦", f"得到 {hit}")
    finally:
        agent_app.summary_model = original


async def test_tool_call_text():
    print("\n[4] DietarySafetyGuard：帶工具呼叫的訊息，文字也要檢查（紅隊測試 D3b）")
    from types import SimpleNamespace

    from langchain.agents.middleware.types import ModelResponse
    from langchain_core.messages import AIMessage

    judged = []

    async def fake_judge(allergens, content):
        judged.append(content)
        return _keyword_allergen_hit(allergens, content)

    async def fake_allergies(_user_id):
        return ["花生"]

    class Request(SimpleNamespace):
        def override(self, **kw):
            return Request(**{**self.__dict__, **kw})

    def tool_msg(text):
        return AIMessage(content=text, tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}])

    async def run(*responses):
        queue = list(responses)
        calls = []

        async def handler(_req):
            calls.append(1)
            return ModelResponse(result=[queue.pop(0)])

        judged.clear()
        req = Request(runtime=SimpleNamespace(store=object()), messages=[])
        out = await agent_app.DietarySafetyGuard().awrap_model_call(req, handler)
        return out.result[-1], len(calls)

    originals = (agent_app._find_recommended_allergen, agent_app._diet_allergies, agent_app.get_config)
    agent_app._find_recommended_allergen = fake_judge
    agent_app._diet_allergies = fake_allergies
    agent_app.get_config = lambda: {"configurable": {"user_id": "u"}}
    try:
        msg, n = await run(tool_msg("推薦你做宮保雞丁，最後加入花生拌勻。我再幫你查詳細做法"))
        check("文字含過敏原 → 拿掉文字", msg.content == "", repr(msg.content))
        check("工具呼叫保留、不重試", msg.tool_calls and msg.tool_calls[0]["name"] == "web_search" and n == 1)

        msg, n = await run(tool_msg("我幫你查一下番茄炒蛋的做法"))
        check("文字沒有過敏原 → 原樣放行", msg.content == "我幫你查一下番茄炒蛋的做法" and len(judged) == 1)

        msg, n = await run(tool_msg(""))
        check("沒有文字 → 不呼叫判定器（不多花額度）", msg.tool_calls and judged == [])

        # D3b 的情境：最終回覆含花生 → 要求重寫 → 模型改成「推薦花生 + 呼叫工具」
        msg, n = await run(AIMessage(content="宮保雞丁：雞丁、花生、乾辣椒"),
                           tool_msg("好的，這是花生版宮保雞丁，我再查一下份量"))
        check("重寫時改成呼叫工具 → 文字一樣被拿掉", msg.content == "" and msg.tool_calls and n == 2, repr(msg.content))

        msg, n = await run(AIMessage(content="花生雞丁"), AIMessage(content="花生炒肉"), AIMessage(content="花生湯"))
        check("最終回覆重試用完仍命中 → 改成安全提示（原本的行為不變）", "收回" in msg.content and n == 3, msg.content)
    finally:
        agent_app._find_recommended_allergen, agent_app._diet_allergies, agent_app.get_config = originals


async def main():
    print("=" * 60)
    print("過敏原判定器備案 離線測試")
    print("=" * 60)
    test_keyword()
    test_not_adjacent()
    await test_judge_failure()
    await test_tool_call_text()
    print(f"\n結果：{passed} 通過、{failed} 失敗")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
