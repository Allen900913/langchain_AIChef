"""離線測試：背景反思寫入偏好時的保護（app/agents/memory_guard.py）。不呼叫任何模型 API。

測試項目：
1. 白名單：kind 錯、多餘欄位、型別錯都丟掉；合格的放行並正規化
2. 誤存的 PatchDoc 套用回原檔案；套不上就丟掉
3. 過敏原只增不減（一般更新、刪除、patch 三種路徑）
4. 非偏好 namespace 原樣轉發；被丟掉的寫入不影響 batch 結果的對應
5. 同步（ReflectionExecutor 走這條）與非同步路徑結果一致
6. 用錄到的模型輸出重播「functions.PatchDoc」：ToolNameFixingChatGroq 從源頭修好
7. 主 agent 模式：*_profile_manage 寫入補上 kind（之後的反思不會 KeyError）、不合格拋錯、不保護過敏原

執行方式：python app/tests/test_memory_guard.py
"""

import asyncio
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq
from langgraph.store.base import PutOp
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

from app.agents.memory_guard import ProfileGuardStore, ToolNameFixingChatGroq
from app.agents.tools import _PROFILE_DOMAINS, _profile_ns

USER = "test_user"
DIET = ("memories", USER, "diet")
KITCHEN = ("memories", USER, "kitchen")

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


def diet(**kw):
    base = {"allergies": [], "dislikes": [], "diet_type": None, "flavor_preferences": [],
            "cuisine_preferences": [], "health_goals": []}
    return {**base, **kw}


def doc_id(ns, key):
    return uuid.uuid5(uuid.NAMESPACE_DNS, str((*ns, key))).hex


def fresh(existing_diet: dict | None = None):
    inner = InMemoryStore()
    if existing_diet is not None:
        inner.put(DIET, "profile", {"kind": "DietProfile", "content": existing_diet})
    return inner, ProfileGuardStore(inner, _PROFILE_DOMAINS)


def records(inner, ns=DIET):
    return [(it.key, it.value) for it in inner.search(ns, limit=20)]


def test_allowlist():
    print("\n[1] 白名單")
    inner, g = fresh(diet(dislikes=["香菜"]))
    g.put(DIET, "profile", {"kind": "PatchDoc", "content": diet()})
    check("kind 不是 DietProfile → 丟掉", records(inner)[0][1]["content"]["dislikes"] == ["香菜"])

    g.put(DIET, "profile", {"kind": "DietProfile", "content": {**diet(), "favorite_color": "red"}})
    check("有 schema 以外的欄位 → 丟掉（不能被 pydantic 忽略後當成空檔案寫入）",
          records(inner)[0][1]["content"]["dislikes"] == ["香菜"])

    g.put(DIET, "profile", {"kind": "DietProfile", "content": diet(dislikes="香菜")})
    check("型別錯（dislikes 不是 list）→ 丟掉", records(inner)[0][1]["content"]["dislikes"] == ["香菜"])

    g.put(DIET, "profile", {"kind": "DietProfile", "content": {"dislikes": ["香菜", "苦瓜"]}})
    content = records(inner)[0][1]["content"]
    check("合格 → 放行，缺的欄位補成 schema 預設值",
          content["dislikes"] == ["香菜", "苦瓜"] and content["allergies"] == [] and "diet_type" in content,
          str(content))
    check("統計：rejected = 3", g.stats["rejected"] == 3, str(dict(g.stats)))

    g.put(KITCHEN, "profile", {"kind": "DietProfile", "content": diet()})
    check("kitchen 領域寫入 DietProfile → 丟掉", records(inner, KITCHEN) == [])
    g.put(KITCHEN, "profile", {"kind": "KitchenProfile", "content": {"equipment": ["電鍋"]}})
    check("kitchen 領域寫入 KitchenProfile → 放行", records(inner, KITCHEN)[0][1]["content"]["equipment"] == ["電鍋"])


def test_patch():
    print("\n[2] 誤存的 PatchDoc")
    inner, g = fresh(diet())
    g.put(DIET, str(uuid.uuid4()), {"kind": "PatchDoc", "content": {
        "json_doc_id": doc_id(DIET, "profile"), "planned_edits": "",
        "patches": [{"op": "add", "path": "/dislikes/-", "value": "香菜"}]}})
    recs = records(inner)
    check("套用到原檔案，沒有多出新記錄", len(recs) == 1 and recs[0][0] == "profile", str(recs))
    check("原檔案的 dislikes 變成 [香菜]", recs[0][1]["content"]["dislikes"] == ["香菜"], str(recs))

    g.put(DIET, str(uuid.uuid4()), {"kind": "PatchDoc", "content": {
        "json_doc_id": doc_id(DIET, "profile"), "planned_edits": "",
        "patches": [{"op": "remove", "path": "/no_such_field/3"}]}})
    check("套不上的 patch → 丟掉，原檔案不變", records(inner) == recs and g.stats["patch_dropped"] == 1)

    g.put(DIET, str(uuid.uuid4()), {"kind": "PatchDoc", "content": {
        "json_doc_id": doc_id(DIET, "profile"), "planned_edits": "",
        "patches": [{"op": "add", "path": "/hacked", "value": True}]}})
    check("套用後格式不合 schema → 白名單擋下", "hacked" not in records(inner)[0][1]["content"])


def test_allergies():
    print("\n[3] 過敏原只增不減")
    inner, g = fresh(diet(allergies=["蝦"], dislikes=["香菜"]))
    g.put(DIET, "profile", {"kind": "DietProfile", "content": diet(dislikes=["香菜"])})
    check("一般更新拿掉過敏原 → 補回", records(inner)[0][1]["content"]["allergies"] == ["蝦"])

    g.put(DIET, str(uuid.uuid4()), {"kind": "PatchDoc", "content": {
        "json_doc_id": doc_id(DIET, "profile"), "planned_edits": "",
        "patches": [{"op": "remove", "path": "/allergies/0", "value": None}]}})
    check("patch 刪過敏原 → 補回", records(inner)[0][1]["content"]["allergies"] == ["蝦"])

    g.delete(DIET, "profile")
    recs = records(inner)
    check("整筆刪除 → 改成只保留過敏原",
          len(recs) == 1 and recs[0][1]["content"]["allergies"] == ["蝦"] and recs[0][1]["content"]["dislikes"] == [],
          str(recs))

    g.put(DIET, "profile", {"kind": "DietProfile", "content": diet(allergies=["花生"])})
    check("新增過敏原 → 新舊都在", sorted(records(inner)[0][1]["content"]["allergies"]) == ["花生", "蝦"])


def test_passthrough():
    print("\n[4] 非偏好 namespace 與 batch 對應")
    inner, g = fresh(diet())
    g.put(("inventory", USER), "items", {"anything": "goes"})
    check("非偏好 namespace 原樣寫入", inner.get(("inventory", USER), "items").value == {"anything": "goes"})
    results = g.batch([
        PutOp(DIET, "profile", {"kind": "Nope", "content": {}}),
        PutOp(("inventory", USER), "x", {"a": 1}),
    ])
    check("被丟掉的寫入在結果中補 None，長度與輸入相同", results == [None, None], str(results))


async def test_async_matches_sync():
    print("\n[5] 非同步路徑")
    inner, g = fresh(diet(allergies=["蝦"]))
    await g.aput(DIET, "profile", {"kind": "PatchDoc", "content": diet()})
    await g.aput(DIET, str(uuid.uuid4()), {"kind": "PatchDoc", "content": {
        "json_doc_id": doc_id(DIET, "profile"), "planned_edits": "",
        "patches": [{"op": "remove", "path": "/allergies/0"}, {"op": "add", "path": "/dislikes/-", "value": "苦瓜"}]}})
    content = (await inner.aget(DIET, "profile")).value["content"]
    check("aput：白名單、patch 修正、過敏原保護都生效",
          content["allergies"] == ["蝦"] and content["dislikes"] == ["苦瓜"] and len(await inner.asearch(DIET)) == 1,
          str(content))


async def test_replay_functions_prefix():
    print("\n[6] 重播「functions.PatchDoc」（不打 API）")

    async def fake_agenerate(self, messages, stop=None, run_manager=None, **kw):
        last = messages[-1]
        if isinstance(last, ToolMessage) and "Unrecognized tool name" in str(last.content):
            tc = {"name": "PatchFunctionName", "id": "c2",
                  "args": {"fixed_name": "PatchDoc", "json_doc_id": last.tool_call_id, "reasoning": ["prefix"]}}
        else:
            tc = {"name": "functions.PatchDoc", "id": "c1", "args": {
                "json_doc_id": doc_id(("memories", "eval", "diet"), "profile"), "planned_edits": "add 香菜",
                "patches": [{"op": "add", "path": "/dislikes/-", "value": "香菜"}]}}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tc]))])

    original = ChatGroq._agenerate
    ChatGroq._agenerate = fake_agenerate  # 子類別的 super()._agenerate 也會走到假的
    try:
        async def run(model_cls, guard=False):
            ns = ("memories", "eval", "diet")
            inner = InMemoryStore()
            await inner.aput(ns, "profile", {"kind": "DietProfile", "content": diet()})
            store = ProfileGuardStore(inner, _PROFILE_DOMAINS) if guard else inner
            mgr = create_memory_store_manager(
                model_cls(model="openai/gpt-oss-20b", api_key="offline"), namespace=_profile_ns("diet"),
                schemas=[_PROFILE_DOMAINS["diet"]], instructions="x", store=store, enable_deletes=True)
            msgs = [HumanMessage("<user_input>我很討厭香菜，以後都不要放</user_input>")]
            await mgr.ainvoke({"messages": msgs}, config={"configurable": {"user_id": "eval"}})
            return [(it.value["kind"], it.value["content"]) for it in await inner.asearch(ns)]

        plain = await run(ChatGroq)
        check("對照組（原版 ChatGroq）重現 bug：多一筆 PatchDoc", any(k == "PatchDoc" for k, _ in plain), str(plain))
        fixed = await run(ToolNameFixingChatGroq)
        check("ToolNameFixingChatGroq：只有一筆、dislikes = [香菜]",
              len(fixed) == 1 and fixed[0][1]["dislikes"] == ["香菜"], str(fixed))
        guarded = await run(ToolNameFixingChatGroq, guard=True)
        check("加上 ProfileGuardStore：LangMem 正常寫入不會被白名單誤擋", guarded == fixed, str(guarded))
        both_bug = await run(ChatGroq, guard=True)
        check("只有 ProfileGuardStore（原版模型）：一樣修好", both_bug == fixed, str(both_bug))
    finally:
        ChatGroq._agenerate = original


async def test_agent_mode():
    print("\n[7] 主 agent 模式（*_profile_manage 工具的寫入）")
    inner = InMemoryStore()
    g = ProfileGuardStore(inner, _PROFILE_DOMAINS, mode="agent")
    # LangMem create_manage_memory_tool 的寫入格式：只有 content，沒有 kind
    await g.aput(DIET, "m1", {"content": {"allergies": ["蝦"]}})
    value = (await inner.aget(DIET, "m1")).value
    check("補上 kind，缺的欄位補預設值", value["kind"] == "DietProfile" and value["content"]["dislikes"] == [], str(value))

    async def no_tool_call(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def reflect(store):
        mgr = create_memory_store_manager(
            ChatGroq(model="openai/gpt-oss-20b", api_key="offline"), namespace=_profile_ns("diet"),
            schemas=[_PROFILE_DOMAINS["diet"]], instructions="x", store=store, enable_deletes=True)
        try:
            await mgr.ainvoke({"messages": [HumanMessage("我討厭香菜")]}, config={"configurable": {"user_id": USER}})
            return "ok"
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    original = ChatGroq._agenerate
    ChatGroq._agenerate = no_tool_call
    try:
        raw = InMemoryStore()
        await raw.aput(DIET, "m1", {"content": {"allergies": ["蝦"]}})
        check("對照組（沒包保護）：之後的反思 KeyError", (await reflect(raw)).startswith("KeyError"))
        check("包了保護：之後的反思正常", await reflect(ProfileGuardStore(inner, _PROFILE_DOMAINS)) == "ok")
    finally:
        ChatGroq._agenerate = original

    try:
        await g.aput(DIET, "m1", {"content": {"allergies": ["蝦"], "favorite_color": "red"}})
        check("格式不合 → 拋錯讓模型重寫", False, "沒有拋錯")
    except ValueError as exc:
        check("格式不合 → 拋錯讓模型重寫", "favorite_color" in str(exc), str(exc))

    await g.aput(DIET, "m1", {"content": {"allergies": []}})
    check("不做過敏原保護：使用者明確要求的更新照寫",
          (await inner.aget(DIET, "m1")).value["content"]["allergies"] == [])
    await g.aput(("inventory", USER), "items", {"items": ["雞蛋"]})
    check("非偏好 namespace 原樣寫入", (await inner.aget(("inventory", USER), "items")).value == {"items": ["雞蛋"]})

    # 反思模式 patch 到一筆早期沒有 kind 的記錄：套用後補上 kind
    legacy = InMemoryStore()
    await legacy.aput(DIET, "old", {"content": diet()})
    await ProfileGuardStore(legacy, _PROFILE_DOMAINS).aput(DIET, str(uuid.uuid4()), {"kind": "PatchDoc", "content": {
        "json_doc_id": doc_id(DIET, "old"), "planned_edits": "",
        "patches": [{"op": "add", "path": "/dislikes/-", "value": "香菜"}]}})
    value = (await legacy.aget(DIET, "old")).value
    check("patch 套到沒有 kind 的舊記錄：套用並補上 kind",
          value.get("kind") == "DietProfile" and value["content"]["dislikes"] == ["香菜"], str(value))


async def main():
    print("=" * 60)
    print("memory_guard 離線測試")
    print("=" * 60)
    test_allowlist()
    test_patch()
    test_allergies()
    test_passthrough()
    await test_async_matches_sync()
    await test_replay_functions_prefix()
    await test_agent_mode()
    print(f"\n結果：{passed} 通過、{failed} 失敗")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
