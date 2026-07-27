"""端對端測試：驗證三大錯誤處理鏈路完整運作。

測試項目：
1. save_error_rule 工具能寫入 Store
2. active_error_rules 能從 Store 讀回（含 hits 門檻過濾）
3. Auditor Agent 能分析軌跡並自動呼叫 save_error_rule
4. 整條鏈路：寫入 → 讀取 → 注入 System Prompt 格式正確

執行方式：python app/tests/test_auditor.py
"""

import asyncio
import os
import sys
import hashlib
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# Windows UTF-8 修復
for _s in (sys.stdout, sys.stderr):
    if _s and getattr(_s, "encoding", None) != "utf-8":
        try: _s.reconfigure(encoding="utf-8")
        except: pass

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ── 被測模組 ──
from app.agents.auditor import (
    save_error_rule,
    active_error_rules,
    init_auditor,
    run_auditor,
    format_trajectory,
    _should_audit,
    _current_user_id,
    _ERROR_RULE_MIN_HITS,
    _DEBOUNCE_SECONDS,
)
import app.agents.auditor as auditor_mod

DB_URI = os.environ["DB_URI"]
TEST_USER = "__test_auditor__"
TEST_NS = ("error_lessons", TEST_USER)

# 統計
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


async def cleanup(store):
    """清除測試用的 namespace，避免髒資料干擾。"""
    try:
        items = await store.asearch(TEST_NS)
        for it in items:
            await store.adelete(TEST_NS, it.key)
    except Exception:
        pass


async def main():
    global passed, failed

    print("=" * 60)
    print("🧪 Auditor Agent 端對端測試")
    print("=" * 60)

    # ── 建立基礎設施 ──
    print("\n📦 建立 DB 連線與 Store...")
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    store = AsyncPostgresStore(pool)
    await store.setup()

    summary_model = init_chat_model(
        "llama-3.1-8b-instant",
        model_provider="groq",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        request_timeout=30.0,
    )
    print("  ✅ 基礎設施就緒")

    # 初始化 Auditor Agent
    init_auditor(summary_model, store)
    print("  ✅ Auditor Agent 初始化完成")

    # 清理舊測試資料
    await cleanup(store)

    # ==================================================================
    # 測試 1：save_error_rule 工具直接寫入 Store
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔬 測試 1：save_error_rule 直接寫入 Store")
    print("─" * 60)

    token = _current_user_id.set(TEST_USER)
    try:
        # 第一次寫入 → hits=1
        result1 = await save_error_rule.ainvoke({
            "trigger_condition": "呼叫 web_search 發生 TimeoutError",
            "corrective_action": "請換一組更精確的關鍵字重試",
        })
        print(f"  第一次寫入回傳: {result1}")
        check("第一次寫入成功", "✅" in result1, result1)
        check("第一次 hits=1", "Hits=1" in result1, result1)

        # 強制清除 debounce（把 last_hit 改到 3 分鐘前）
        trigger = "呼叫 web_search 發生 TimeoutError"
        key = hashlib.md5(trigger.encode("utf-8")).hexdigest()[:12]
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        existing = await store.aget(TEST_NS, key)
        if existing:
            v = existing.value.copy()
            v["last_hit"] = old_time
            await store.aput(TEST_NS, key, v)

        # 第二次寫入 → hits=2（達到門檻）
        result2 = await save_error_rule.ainvoke({
            "trigger_condition": "呼叫 web_search 發生 TimeoutError",
            "corrective_action": "請換一組更精確的關鍵字重試",
        })
        print(f"  第二次寫入回傳: {result2}")
        check("第二次寫入成功", "✅" in result2, result2)
        check("第二次 hits=2", "Hits=2" in result2, result2)
        check("達門檻提示出現", "門檻" in result2, result2)

        # 驗證 DB 裡的值
        item = await store.aget(TEST_NS, key)
        check("DB 中 hits=2", item is not None and item.value.get("hits") == 2,
              f"actual: {item.value if item else 'None'}")
        check("DB 中 trigger_condition 正確",
              item is not None and item.value.get("trigger_condition") == trigger)

    finally:
        _current_user_id.reset(token)

    # ==================================================================
    # 測試 2：Debounce 機制
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔬 測試 2：Debounce 機制（同一 run 內不重複累計）")
    print("─" * 60)

    token = _current_user_id.set(TEST_USER)
    try:
        result3 = await save_error_rule.ainvoke({
            "trigger_condition": "呼叫 web_search 發生 TimeoutError",
            "corrective_action": "請換一組更精確的關鍵字重試",
        })
        print(f"  Debounce 寫入回傳: {result3}")
        check("Debounce 生效（跳過）", "Debounce" in result3 or "跳過" in result3, result3)

        # 再次確認 hits 沒有增加
        item = await store.aget(TEST_NS, key)
        check("Debounce 後 hits 仍=2",
              item is not None and item.value.get("hits") == 2,
              f"actual hits: {item.value.get('hits') if item else 'None'}")
    finally:
        _current_user_id.reset(token)

    # ==================================================================
    # 測試 3：active_error_rules 讀取與門檻過濾
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔬 測試 3：active_error_rules 讀取與門檻過濾")
    print("─" * 60)

    # 寫入一條 hits=1（未達門檻）的規則
    token = _current_user_id.set(TEST_USER)
    try:
        await save_error_rule.ainvoke({
            "trigger_condition": "測試用低頻規則_不應被注入",
            "corrective_action": "這條不應該出現在 prompt 中",
        })
    finally:
        _current_user_id.reset(token)

    # 讀取
    rules = await active_error_rules(TEST_USER)
    print(f"  讀取到 {len(rules)} 條規則:")
    for r in rules:
        print(f"    → {r}")

    check("至少讀取到 1 條規則", len(rules) >= 1, f"got {len(rules)}")
    check("包含 TimeoutError 規則",
          any("TimeoutError" in r for r in rules),
          f"rules: {rules}")
    check("不包含未達門檻的規則",
          not any("不應被注入" in r for r in rules),
          f"rules: {rules}")

    # 驗證格式：應為 "當「...」時，..."
    if rules:
        check("規則格式正確（當…時，…）",
              rules[0].startswith("當「") and "時，" in rules[0],
              f"actual: {rules[0][:60]}")

    # ==================================================================
    # 測試 4：format_trajectory 與 _should_audit
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔬 測試 4：format_trajectory 與 _should_audit")
    print("─" * 60)

    # 模擬一段有工具呼叫的軌跡
    mock_messages = [
        HumanMessage(content="幫我推薦低卡料理"),
        AIMessage(content="", tool_calls=[
            {"name": "profiles_get", "args": {}, "id": "tc1", "type": "tool_call"}
        ]),
        ToolMessage(content='{"diet": []}', tool_call_id="tc1", name="profiles_get"),
        AIMessage(content="", tool_calls=[
            {"name": "web_search", "args": {"query": "低卡料理食譜"}, "id": "tc2", "type": "tool_call"}
        ]),
        ToolMessage(
            content="[System_Error:TimeoutError] 搜尋逾時。請換一組更精確的關鍵字重試。",
            tool_call_id="tc2", name="web_search"
        ),
        AIMessage(content="很抱歉搜尋暫時無法使用，但我可以根據常見的低卡料理為您推薦…"),
    ]

    trajectory = format_trajectory(mock_messages)
    print(f"  軌跡長度: {len(trajectory)} 字")
    check("軌跡非空", len(trajectory) > 0)
    check("軌跡包含 [System_Error]", "System_Error" in trajectory, trajectory[:200])
    check("軌跡包含工具呼叫記錄", "web_search" in trajectory)

    should = _should_audit(mock_messages, "推薦低卡料理")
    check("_should_audit 判定為 True（有目標 + ≥2 次工具呼叫）", should)

    no_goal = _should_audit(mock_messages, "")
    check("_should_audit 無目標時為 False", not no_goal)

    # ==================================================================
    # 測試 5：Auditor Agent 完整執行（LLM 分析軌跡 + 呼叫工具）
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔬 測試 5：Auditor Agent 完整執行（LLM 分析含錯誤的軌跡）")
    print("─" * 60)

    # 先清理，用一個獨特的 user_id 避免被之前測試干擾
    TEST_USER_AGENT = "__test_auditor_agent__"
    agent_ns = ("error_lessons", TEST_USER_AGENT)
    try:
        items = await store.asearch(agent_ns)
        for it in items:
            await store.adelete(agent_ns, it.key)
    except:
        pass

    # 構造一段明顯有錯誤的軌跡（連續搜尋逾時 + 認知失敗標籤）
    error_messages = [
        HumanMessage(content="幫我找減脂雞胸肉食譜"),
        AIMessage(content="", tool_calls=[
            {"name": "web_search", "args": {"query": "減脂雞胸肉食譜"}, "id": "tc1", "type": "tool_call"}
        ]),
        ToolMessage(
            content="[System_Error:TimeoutError] 搜尋逾時。請換一組更精確的關鍵字重試。",
            tool_call_id="tc1", name="web_search"
        ),
        AIMessage(content="", tool_calls=[
            {"name": "web_search", "args": {"query": "減脂雞胸肉食譜"}, "id": "tc2", "type": "tool_call"}
        ]),
        ToolMessage(
            content="[System_Error:TimeoutError] 搜尋逾時。請換一組更精確的關鍵字重試。",
            tool_call_id="tc2", name="web_search"
        ),
        AIMessage(content="[COGNITIVE_FAILURE:REPEAT:web_search] web_search 以完全相同的參數連續呼叫 3 次，已被強制中斷。"),
    ]

    print("  正在執行 Auditor Agent（呼叫 LLM 分析軌跡）...")
    await run_auditor(error_messages, "找減脂雞胸肉食譜", TEST_USER_AGENT)
    print("  Auditor Agent 執行完畢")

    # 等一下讓背景任務完成
    await asyncio.sleep(1)

    # 檢查是否有寫入
    items = await store.asearch(agent_ns)
    rules_written = list(items)
    print(f"  Auditor Agent 寫入了 {len(rules_written)} 條規則:")
    for it in rules_written:
        v = it.value
        print(f"    → 情境: {v.get('trigger_condition', '?')}")
        print(f"      做法: {v.get('corrective_action', '?')}")
        print(f"      hits: {v.get('hits', '?')}")

    check("Auditor Agent 至少寫入 1 條規則", len(rules_written) >= 1,
          f"寫入了 {len(rules_written)} 條")

    # ==================================================================
    # 測試 6：System Prompt 注入格式
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔬 測試 6：模擬 System Prompt 注入")
    print("─" * 60)

    rules = await active_error_rules(TEST_USER)
    if rules:
        system_message = "你是一名私人廚師助理。"
        injected = (
            f"{system_message}\n\n【過去發生過的錯誤與防呆規則（務必遵守）】\n"
            + "\n".join(f"- {r}" for r in rules)
        )
        print(f"  注入後的 System Prompt 片段:")
        # 只顯示注入的部分
        inject_part = injected[len(system_message):]
        for line in inject_part.strip().split("\n"):
            print(f"    {line}")

        check("注入內容包含規則標題", "【過去發生過的錯誤與防呆規則" in injected)
        check("注入內容包含具體規則", "- 當「" in injected, injected[-200:])
    else:
        check("有規則可注入", False, "active_error_rules 回傳空")

    # ==================================================================
    # 清理
    # ==================================================================
    print("\n" + "─" * 60)
    print("🧹 清理測試資料")
    print("─" * 60)
    await cleanup(store)
    try:
        items = await store.asearch(agent_ns)
        for it in items:
            await store.adelete(agent_ns, it.key)
    except:
        pass
    print("  ✅ 測試資料已清除")

    # ── 總結 ──
    await pool.close()
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"🏁 測試結果：{passed}/{total} 通過", end="")
    if failed:
        print(f"，{failed} 個失敗 ❌")
    else:
        print(" ✅ 全部通過！")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    # Windows 需要 SelectorEventLoop 才能正常使用 psycopg async
    import selectors
    selector = selectors.SelectSelector()
    loop = asyncio.SelectorEventLoop(selector)
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(main())
    finally:
        loop.close()
    sys.exit(0 if success else 1)
