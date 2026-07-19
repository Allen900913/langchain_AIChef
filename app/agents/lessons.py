"""教訓記憶（procedural memory）：把程式可驗證的失敗訊號累積成常駐行為規則。

設計原則：
1. 只從確定性訊號寫入——不讓模型自述犯錯（模型往往不知道自己錯了，
   這與 DietarySafetyGuard 的零信任設計一致）。
2. 用結構化 key 而非語意相似度比對——樸素相似度檢索在「表面相似、根因不同」
   的案例上假陽性率可達 75%，而錯誤注入會沿整條決策路徑傳播，比沒撈到更糟。
3. 只有重複發生過的才升格常駐——單次失誤多半是情境特例，不是模式。
"""

import asyncio
from datetime import datetime, timedelta, timezone

_LESSON_MIN_HITS = 2       # 重複這麼多次才升格為常駐規則
_LESSON_MAX_INJECT = 5     # 每輪最多注入幾條（成本封頂 ~150 token）
_LESSON_TTL_DAYS = 30      # 超過這麼久沒再觸發就不再注入

# 同一個 key 在這段時間內只累加一次 hits。
# hits 的語意是「在幾個不同場合發生過」，不是「總共觸發過幾次」——單一次請求內
# 同一個訊號可能被觸發多次（例如 DietarySafetyGuard 在一輪 run 的多個 model round
# 各命中一次），若照單全收，一次請求就能灌破門檻，「必須重複發生」的設計就失效了。
# 取 MAX_TIMEOUT 的值：一次 run 最長就這麼久，故同一 run 內必定只計一次。
_LESSON_DEBOUNCE_SECONDS = 120

# 由 init_agent_infra() 設定。關掉時 record/active 都直接短路，供 eval A/B 比較
# 「有無教訓注入」的表現差異——否則無從得知這套機制是幫倒忙還是真有效。
_enabled = True


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


# 持有 task 參照，避免 fire-and-forget 的背景任務在完成前被 GC 回收
_background_tasks: set[asyncio.Task] = set()


def fire_and_forget(coro) -> None:
    """背景記錄教訓，不阻塞主流程（記錄失敗不該拖慢或中斷使用者的請求）。"""
    if not _enabled:
        coro.close()
        return
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# signal → 人話規則模板。用固定模板而非 LLM 生成：訊號種類有限且已知，
# 模板零成本、零幻覺；等模板真的不夠用再換成 LLM 生成。
# 措辭一律用「該怎麼做」而非「不要怎樣」，並保留例外條件避免過度矯正。
_LESSON_TEMPLATES = {
    "repeat": "你曾以完全相同的參數反覆呼叫 `{detail}` 導致卡住。若第一次結果不理想，"
              "請改變參數或改用其他工具，不要重複同樣的呼叫。",
    "hitl_reject": "使用者曾多次拒絕你執行 `{detail}`。除非使用者這次明確要求，"
                   "否則不要主動提議這個動作。",
    "allergen": "你曾在推薦內容中誤用使用者的過敏原「{detail}」。推薦前請先確認"
                "菜名與所有食材都不含它。",
    "soft_landing": "你曾在單輪內用掉過多工具呼叫才收尾。請一次用 profiles_get / "
                    "inventory_get 取足資訊，避免零碎地反覆查詢。",
    "search_thrash": "你曾在同一次任務中反覆更換關鍵字搜尋多次才得到堪用結果。"
                     "搜尋食譜時請一次帶足限定詞（菜名 +「食譜」+「步驟」或「做法」），"
                     "而不是先用籠統關鍵字再逐次修正。",
}

# 刻意不收錄的訊號（記了反而有害，見 README 註解）：
#   - web_search 逾時 / Tavily API 故障：環境噪音而非行為模式，換一次執行未必再發生；
#     注入只會讓模型對其實正常的工具無謂畏縮。處置建議已寫在工具回傳字串裡。
#   - guardrail 命中注入：防禦已生效（片段已切除），對未來決策無可行動教訓。
# 判準：「這個失敗，換一次執行還會不會發生？」會（策略問題）才記，不會（環境抖動）不記。


def _lesson_ns(user_id: str) -> tuple[str, str]:
    return ("lessons", user_id)


async def record_lesson(store, user_id: str, signal: str, detail: str = "") -> None:
    """記錄一次失敗訊號並累加 hits。失敗不拋例外，教訓記錄壞掉不該影響主流程。"""
    if not _enabled or store is None or signal not in _LESSON_TEMPLATES:
        return
    key = f"{signal}:{detail}" if detail else signal
    now = datetime.now(timezone.utc)
    try:
        existing = await store.aget(_lesson_ns(user_id), key)
        hits = existing.value.get("hits", 0) if existing else 0

        # debounce：同一 run 內重複觸發只算一次「場合」。不更新 last_hit，
        # 否則持續觸發會不斷延長靜默窗、讓下一個真正的場合也被吃掉。
        if existing:
            try:
                if (now - datetime.fromisoformat(existing.value["last_hit"])).total_seconds() \
                        < _LESSON_DEBOUNCE_SECONDS:
                    return
            except (KeyError, ValueError):
                pass

        hits += 1
        await store.aput(_lesson_ns(user_id), key, {
            "signal": signal,
            "detail": detail,
            "hits": hits,
            "last_hit": now.isoformat(),
        })
        print(f"📓 [LESSON] {key} → hits={hits}"
              f"{'（已達門檻，將開始注入）' if hits == _LESSON_MIN_HITS else ''}")
    except Exception as exc:
        print(f"⚠️ [LESSON] 記錄失敗：{exc}")


async def active_lessons(store, user_id: str) -> list[str]:
    """取出已升格的常駐規則（hits 達門檻且未過期），按 hits 由高到低最多 N 條。"""
    if not _enabled or store is None:
        return []
    try:
        items = await store.asearch(_lesson_ns(user_id))
    except Exception as exc:
        print(f"⚠️ [LESSON] 讀取失敗：{exc}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=_LESSON_TTL_DAYS)
    fresh = []
    for it in items:
        v = it.value
        if v.get("hits", 0) < _LESSON_MIN_HITS or v.get("signal") not in _LESSON_TEMPLATES:
            continue
        try:
            if datetime.fromisoformat(v["last_hit"]) < cutoff:
                continue
        except (KeyError, ValueError):
            continue
        fresh.append(v)

    fresh.sort(key=lambda v: v["hits"], reverse=True)
    return [
        _LESSON_TEMPLATES[v["signal"]].format(detail=v.get("detail", ""))
        for v in fresh[:_LESSON_MAX_INJECT]
    ]
