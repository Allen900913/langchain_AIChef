"""離線測試：tool router 的工具選擇（app/agents/router.py）。

不呼叫任何模型 API：Qdrant 用記憶體模式，embedding 用本機 fastembed，查詢字串直接用
2026-11-01 瀏覽器實測時 query rewrite 實際產出、當時漏抓工具的那幾句。

執行方式：python app/tests/test_tool_router.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8")

from qdrant_client import QdrantClient, models

from app.agents.embedding import (
    DENSE_VEC, SPARSE_VEC, embed_passages_sync, embed_query_sync, sparse_vectors_config, vectors_config,
)
from app.agents.router import _BASE_TOOLS, _RETRIEVE_LIMIT, _expand_families
from app.agents.tool_index import _tool_text, exclude_filter
from app.agents.tools import ALL_TOOLS

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


def main():
    print("1. 族群展開（純函式）")
    check("第一名是 kitchen 的 manage → 補上 diet / household",
          {"diet_profile_manage", "household_profile_manage"} <= _expand_families(["kitchen_profile_manage"]))
    check("排在後面的不展開（這族約 2000 token）",
          "diet_profile_manage" not in _expand_families(["step_tracker", "kitchen_profile_manage"]))
    check("不在族群裡的工具原樣保留、不多補",
          _expand_families(["nutrition_lookup", "web_search"]) == {"nutrition_lookup", "web_search"})

    print("2. 實測漏抓的查詢：合併工具 / 族群展開後都有")
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=vectors_config(), sparse_vectors_config=sparse_vectors_config())
    vectors = embed_passages_sync([_tool_text(t) for t in ALL_TOOLS])
    client.upsert("t", points=[
        models.PointStruct(id=i, vector=vectors[i], payload={"name": t.name}) for i, t in enumerate(ALL_TOOLS)
    ])

    def route(query: str) -> tuple[list[str], set[str]]:
        dense, sparse = embed_query_sync(query)
        flt = exclude_filter(_BASE_TOOLS)
        result = client.query_points(
            "t",
            prefetch=[
                models.Prefetch(query=dense, using=DENSE_VEC, limit=_RETRIEVE_LIMIT * 2, filter=flt),
                models.Prefetch(query=sparse, using=SPARSE_VEC, limit=_RETRIEVE_LIMIT * 2, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=_RETRIEVE_LIMIT,
        )
        dynamic = [p.payload["name"] for p in result.points]
        return dynamic, set(_BASE_TOOLS) | _expand_families(dynamic)

    for query, must in (
        ("不要有蝦", "diet_profile_manage"),
        ("先給我第一步，番茄炒蛋", "step_tracker"),
        ("請給我番茄炒蛋的下一步", "step_tracker"),
        ("一步一步教我做番茄炒蛋", "step_tracker"),
    ):
        # 只看動態檢索（含族群展開），不算基底工具：inventory 是基底，算進去就一定會過
        dynamic, _ = route(query)
        check(f"「{query}」→ 檢索到 {must}", must in _expand_families(dynamic), f"dynamic={dynamic}")
        check(f"「{query}」→ 基底工具不佔名額", not set(dynamic) & _BASE_TOOLS, f"dynamic={dynamic}")

    # 冰箱是基底工具、每輪都在，不靠檢索（「雞蛋用完了」這種短句的動態檢索撈不到它）
    check("inventory 是基底工具", "inventory" in _BASE_TOOLS)

    print("3. 改寫丟掉關鍵字：原話也檢索一次、取聯集")
    for rewritten, latest, must in (
        ("請教我做哪一道菜的步驟", "我在之前的對話裡請你教我做哪一道菜的步驟？", "conversation_search"),
        ("不要有蝦", "我對蝦過敏，以後推薦料理都不要有蝦。", "diet_profile_manage"),
    ):
        raw_dynamic, raw_selected = route(latest)
        _, rewritten_selected = route(rewritten)
        check(f"「{rewritten}」加上原話 → 有 {must}", must in rewritten_selected | raw_selected, f"raw={raw_dynamic}")

    print(f"\n通過 {passed}，失敗 {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
