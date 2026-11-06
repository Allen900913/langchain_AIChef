"""LLM tool router 的檢索後端：把工具 description 存進 Qdrant，用
dense（bge-small-zh）+ BM25 sparse 混合檢索、RRF 融合，選出與當前查詢最相關的工具。

規模說明（務必知道）：這套 Query Rewrite + Hybrid Search + RRF + Qdrant 管線是為
「工具多到塞不進 context」（上百上千個）的規模設計的。本專案只有 12 個工具，
其實把全部工具丟給模型就夠了；這裡完整實作純為練習這條業界標準管線。

管線分工：
- 索引（一次性）：每個工具 = 一個 Qdrant point，payload 存工具名與被 embed 的文字，
  同時存 dense 向量（語意）與 bm25 sparse 向量（字面）。
- 查詢（每輪）：對「重寫後的查詢字串」同時做 dense + bm25 檢索，
  用 Qdrant 原生的 RRF 融合兩份排名，回傳 Top-K 工具名。

BM25 用 Qdrant 的 IDF modifier：fastembed 只產原始詞頻，IDF 由 Qdrant server 端
在檢索時計算（見 create_collection 的 modifier=IDF），符合 Qdrant 官方 hybrid 範式。
"""

import asyncio
import sys

# 以 -m 獨立執行時不會經過 app.py 的 stdout reconfigure，cp950 終端機印 emoji 會炸。
# 同樣在此保護，確保 ensure_index 的 print 不中斷索引流程。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and getattr(_stream, "encoding", None) != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

from qdrant_client import QdrantClient, models

# app.common.qdrant 在 import 時載入 .env 並讀 QDRANT_URL / QDRANT_API_KEY（直接以 -m 執行也適用）
from app.common.qdrant import QDRANT_API_KEY, QDRANT_URL, get_async_client
from app.agents.embedding import (
    DENSE_VEC,
    SPARSE_VEC,
    embed_passages_sync,
    embed_query_sync,
    sparse_vectors_config,
    vectors_config,
)
from app.agents.tools import ALL_TOOLS

COLLECTION = "chef_tools"


def _tool_text(tool) -> str:
    """要 embed 的文字：工具名 + description（description 即 @tool 的 docstring）。
    兩者都放進去，dense 抓語意、bm25 抓字面關鍵字（含工具名本身）。"""
    desc = (tool.description or "").strip()
    return f"{tool.name}：{desc}"


# ==============================================================================
# 索引建立（一次性 / 冪等）
# ==============================================================================

def ensure_index(force: bool = False) -> None:
    """建立或補齊 chef_tools collection（同步，供啟動時與 __main__ 呼叫）。

    冪等：collection 已存在、而且裡面每個工具的名稱與 description 都跟程式一致就直接返回（不重建）；
    工具增減、改名或只改了 description 都會重建。force=True 強制重建。
    用同步 QdrantClient——這是啟動階段的一次性動作，不需要非同步，也避開事件迴圈細節。
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        exists = client.collection_exists(COLLECTION)
        if exists and not force:
            points, _ = client.scroll(COLLECTION, limit=len(ALL_TOOLS) + 100, with_payload=True)
            indexed = sorted((p.payload.get("name"), p.payload.get("text")) for p in points)
            if indexed == sorted((t.name, _tool_text(t)) for t in ALL_TOOLS):
                print(f"🧭 [TOOL-INDEX] {COLLECTION} 已就緒（{len(points)} 個工具），跳過重建")
                return
            print(f"🧭 [TOOL-INDEX] 工具名稱或 description 有變（索引 {len(points)} 個、程式 {len(ALL_TOOLS)} 個），重建")

        if exists:
            client.delete_collection(COLLECTION)

        client.create_collection(
            COLLECTION,
            vectors_config=vectors_config(),
            sparse_vectors_config=sparse_vectors_config(),
        )

        vectors = embed_passages_sync([_tool_text(t) for t in ALL_TOOLS])
        points = [
            models.PointStruct(id=i, vector=vectors[i], payload={"name": tool.name, "text": _tool_text(tool)})
            for i, tool in enumerate(ALL_TOOLS)
        ]
        client.upsert(COLLECTION, points=points)
        print(f"🧭 [TOOL-INDEX] 已索引 {len(points)} 個工具進 {COLLECTION}")
    finally:
        client.close()


# ==============================================================================
# 查詢（每輪 / 非同步）
# ==============================================================================

def exclude_filter(names) -> models.Filter | None:
    """排除指定工具的 Qdrant filter（基底工具每輪都在，不該佔檢索名額）。"""
    if not names:
        return None
    return models.Filter(must_not=[models.FieldCondition(key="name", match=models.MatchAny(any=list(names)))])


async def retrieve_tool_names(query: str, limit: int = 5, exclude=()) -> list[str]:
    """Hybrid Search + RRF：回傳與 query 最相關的 Top-K 工具名（依融合分數高到低），不含 exclude 裡的工具。

    失敗時回傳空 list——由 router 用基底工具兜底，不讓檢索故障中斷整輪對話。
    """
    query = (query or "").strip()
    if not query:
        return []
    try:
        # fastembed 是 CPU 同步推論，丟進執行緒避免阻塞事件迴圈
        dense_q, sparse_q = await asyncio.to_thread(embed_query_sync, query)
        client = get_async_client()
        flt = exclude_filter(exclude)
        result = await client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_q, using=DENSE_VEC, limit=limit * 2, filter=flt),
                models.Prefetch(query=sparse_q, using=SPARSE_VEC, limit=limit * 2, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [p.payload["name"] for p in result.points if p.payload]
    except Exception as exc:
        print(f"⚠️ [TOOL-INDEX] 檢索失敗，改用基底工具兜底：{type(exc).__name__}: {exc}")
        return []


if __name__ == "__main__":
    # 直接執行 = 強制重建索引：python -m app.agents.tool_index
    ensure_index(force=True)
