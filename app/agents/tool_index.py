"""LLM tool router 的檢索後端：把工具 description 存進 Qdrant，用
dense（bge-small-zh）+ BM25 sparse 混合檢索、RRF 融合，選出與當前查詢最相關的工具。

規模說明（務必知道）：這套 Query Rewrite + Hybrid Search + RRF + Qdrant 管線是為
「工具多到塞不進 context」（上百上千個）的規模設計的。本專案只有 14 個工具，
其實把全部工具丟給模型就夠了；這裡完整實作純為練習這條業界標準管線。

管線分工：
- 索引（一次性）：每個工具 = 一個 Qdrant point，payload 存工具名，
  同時存 dense 向量（語意）與 bm25 sparse 向量（字面）。
- 查詢（每輪）：對「重寫後的查詢字串」同時做 dense + bm25 檢索，
  用 Qdrant 原生的 RRF 融合兩份排名，回傳 Top-K 工具名。

BM25 用 Qdrant 的 IDF modifier：fastembed 只產原始詞頻，IDF 由 Qdrant server 端
在檢索時計算（見 create_collection 的 modifier=IDF），符合 Qdrant 官方 hybrid 範式。
"""

import asyncio
import os
import sys
from functools import lru_cache

from dotenv import load_dotenv

# 以 -m 獨立執行時不會經過 app.py 的 stdout reconfigure，cp950 終端機印 emoji 會炸。
# 同樣在此保護，確保 ensure_index 的 print 不中斷索引流程。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and getattr(_stream, "encoding", None) != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

# tools.py 匯入時會實例化 TavilySearch（需 TAVILY_API_KEY）；直接以 -m 執行本模組
# 時不會經過 app.py 的 load_dotenv，故在匯入 tools 前先載入 .env（重複呼叫無害）。
load_dotenv()

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding

from app.agents.tools import ALL_TOOLS

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "chef_tools"

# 中文專用、512 維、僅 90MB：工具 description 全是中文，用中文模型召回優於多語通用模型。
DENSE_MODEL = "BAAI/bge-small-zh-v1.5"
DENSE_DIM = 512
SPARSE_MODEL = "Qdrant/bm25"

DENSE_VEC = "dense"
SPARSE_VEC = "bm25"


# fastembed 模型單例（首次載入含下載，之後走本地快取）。onnxruntime CPU 推論，
# 單條短字串約數十毫秒；查詢路徑會用 asyncio.to_thread 包起來避免阻塞事件迴圈。
@lru_cache(maxsize=1)
def _dense_model() -> TextEmbedding:
    return TextEmbedding(DENSE_MODEL)


@lru_cache(maxsize=1)
def _sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(SPARSE_MODEL)


def _tool_text(tool) -> str:
    """要 embed 的文字：工具名 + description（description 即 @tool 的 docstring）。
    兩者都放進去，dense 抓語意、bm25 抓字面關鍵字（含工具名本身）。"""
    desc = (tool.description or "").strip()
    return f"{tool.name}：{desc}"


def _to_sparse_vector(sparse_embedding) -> models.SparseVector:
    """fastembed 的 SparseEmbedding → Qdrant SparseVector。"""
    return models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )


# ==============================================================================
# 索引建立（一次性 / 冪等）
# ==============================================================================

def ensure_index(force: bool = False) -> None:
    """建立或補齊 chef_tools collection（同步，供啟動時與 __main__ 呼叫）。

    冪等：collection 已存在且點數等於工具數就直接返回（不重建）；force=True 強制重建。
    用同步 QdrantClient——這是啟動階段的一次性動作，不需要非同步，也避開事件迴圈細節。
    """
    client = QdrantClient(url=QDRANT_URL)
    try:
        exists = client.collection_exists(COLLECTION)
        if exists and not force:
            count = client.count(COLLECTION).count
            if count == len(ALL_TOOLS):
                print(f"🧭 [TOOL-INDEX] {COLLECTION} 已就緒（{count} 個工具），跳過重建")
                return
            print(f"🧭 [TOOL-INDEX] 點數不符（{count} != {len(ALL_TOOLS)}），重建")

        if exists:
            client.delete_collection(COLLECTION)

        client.create_collection(
            COLLECTION,
            vectors_config={
                DENSE_VEC: models.VectorParams(
                    size=DENSE_DIM, distance=models.Distance.COSINE
                ),
            },
            # IDF modifier：bm25 的 IDF 權重由 Qdrant 檢索時計算（fastembed 只給詞頻）
            sparse_vectors_config={
                SPARSE_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

        texts = [_tool_text(t) for t in ALL_TOOLS]
        dense_vecs = list(_dense_model().passage_embed(texts))
        sparse_vecs = list(_sparse_model().embed(texts))

        points = [
            models.PointStruct(
                id=i,
                vector={
                    DENSE_VEC: dense_vecs[i].tolist(),
                    SPARSE_VEC: _to_sparse_vector(sparse_vecs[i]),
                },
                payload={"name": tool.name},
            )
            for i, tool in enumerate(ALL_TOOLS)
        ]
        client.upsert(COLLECTION, points=points)
        print(f"🧭 [TOOL-INDEX] 已索引 {len(points)} 個工具進 {COLLECTION}")
    finally:
        client.close()


# ==============================================================================
# 查詢（每輪 / 非同步）
# ==============================================================================

_async_client: AsyncQdrantClient | None = None


def _get_async_client() -> AsyncQdrantClient:
    """延遲建立 AsyncQdrantClient 單例（須在事件迴圈內首次呼叫）。"""
    global _async_client
    if _async_client is None:
        _async_client = AsyncQdrantClient(url=QDRANT_URL)
    return _async_client


def _embed_query_sync(query: str) -> tuple[list[float], models.SparseVector]:
    """對查詢字串同時產 dense + bm25 sparse 向量（用 query 端方法，非對稱檢索）。
    bge 的 query_embed 會自動加上查詢指令前綴；bm25 的 query_embed 產查詢側詞權重。"""
    dense = list(_dense_model().query_embed(query))[0].tolist()
    sparse = _to_sparse_vector(list(_sparse_model().query_embed(query))[0])
    return dense, sparse


async def retrieve_tool_names(query: str, limit: int = 5) -> list[str]:
    """Hybrid Search + RRF：回傳與 query 最相關的 Top-K 工具名（依融合分數高到低）。

    失敗時回傳空 list——由 router 用基底工具兜底，不讓檢索故障中斷整輪對話。
    """
    query = (query or "").strip()
    if not query:
        return []
    try:
        # fastembed 是 CPU 同步推論，丟進執行緒避免阻塞事件迴圈
        dense_q, sparse_q = await asyncio.to_thread(_embed_query_sync, query)
        client = _get_async_client()
        result = await client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_q, using=DENSE_VEC, limit=limit * 2),
                models.Prefetch(query=sparse_q, using=SPARSE_VEC, limit=limit * 2),
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
