"""dense（bge-small-zh）+ BM25 sparse 的 embedding，tool router 的工具索引與對話記憶共用。

模型在本機用 onnxruntime CPU 推論，不呼叫任何 API、不吃模型額度。首次載入含下載，之後走本地快取；
單條短字串約數十毫秒，非同步路徑一律用 asyncio.to_thread 包起來避免阻塞事件迴圈。
"""

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

# 中文專用、512 維、僅 90MB：工具 description 與對話全是中文，用中文模型召回優於多語通用模型。
DENSE_MODEL = "BAAI/bge-small-zh-v1.5"
DENSE_DIM = 512
SPARSE_MODEL = "Qdrant/bm25"

DENSE_VEC = "dense"
SPARSE_VEC = "bm25"


@lru_cache(maxsize=1)
def dense_model() -> TextEmbedding:
    return TextEmbedding(DENSE_MODEL)


@lru_cache(maxsize=1)
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(SPARSE_MODEL)


def to_sparse_vector(sparse_embedding) -> models.SparseVector:
    """fastembed 的 SparseEmbedding → Qdrant SparseVector。"""
    return models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )


def vectors_config() -> dict[str, models.VectorParams]:
    return {DENSE_VEC: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)}


def sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    # IDF modifier：bm25 的 IDF 權重由 Qdrant 檢索時計算（fastembed 只給詞頻）
    return {SPARSE_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF)}


def embed_passages_sync(texts: list[str]) -> list[dict]:
    """要存進索引的文字 → 每筆一個 {dense, bm25} 向量（用 passage 端方法）。"""
    dense = list(dense_model().passage_embed(texts))
    sparse = list(sparse_model().embed(texts))
    return [
        {DENSE_VEC: d.tolist(), SPARSE_VEC: to_sparse_vector(s)}
        for d, s in zip(dense, sparse)
    ]


def embed_query_sync(query: str) -> tuple[list[float], models.SparseVector]:
    """對查詢字串同時產 dense + bm25 sparse 向量（用 query 端方法，非對稱檢索）。
    bge 的 query_embed 會自動加上查詢指令前綴；bm25 的 query_embed 產查詢側詞權重。"""
    dense = list(dense_model().query_embed(query))[0].tolist()
    sparse = to_sparse_vector(list(sparse_model().query_embed(query))[0])
    return dense, sparse
