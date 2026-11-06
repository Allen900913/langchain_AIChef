"""Qdrant 連線設定與對話紀錄的刪除：API 伺服器與 agent worker 共用。

刻意只依賴 qdrant_client：API 刪除對話時也要刪掉向量庫裡的對話紀錄，但不該為此載入
fastembed 模型（embedding 只在 worker 做，見 app/agents/conversation_memory.py）。
"""

import os
import warnings

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models

# 直接以 -m 執行 tool_index 時不會經過 app.py 的 load_dotenv，先載入 .env（重複呼叫無害）
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6397")
# Qdrant 開了 API key 驗證（docker-compose 的 QDRANT__SERVICE__API_KEY），每個請求都要帶
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# 走 compose 內部網路 / 本機 loopback 的 http，用 API key 會觸發這個警告；連線本來就不出這台機器
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.")

# 每輪對話一筆（使用者訊息 + 助理最終回答），payload 帶 user_id / thread_id / created_at
CONVERSATION_COLLECTION = "conversation_turns"

_async_client: AsyncQdrantClient | None = None


def get_async_client() -> AsyncQdrantClient:
    """延遲建立 AsyncQdrantClient 單例（須在事件迴圈內首次呼叫）。"""
    global _async_client
    if _async_client is None:
        _async_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _async_client


async def delete_thread_turns(thread_id: str) -> None:
    """刪掉這個對話存進向量庫的所有紀錄。collection 還沒建立（從沒存過任何一輪）視同已刪除。"""
    client = get_async_client()
    if not await client.collection_exists(CONVERSATION_COLLECTION):
        return
    await client.delete(
        CONVERSATION_COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="thread_id", match=models.MatchValue(value=thread_id)),
        ])),
        wait=True,
    )
