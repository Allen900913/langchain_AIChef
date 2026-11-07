from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated, Literal, Optional

# 單則訊息上限，約 2,100 tokens（中文約 1 字 1 token）。Groq 每分鐘上限 8,000 tokens，
# 一次主模型呼叫本身就要 2,000–6,000，訊息再長就可能一個請求吃光整分鐘的額度。
# 前端 Composer.tsx 的 MESSAGE_MAX_CHARS 要跟這裡一致。
MESSAGE_MAX_CHARS = 2000
# 伺服器產生的 thread_<32 hex> 是 39 字，留一點餘裕
THREAD_ID_MAX_CHARS = 64
# 一次 HITL 中斷的操作數遠小於這個數字，只用來擋異常的請求
DECISIONS_MAX = 50
DECISION_MESSAGE_MAX_CHARS = 500


# --- 2. 資料模型 ---
# 刻意沒有 user_id 欄位：身分只從伺服器簽發的 cookie 解出（app/api/deps.py），
# 讓前端無法透過請求本文冒充別人。
class ChatRequest(BaseModel):
    # 長度以去掉前後空白後為準：只有空白的訊息等於空訊息
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MESSAGE_MAX_CHARS)]
    # 圖片網址的長度由 check_message_image_url 檢查（Data URL 與簽名 URL 的上限不同）
    image_url: Optional[str] = None
    # 新對話不帶：由伺服器產生 thread_id 回傳，前端再跳轉到 /chat/{thread_id}
    thread_id: Optional[str] = Field(default=None, max_length=THREAD_ID_MAX_CHARS)


class Decision(BaseModel):
    type: Literal["approve", "reject"]
    message: Optional[str] = Field(default=None, max_length=DECISION_MESSAGE_MAX_CHARS)


class ResumeRequest(BaseModel):
    thread_id: str = Field(max_length=THREAD_ID_MAX_CHARS)
    decisions: list[Decision] = Field(max_length=DECISIONS_MAX)
