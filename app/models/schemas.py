from pydantic import BaseModel
from typing import Optional

# --- 2. 資料模型 ---
# 刻意沒有 user_id 欄位：身分只從伺服器簽發的 cookie 解出（app/api/deps.py），
# 讓前端無法透過請求本文冒充別人。
class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None
    # 新對話不帶：由伺服器產生 thread_id 回傳，前端再跳轉到 /chat/{thread_id}
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[dict]
