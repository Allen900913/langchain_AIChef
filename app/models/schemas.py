from pydantic import BaseModel
from typing import Optional

# --- 2. 資料模型 ---
class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None
    thread_id: str
    user_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[dict]
    user_id: Optional[str] = None