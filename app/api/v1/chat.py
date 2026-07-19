from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ResumeRequest
from app.agents.app import call_agent, resume_agent, get_checkpointer, delete_checkpointer
from app.common.image_utils import encode_image_to_base64_url, get_content_type

router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式對話"""
    return StreamingResponse(
        await call_agent(
            request.message,
            request.image_url,
            request.thread_id,
            request.user_id,
        ),
        media_type="text/event-stream"
    )


@router.post("/chat/resume")
async def resume_endpoint(request: ResumeRequest):
    """針對 HITL 中斷，傳入審核結果（approve / reject）以繼續執行"""
    return StreamingResponse(
        resume_agent(
            request.thread_id,
            request.decisions,
            request.user_id,
        ),
        media_type="text/event-stream"
    )


@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """獲取歷史訊息"""
    return await get_checkpointer(thread_id)


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空歷史訊息"""
    await delete_checkpointer(thread_id)
    return {"message": "Chat history cleared successfully"}


@router.post("/chat/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """上傳圖片並轉換為 Base64 Data URL"""
    MAX_SIZE = 10 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        return {"error": "檔案大小不可超過 10MB"}

    content_type = get_content_type(file.filename or "image.png")
    base64_url = encode_image_to_base64_url(contents, content_type)

    return {
        "image_url": base64_url,
        "filename": file.filename,
        "size": len(contents),
        "content_type": content_type,
    }