"""對話 API：只負責「收任務進佇列」與「把 worker 的輸出轉成 SSE」，不執行 agent。

兩階段流程：
1. POST /chat/tasks（或 /chat/tasks/resume）→ 任務進 Redis，立即回傳 task_id / thread_id
2. GET  /chat/tasks/{task_id}/events      → 讀該任務的事件串流，以 SSE 推給前端；
   支援 Last-Event-ID 斷點續傳（瀏覽器 EventSource 重連時會自動帶上）

同一對話連發訊息採 enqueue：前一則還在處理時，新訊息排進該 thread 的等待清單，
前一則結束後自動接著跑。POST /chat/tasks/{task_id}/cancel 可停止執行中的任務或撤回排隊中的訊息。

歷史載入：GET /chat/threads/{thread_id} 回傳「可直接渲染的歷史」+「需要接續/排隊中的任務」
+「待審核的 HITL」（讀自 checkpointer，沒有期限）。

權限：身分一律從 cookie 解出（見 app/api/deps.py），請求本文不收 user_id。
每支碰到 thread 或 task 的路由都先確認擁有者，不是自己的一律回 404。

訪客（沒登入）不留任何紀錄：訊息照樣走佇列由 worker 執行，但 worker 用的是沒有 checkpointer、
沒有長期記憶的訪客 agent（payload.ephemeral = true），輸出只透過 SSE 送出，不寫進 threads 表、
不寫進 checkpointer。每個訪客 session 共用一個只存在 Redis 的佇列鍵 guest_<hex>，
同一個訪客的訊息依序執行（也避免一個人同時佔滿 worker）。每則訊息都是獨立的一問一答。
"""

import hashlib
import json
import os
import uuid

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import Principal, optional_principal, owned_thread, principal_or_new_guest, require_principal
from app.common import task_queue as tq
from app.common.qdrant import delete_thread_turns
from app.common.image_utils import (
    MAX_DATA_URL_IMAGE_BYTES,
    check_message_image_url,
    encode_image_to_base64_url,
    sniff_image_type,
)
from app.models.schemas import ChatRequest, ResumeRequest
from app.services import history
from app.services import threads as thread_repo

router = APIRouter()


def _public_task(task: dict) -> dict:
    """轉成前端需要的任務摘要（不回傳 payload 原文）。"""
    payload = json.loads(task.get("payload") or "{}")
    image_url = payload.get("image_url") or ""
    return {
        "task_id": task["task_id"],
        "kind": task.get("kind"),
        "status": task.get("status"),
        "user_message": payload.get("message"),
        # base64 data URL 可能好幾 MB，只回傳一般網址（GCS 簽名 URL）
        "image_url": image_url if image_url.startswith("http") else None,
    }


async def _owned_task(task_id: str, principal: Principal) -> dict:
    """取得自己送出的任務；不存在、已過期、不是自己的一律 404。

    任務的 user_id 是送出當下由伺服器從 cookie 寫入的（不是前端給的），可以直接比對。
    """
    task = await tq.get_task(tq.get_redis(), task_id)
    if not task or task.get("user_id") != principal.id:
        raise HTTPException(status_code=404, detail="任務不存在或已過期")
    return task


def _guest_queue_key(principal: Principal) -> str:
    """訪客的佇列鍵（只用於 Redis 的排隊與鎖，checkpointer / threads 表都不會出現它）。"""
    return "guest_" + principal.id.removeprefix("guest:")


@router.get("/chat/threads")
async def list_threads(principal: Principal | None = Depends(optional_principal)):
    """側邊欄的對話清單：只列出目前使用者擁有的對話（經過 Row-Level Security 讀取）。訪客沒有紀錄。"""
    if principal is None or principal.is_guest:
        return {"threads": []}
    return {"threads": [thread_repo.public_thread(t) for t in await thread_repo.list_owned(principal.id)]}


def _request_fingerprint(request: ChatRequest) -> str:
    """請求內容的雜湊：圖片可能好幾 MB，只存雜湊，用來發現「同一個 key 卻是不同內容」。"""
    raw = json.dumps([request.message, request.image_url, request.thread_id], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


async def _replay_chat_task(earlier: dict, fingerprint: str, ephemeral: bool) -> dict:
    """同一個冪等 key 又送來一次：回傳原本的任務，不建立新任務。"""
    if earlier.get("fingerprint", fingerprint) != fingerprint:
        raise HTTPException(status_code=422, detail="這個 Idempotency-Key 已用於不同的內容")
    result = await tq.replay_submission(tq.get_redis(), earlier.get("task_id", ""))
    if result is None:  # 第一次的請求還在建立任務（兩個請求同時到），讓前端晚點再試
        raise HTTPException(status_code=409, detail="這則訊息正在送出，請稍後再試")
    return {
        "task_id": earlier["task_id"], "thread_id": earlier["thread_id"], "ephemeral": ephemeral,
        "replayed": True, **result,
    }


@router.post("/chat/tasks", status_code=202)
async def create_chat_task(
    request: ChatRequest,
    principal: Principal = Depends(principal_or_new_guest),
    idempotency_key: str | None = Header(default=None, pattern=r"^[A-Za-z0-9_-]{16,128}$"),
):
    """送出一則訊息。立即回傳 task_id / thread_id，agent 由 worker 非同步執行。

    登入使用者：沒帶 thread_id = 新對話，伺服器產生 thread_id，先寫進 threads 表（記下擁有者）才送進佇列；
    帶了 thread_id 就必須是自己的對話。
    訪客：不建立對話、回傳 thread_id = null，前端只靠 task_id 開 SSE 拿輸出。
    status = pending：直接開始；queued：前面還有任務，排隊等候（held = 等使用者先完成 HITL 審核）。
    image_url 只接受自己產生的 Data URL 或自己 GCS 目錄的簽名 URL（見 check_message_image_url）。

    Idempotency-Key：key 代表「這一次發送」，不是訊息內容。回應在途中遺失時，前端用同一個 key 重送，
    這裡回傳原本的任務（replayed = true）；使用者重新發一則相同的文字是新的 key，照常建立任務。
    key 依使用者分開存，別人用同一個 key 拿不到你的任務。
    """
    if request.image_url:
        error = check_message_image_url(
            request.image_url, None if principal.is_guest else principal.id, os.getenv("GCS_BUCKET")
        )
        if error:
            raise HTTPException(status_code=400, detail=error)
    # queue_id = 排隊用的鍵；thread_id = 回給前端的對話（訪客沒有對話）
    is_new_thread = False
    if principal.is_guest:
        if request.thread_id:
            raise HTTPException(status_code=404, detail="對話不存在")
        queue_id = _guest_queue_key(principal)
    elif request.thread_id:
        if await thread_repo.touch_owned(request.thread_id, principal.id) is None:
            raise HTTPException(status_code=404, detail="對話不存在")
        queue_id = request.thread_id
    else:
        queue_id = f"thread_{uuid.uuid4().hex}"
        is_new_thread = True
    thread_id = None if principal.is_guest else queue_id

    r = tq.get_redis()
    task_id = tq.new_task_id()
    if idempotency_key:
        fingerprint = _request_fingerprint(request)
        earlier = await tq.claim_idempotency(
            r, principal.id, idempotency_key,
            {"task_id": task_id, "thread_id": thread_id, "fingerprint": fingerprint},
        )
        if earlier is not None:
            return await _replay_chat_task(earlier, fingerprint, principal.is_guest)

    payload = {"message": request.message, "image_url": request.image_url}
    if principal.is_guest:
        payload["ephemeral"] = True
    thread_created = False
    try:
        if is_new_thread:
            await thread_repo.create(queue_id, principal.id, thread_repo.make_title(request.message))
            thread_created = True
        result = await tq.submit_task(
            r, task_id=task_id, kind="chat", thread_id=queue_id, user_id=principal.id, payload=payload,
        )
    except Exception:
        # 重送會用新的 thread_id，這裡不清掉，側邊欄就會多一個空對話
        if thread_created:
            await thread_repo.delete(queue_id)
        if idempotency_key:
            await tq.release_idempotency(r, principal.id, idempotency_key)
        raise
    if idempotency_key:
        await tq.confirm_idempotency(r, principal.id, idempotency_key)
    return {"task_id": task_id, "thread_id": thread_id, "ephemeral": principal.is_guest, **result}


@router.post("/chat/tasks/resume", status_code=202)
async def create_resume_task(request: ResumeRequest, principal: Principal = Depends(require_principal)):
    """針對 HITL 中斷送出審核結果（approve / reject），以新任務的形式接續執行。

    resume 也要走佇列：接手的可能是另一台 worker，中斷點狀態從 checkpointer 讀回。
    它會插隊到排隊中的訊息之前（審核本來就該先處理），但不能跟執行中的任務並行。
    """
    await owned_thread(request.thread_id, principal)
    _, pending = await history.get_thread_state(request.thread_id)
    if pending is None:
        return JSONResponse(status_code=409, content={"detail": "這個審核已失效（對話已繼續或已被處理）"})
    if len(request.decisions) != len(pending["action_requests"]):
        raise HTTPException(status_code=422, detail="decisions 數量須與待審核的操作數量一致")
    task_id = tq.new_task_id()
    try:
        await tq.submit_task(
            tq.get_redis(),
            task_id=task_id,
            kind="resume",
            thread_id=request.thread_id,
            user_id=principal.id,
            payload={"decisions": request.decisions},
        )
    except tq.ThreadBusyError as exc:
        return JSONResponse(
            status_code=409,
            content={"detail": "這個對話還有任務在處理中", "task_id": exc.task_id},
        )
    await thread_repo.touch_owned(request.thread_id, principal.id)
    return {"task_id": task_id, "thread_id": request.thread_id, "status": tq.TaskStatus.PENDING}


@router.post("/chat/tasks/{task_id}/cancel", status_code=202)
async def cancel_task(task_id: str, principal: Principal = Depends(require_principal)):
    """停止任務（冪等）。

    - 排隊中 / 等 worker：直接取消，回傳 cancelled
    - 執行中：通知 worker 中止，回傳 cancelling；事件串流稍後以 status(CANCELLED) + done 收尾
    - 已結束：回傳原本的終態，不做任何事
    """
    await _owned_task(task_id, principal)
    status = await tq.cancel_task(tq.get_redis(), task_id)
    if status == "missing":
        raise HTTPException(status_code=404, detail="任務不存在或已過期")
    return {"task_id": task_id, "status": status}


@router.get("/chat/tasks/{task_id}")
async def get_task_status(task_id: str, principal: Principal = Depends(require_principal)):
    task = await _owned_task(task_id, principal)
    task.pop("payload", None)
    task.pop("user_id", None)
    return task


@router.get("/chat/tasks/{task_id}/events")
async def stream_task_events(
    task_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None),
    principal: Principal = Depends(require_principal),
):
    """以 SSE 推送任務事件。每個事件的 id 即 Redis Stream entry id，重連時從斷點續傳。

    EventSource 不能自訂 header，身分靠同源請求自動帶上的 cookie。
    """
    r = tq.get_redis()
    # 回 404 而非空串流：EventSource 收到非 200 會停止重連，不會無限重試
    await _owned_task(task_id, principal)

    start_id = last_event_id or request.query_params.get("last_event_id") or "0-0"

    async def sse():
        yield "retry: 2000\n\n"
        async for item in tq.read_events(r, task_id, last_id=start_id):
            if item is None:
                yield ": keep-alive\n\n"
                continue
            entry_id, event, data = item
            yield f"id: {entry_id}\nevent: {event}\ndata: {data}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/threads/{thread_id}")
async def get_thread(thread_id: str, principal: Principal = Depends(require_principal)):
    """載入對話：回傳可直接渲染的歷史，以及（若有）需要接續的任務。

    - 沒有任務在跑：checkpointer 就是完整紀錄，active_task = null，前端不必開 SSE。
    - 有任務排隊中/執行中：messages 只回傳到任務開始前的界線為止，該任務的輸出
      全部交給事件串流從頭重播——以界線切開，兩邊不會重疊，不需要比對時間戳去重。
    - queued_tasks：排在後面、還沒開始的訊息（held = 等使用者先完成 HITL 審核）。
    - pending_interrupt：checkpointer 裡還沒審核的 HITL，前端重建審核卡片（沒有期限）。

    讀 checkpointer 與讀任務狀態是兩次獨立的讀取：若中間剛好有任務結束、換下一個開始，
    兩邊會對不上（歷史少了剛結束那則回答），所以前後各讀一次任務狀態，不一致就重讀。
    """
    meta = await owned_thread(thread_id, principal)
    r = tq.get_redis()
    for _ in range(3):
        before = await tq.get_thread_tasks(r, thread_id)
        raw, pending_interrupt = await history.get_thread_state(thread_id)
        tasks = await tq.get_thread_tasks(r, thread_id)
        if (before["active"] or {}).get("task_id") == (tasks["active"] or {}).get("task_id"):
            break

    active = tasks["active"]
    if active:
        # 任務還沒被 worker 拉走時尚無界線，這時 checkpointer 也還沒有它的任何輸出，整份歷史都可以渲染
        if active.get("status") == tq.TaskStatus.PROCESSING:
            raw = history.slice_until(raw, active.get("boundary_id") or None)
        # 有任務在跑（例如 resume 或略過審核的新訊息）時，checkpointer 裡的中斷即將被處理掉
        pending_interrupt = None

    return {
        "thread_id": thread_id,
        "title": meta["title"],
        "messages": history.render_messages(raw),
        "active_task": _public_task(active) if active else None,
        "queued_tasks": [_public_task(t) for t in tasks["queued"]],
        "held": tasks["held"] and pending_interrupt is not None,
        "pending_interrupt": pending_interrupt,
    }


@router.delete("/chat/threads/{thread_id}")
async def delete_thread(thread_id: str, principal: Principal = Depends(require_principal)):
    """刪除對話（向量庫的對話紀錄 + 歷史 + 擁有權紀錄）。任務執行中不可刪，否則 worker 會在刪除後繼續寫回 checkpointer。

    對話紀錄最先刪：刪不掉就整個中止，不留下「對話不見了，conversation_search 卻還查得到」的孤兒資料。
    """
    await owned_thread(thread_id, principal)
    r = tq.get_redis()
    if await tq.is_thread_busy(r, thread_id):
        raise HTTPException(status_code=409, detail="這個對話還有任務在處理中，請先停止或稍後再刪除")
    try:
        await delete_thread_turns(thread_id)
    except Exception as exc:
        print(f"⚠️ [DELETE] 刪除對話紀錄失敗 thread={thread_id}：{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="暫時無法刪除對話，請稍後再試")
    await tq.clear_thread(r, thread_id)
    await history.delete_thread(thread_id)
    await thread_repo.delete(thread_id)
    return {"message": "Chat history cleared successfully"}


@router.post("/chat/upload-image")
async def upload_image(file: UploadFile = File(...), _principal: Principal = Depends(principal_or_new_guest)):
    """上傳圖片並轉換為 Base64 Data URL（不落地任何儲存；訪客上傳圖片走這裡，不走 GCS）

    類型看檔頭決定，不看副檔名：改名成 .png 的 HTML / SVG 一律拒絕。
    """
    contents = await file.read(MAX_DATA_URL_IMAGE_BYTES + 1)  # 多讀 1 byte 就知道有沒有超過
    if len(contents) > MAX_DATA_URL_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"圖片大小不可超過 {MAX_DATA_URL_IMAGE_BYTES / 1024 / 1024:g}MB")
    content_type = sniff_image_type(contents[:16])
    if content_type is None:
        raise HTTPException(status_code=415, detail="只接受 jpg / png / gif / webp 圖片")
    base64_url = encode_image_to_base64_url(contents, content_type)

    return {
        "image_url": base64_url,
        "filename": file.filename,
        "size": len(contents),
        "content_type": content_type,
    }
