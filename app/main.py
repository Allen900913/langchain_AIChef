import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.common.logger import setup_logging
from app.api.v1 import auth as auth_api
from app.api.v1 import chat
from app.api.v1 import oss
from app.common import task_queue
from app.services import auth, db, history

# API 伺服器刻意不 import agent（app/agents/*）：模型、fastembed、E2B 等重依賴只存在
# worker（app/worker.py）。API 只負責收任務進佇列、把 worker 輸出轉成 SSE、讀歷史。

# 初始化日誌配置
setup_logging()


# Windows 上 uvicorn 預設的 asyncio loop factory 會直接回傳 ProactorEventLoop，
# 但 psycopg 的 async 模式不支援它，必須改用 SelectorEventLoop。
# 啟動時加上 --loop app.main:selector_loop_factory 套用此設定。
def selector_loop_factory(use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop()


# 過期 refresh token 的清理週期。多個 API 行程時以 advisory lock 確保同一時間只有一個在清
_CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))
_CLEANUP_LOCK_ID = 7_412_002


async def _cleanup_loop() -> None:
    while True:
        try:
            async with db.get_pool().connection() as conn:
                cur = await conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (_CLEANUP_LOCK_ID,))
                if (await cur.fetchone())["ok"]:
                    try:
                        tokens = await auth.purge_expired_refresh_tokens()
                        if tokens:
                            print(f"🧹 [CLEANUP] 刪除過期 refresh token {tokens} 筆")
                    finally:
                        await conn.execute("SELECT pg_advisory_unlock(%s)", (_CLEANUP_LOCK_ID,))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 清理失敗不影響服務，下一輪再試
            print(f"⚠️ [CLEANUP] 清理失敗：{exc!r}")
        await asyncio.sleep(_CLEANUP_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth.check_config()  # JWT_SECRET 沒設定就直接啟動失敗
    await db.init_pool()
    await db.migrate()
    await history.init_history()
    await task_queue.get_redis().ping()  # 啟動時就確認 Redis 可達，而不是等第一個請求才炸
    cleanup = asyncio.create_task(_cleanup_loop())
    yield
    cleanup.cancel()
    await task_queue.close_redis()
    await history.close_history()
    await db.close_pool()


app = FastAPI(
    title="Personal Chief API",
    description="私廚",
    version="0.1.0",
    lifespan=lifespan,
)

# 1. CORS：正式環境前端由 nginx 同源反向代理 /api，不需要跨域；這裡只放行開發用的
#    Vite dev server（npm run dev，預設 http://localhost:5173）。以逗號分隔設定 CORS_ORIGINS。
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def reject_cross_site_writes(request: Request, call_next):
    """CSRF 防護：身分放在 cookie，瀏覽器會自動帶上，所以要擋「別的網站代替使用者送出的寫入請求」。

    cookie 已設 SameSite=Lax（跨站 POST 不帶 cookie），這裡再檢查 Origin 當第二層：
    瀏覽器送出跨站寫入時一定帶 Origin，主機名稱不是自己、也不在 CORS 白名單就拒絕。
    沒有 Origin（curl、伺服器對伺服器）不在 CSRF 的威脅範圍內，放行。
    nginx 轉送的 Host 不含 port，所以只比主機名稱。
    """
    origin = request.headers.get("origin")
    if request.method in _UNSAFE_METHODS and origin and origin not in _CORS_ORIGINS:
        host = urlsplit("//" + (request.headers.get("host") or "")).hostname
        if urlsplit(origin).hostname != host:
            return JSONResponse(status_code=403, content={"detail": "不允許跨站請求"})
    return await call_next(request)

# 2. 掛載路由（前端是獨立的 React 專案 frontend/，不再由 FastAPI 提供靜態檔）
app.include_router(auth_api.router, prefix="/api/v1", tags=["認證"])
app.include_router(chat.router, prefix="/api/v1", tags=["對話"])
app.include_router(oss.router, prefix="/api/v1", tags=["GCS 預簽名上傳"])


@app.get("/api/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8741, reload=True)