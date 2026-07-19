import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from app.common.logger import setup_logging
# 假設 chat 與 oss 路由模組位於以下路徑
from app.api.v1 import chat
from app.api.v1 import oss
from app.agents.app import init_agent_infra

# 註：cp950 → UTF-8 的 stdout/stderr reconfigure 已下沉到 app/agents/app.py 的
# import 階段（emoji print 來源處），上面 import init_agent_infra 時即已生效。

# 初始化日誌配置
setup_logging()


# Windows 上 uvicorn 預設的 asyncio loop factory 會直接回傳 ProactorEventLoop，
# 但 psycopg 的 async 模式不支援它，必須改用 SelectorEventLoop。
# 啟動時加上 --loop app.main:selector_loop_factory 套用此設定。
def selector_loop_factory(use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_agent_infra()
    yield


app = FastAPI(
    title="Personal Chief API",
    description="私廚",
    version="0.1.0",
    lifespan=lifespan,
)

# 1. 配置跨域資源共享 (CORS)
# 外掛開發中，由於請求來自瀏覽器擴充套件環境，必須正確配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境建議指定外掛的 ID 或具體域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 掛載路由
# /api/v1
app.include_router(chat.router, prefix="/api/v1", tags=["對話"])
# /api/v1
app.include_router(oss.router, prefix="/api/v1", tags=["GCS 預簽名上傳"])

# 3. 掛載前端資源
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# 前端 fallback 路由 - 只處理非 API 請求
# /{path:path}
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str):
    # 排除 API 路徑
    if path.startswith("api/"):
        return JSONResponse(content={"error": "Not Found"}, status_code=404)
    
    # 如果請求的是靜態檔案，直接返回
    file_path = os.path.join(static_dir, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # 否則返回 index.html (SPA fallback)
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {"message": "你的獨傢俬廚上線了~", "status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001 , reload=True)