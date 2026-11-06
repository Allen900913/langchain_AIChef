# AI Chef（私廚 AI 助理）

FastAPI + LangGraph 打造的私廚助理：依冰箱庫存、飲食限制與廚房條件推薦料理，支援圖片辨識食材、
網路搜尋食譜、逐步做菜引導，並記住使用者的長期偏好。過敏原由程式強制檢查，會改動長期資料的操作要使用者審核。

本文件說明怎麼跑起來；架構與設計見 [ARCHITECTURE.md](ARCHITECTURE.md)。

```
瀏覽器 ─HTTPS→ proxy（Caddy）→ web（nginx + React）→ api（FastAPI）──Redis 佇列──→ worker（agent）
                                                         │                         │
                                                         └──── Postgres ───────────┤
                                                                                    ├─ Qdrant（工具索引）
                                                    外部服務：Groq（模型）、E2B（沙箱）、Tavily（搜尋）、
                                                              Langfuse（trace / prompt）、GCS（圖片）
```

## 快速開始（Docker）

1. **先決條件**：Python 3.12（見 `.python-version`）、[uv](https://docs.astral.sh/uv/)、Docker Desktop、Node 22（只有本機開發前端才需要）。
2. **手動帶過去的檔案**（都在 `.gitignore` 裡，git clone 不會有）：
   - `.env`（變數見下一節）
   - GCS service account 金鑰 JSON（`GOOGLE_APPLICATION_CREDENTIALS` 指向它，例如 `app/rugged-choir-*.json`）
3. **裝套件**（本機跑測試 / 評測用；容器會自己裝）：
   ```bash
   uv sync                 # 主程式
   uv sync --group eval    # 要跑 evals/ 才需要（ragas 等）
   ```
4. **起全部服務**（postgres / qdrant / redis / api / worker / web / proxy）：
   ```bash
   docker compose up -d --build
   ```
   開 `https://localhost`。路徑是 Caddy（HTTPS）→ web（nginx，HTTP）→ api（HTTP）；
   `http://` 會自動轉址到 `https://`。api 的 8741、web 的 8080 只綁 127.0.0.1，外部連不到。
   worker 可水平擴充：`docker compose up -d --scale worker=3`。

   **本機憑證**：`SITE_ADDRESS=localhost` 時憑證由 Caddy 自己的 CA 簽發，瀏覽器第一次會警告。
   把根憑證加進 Windows「目前使用者」的信任清單即可（PowerShell，只需做一次；刪掉 `caddy_data` volume 會換一張新的 CA）：
   ```powershell
   docker compose cp proxy:/data/caddy/pki/authorities/local/root.crt "$env:TEMP\caddy-root.crt"
   certutil -user -addstore Root "$env:TEMP\caddy-root.crt"
   ```
   **正式網域**：`.env` 設 `SITE_ADDRESS=chef.example.com`，DNS 指到這台機器、防火牆開 80 / 443，
   `docker compose up -d proxy` 後 Caddy 會自動向 Let's Encrypt 申請，到期前自動續期。

   改了 `Caddyfile` 之後重新載入（Git Bash 要加 `MSYS_NO_PATHCONV=1`，否則容器內路徑會被改寫）：
   ```bash
   MSYS_NO_PATHCONV=1 docker compose exec -T proxy caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
   ```
5. **Langfuse**（選用）：另外用官方 repo 部署，見下方「Langfuse」。

## 環境變數

`.env` 會整份傳進 api 與 worker 容器（compose 的 `env_file`），下表以外的變數不要放。
容器內的 `DB_URI`、`REDIS_URL`、`QDRANT_URL`、`LANGFUSE_BASE_URL`、`GOOGLE_APPLICATION_CREDENTIALS`
由 `docker-compose.yml` 改寫成容器內的位址，`.env` 裡填的是本機直接執行用的值。

| 變數 | 誰在用 | 必要性 |
|------|--------|--------|
| `POSTGRES_PASSWORD` | `docker-compose.yml` 的 postgres | 必要，沒設 compose 拒絕啟動。只在資料目錄第一次初始化時生效，既有資料要換密碼見下方 |
| `DB_URI` | API、worker | 必要。本機：`postgresql://postgres:<POSTGRES_PASSWORD>@localhost:5487/ai_memory` |
| `REDIS_PASSWORD` | Redis `requirepass` | 必要，沒設 compose 拒絕啟動 |
| `REDIS_URL` | `common/task_queue.py` | 必要。本機：`redis://:<REDIS_PASSWORD>@localhost:6389/0`（程式預設值不帶密碼，連不上） |
| `QDRANT_API_KEY` | Qdrant `QDRANT__SERVICE__API_KEY`；`agents/tool_index.py` | 必要，沒設 compose 拒絕啟動 |
| `QDRANT_URL` | `agents/tool_index.py` | 選用，預設 `http://localhost:6397` |
| `GROQ_API_KEY` | 主模型 `gpt-oss-120b` | 必要 |
| `GROQ_API_KEY2` | `gpt-oss-20b`：摘要、過敏原判定、反思、Prompt Guard、Query Rewrite、搜尋蒸餾與注入偵測、線上評分 | 必要。**評測不可以用這把**（見「評測」） |
| `GROQ_API_KEY3` | 視覺模型 `qwen3.8-27b` | 必要；評測預設也用它 |
| `E2B_API_KEY` | E2B SDK（`web_search`、`run_python_in_sandbox`） | 必要 |
| `TAVILY_API_KEY` | `web_search` | 必要 |
| `JWT_SECRET` | `services/auth.py` 簽 JWT | 必要，至少 32 字元的隨機字串；沒設 API 會啟動失敗。換掉會讓所有人重新登入 |
| `GCS_BUCKET`、`GOOGLE_APPLICATION_CREDENTIALS` | `api/v1/oss.py`、`/chat/tasks` 的附圖網址檢查 | 登入使用者上傳圖片需要；compose 會把金鑰檔唯讀掛進 api 容器 |
| `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL` | trace + `agents/prompt_manager.py` | 選用，沒設就不送 trace、prompt 用本地預設值 |
| `COOKIE_SECURE` | `api/deps.py` | 選用，預設 `true`。用區網 IP 以 http 測試時設 `false`（`http://localhost` 不需要） |
| `CORS_ORIGINS` | `main.py` | 選用，預設 `http://localhost:5173`（Vite 開發伺服器），逗號分隔。寫入請求的 Origin 檢查也信任這份清單 |
| `SITE_ADDRESS` | `docker-compose.yml` → `Caddyfile` | 選用，預設 `localhost`。填正式網域就自動向 Let's Encrypt 申請憑證 |
| `ACCESS_TOKEN_MINUTES`、`REFRESH_TOKEN_DAYS`、`GUEST_TTL_HOURS` | token 期限 | 選用，預設 15 分鐘 / 30 天 / 24 小時 |
| `CLEANUP_INTERVAL_SECONDS` | `main.py` 過期 refresh token 清理週期 | 選用，預設 3600 |
| `WORKER_CONCURRENCY` | `worker.py` 每台同時處理的任務數 | 選用，預設 4 |
| `NVIDIA_API_KEY` | `evals/run_eval.py` 的 judge 模型 | 只有跑 `run_eval` 才需要 |

### 產生與更換密碼

產生 `JWT_SECRET`：

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
```

產生 `POSTGRES_PASSWORD`、`REDIS_PASSWORD`、`QDRANT_API_KEY`（用 hex，會放進連線網址，不能有 `@ : /` 這類字元）：

```bash
uv run python -c "import secrets; print(secrets.token_hex(32))"
```

- **Redis / Qdrant**：改 `.env` 後 `docker compose up -d`，容器會以新設定重建。本機用的 `REDIS_URL` 記得一起改。
- **Postgres**：`POSTGRES_PASSWORD` 只在第一次初始化生效，既有的 volume 要在資料庫裡改：
  ```bash
  docker compose exec postgres psql -U postgres -c "ALTER USER postgres WITH PASSWORD '<新密碼>'"
  # 再更新 .env 的 POSTGRES_PASSWORD 與 DB_URI，然後 docker compose up -d
  ```

## 本機開發（不用容器跑 api / worker / web）

資料服務（postgres / qdrant / redis）照樣用 compose 起，其餘直接在本機跑：

```bash
docker compose up -d postgres qdrant redis
uv run uvicorn app.main:app --reload --port 8741 --loop app.main:selector_loop_factory
uv run python -m app.worker
cd frontend && npm install && npm run dev      # http://localhost:5173，/api 代理到 8741
```

## 測試

離線測試（不呼叫任何模型 API）：

```bash
uv run python app/tests/test_allergen_fallback.py   # 過敏原判定器失敗時的關鍵字備案、工具呼叫訊息的文字判定
uv run python app/tests/test_memory_guard.py        # 背景反思寫入偏好時的保護
uv run python app/tests/test_prompt_isolation.py    # 標籤跳脫、偏好長度限制與讀取端清洗
uv run python app/tests/test_upload_validation.py   # 圖片檔頭、附圖網址白名單、GCS 簽名（加 --gcs 會連真的 GCS）
```

整合測試（需要 compose 的 postgres / redis 已啟動；`integration_auth` 另外需要 worker 至少啟動過一次建好 store 資料表）：

```bash
PYTHONPATH=. .venv/Scripts/python.exe app/tests/integration_auth.py         # 登入、資料隔離、RLS（會清空 Redis DB 14）
PYTHONPATH=. .venv/Scripts/python.exe app/tests/integration_task_queue.py   # 佇列、排隊、停止、崩潰回收、訪客（會清空 Redis DB 15）
```

真瀏覽器 UI 走查（會實際呼叫 Groq）：先以 `chrome.exe --remote-debugging-port=9333 --user-data-dir=<空資料夾> http://localhost:8080`
開好 Chrome，再 `cd frontend && npm run e2e`。

## 評測（evals/）

Groq 的每日 token 額度是**每把 key、每個模型分開計算**（滾動 24 小時，20b 每把 200k）。`GROQ_API_KEY2` 的 20b 額度
撐著正式環境的摘要、過敏原判定、反思與 guard，被評測用完時：過敏原判定退回關鍵字比對、guard 直接放行。
所以評測一律不用 `GROQ_API_KEY2`，跑之前先估算用量（一次反思約 2.5k token）。

```bash
# 反思（偏好萃取）對照：20b 用哪把 key 由 --key20b 指定（預設 GROQ_API_KEY3）
uv run python -m evals.reflection_eval --reps 2 --variants A,B --key20b GROQ_API_KEY3

# 紅隊測試：主模型 120b 預設 GROQ_API_KEY3、20b 預設 GROQ_API_KEY；指定 GROQ_API_KEY2 會直接拒絕執行
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u -m evals.redteam_eval --cases D3b,M2b --reps 3 --pause 40
#   --skip-guard 跳過 Prompt Guard 探測、--real-image 用真的視覺模型；結果寫到 evals/results/redteam_*.json

# Ragas + 自訂指標（需要 uv sync --group eval 與 NVIDIA_API_KEY）
uv run python -m evals.run_eval --limit 5
```

⚠️ `run_eval` 沒有換 key 的參數，會直接用正式環境的 `GROQ_API_KEY` / `GROQ_API_KEY2`，跑之前先確認額度。

## Langfuse

**不要**把 langfuse 的服務塞進本專案的 `docker-compose.yml`。langfuse 自己那套（web、worker、
postgres、clickhouse、redis、minio）版本相依很緊，直接用官方 repo 起：

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

起來後開 `http://localhost:3000` 建帳號 → 建 project → 拿 public/secret key 填回本專案 `.env` 的
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`，`LANGFUSE_BASE_URL` 設 `http://localhost:3000`
（worker 容器內會改成 `http://host.docker.internal:3000`）。

Prompt 管理是選用的：`agents/prompt_manager.py` 會去 langfuse 抓 7 個 prompt，抓不到（沒設 key／服務沒起／
prompt 還沒建）就自動用檔案裡的 `DEFAULT_*`，不會讓 app 掛掉。要把本地預設值建到 Langfuse：

```bash
uv run python create_langfuse_prompts.py
```

本專案 `docker-compose.yml` 底部的 `langfuse_*` volume 宣告沒有服務在用，留著不影響運作。

## 資料與備份

| 資料 | 存在哪裡 | 換機器 / 重建時 |
|------|----------|------------------|
| 帳號、對話、偏好、冰箱 | Postgres（volume `postgres_data`） | 不會跟著 git 走，要自己備份與還原 |
| 任務佇列、事件串流 | Redis（volume `redis_data`，AOF） | 暫存資料，不用搬 |
| 工具索引 | Qdrant（volume `qdrant_data`） | worker 啟動時自動建好，不用搬 |

目前沒有自動備份。手動備份與還原（在容器內產生檔案再複製出來，避免 PowerShell 的 `>` 把二進位輸出轉碼弄壞）：

```bash
docker compose exec postgres pg_dump -U postgres -Fc -f /tmp/ai_memory.dump ai_memory
docker compose cp postgres:/tmp/ai_memory.dump ./ai_memory.dump

docker compose cp ./ai_memory.dump postgres:/tmp/ai_memory.dump
docker compose exec postgres pg_restore -U postgres -d ai_memory --clean --if-exists /tmp/ai_memory.dump
```

在 Git Bash 執行時，帶容器內路徑的指令前面加 `MSYS_NO_PATHCONV=1`。

## 疑難排解（Windows）

- **psycopg async 不支援 `ProactorEventLoop`**：本機跑 uvicorn 要加 `--loop app.main:selector_loop_factory`；
  `python -m app.worker` 已自己改用 SelectorEventLoop。
- **cp950 終端機無法輸出 emoji**：`agents/app.py` 在 import 階段把 stdout / stderr 改成 UTF-8；
  其他腳本請設 `PYTHONIOENCODING=utf-8`。
- **輸出導向檔案時 log 延遲出現**：非 TTY 時 `print()` 是 block-buffered，背景執行請加 `python -u`
  （容器已設 `PYTHONUNBUFFERED=1`）。
- **Port 被佔用**：`netstat -ano | findstr :8741` 找出 PID，再 `taskkill /PID <pid> /F`。
- **`uv sync` 寫到一半失敗**：還有其他行程正在用 `.venv`（例如背景的 worker 或測試），Windows 會鎖住已載入的
  `.pyd` / DLL，留下版本混亂的安裝。先停掉那些行程再重跑 `uv sync`。
- **`load_dotenv()` 要在 import agent 之前執行**：`agents/app.py` 在 import 階段就讀 `DB_URI` 並建立模型，
  自己寫的腳本要先 `load_dotenv()` 再 import（`worker.py` 就是這樣寫的）。
- **`init_chat_model` 的參數**：`reasoning_effort`、`request_timeout` 等要當直接的命名參數傳入，
  不能塞進 `model_kwargs`，否則執行期會拋 `ValidationError`。
