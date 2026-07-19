# app/ 架構總覽

一句話：**FastAPI + LangGraph 打造的「私廚 AI 助理」，核心是一個帶四道防護（步數/重複/逾時/HITL）＋
Postgres 長期記憶的 ReAct agent，圖片辨識則交由獨立的 vision 模型單次呼叫完成，不進 agent 迴圈。**

## 目錄結構

```
app/
├── main.py                 FastAPI 入口、CORS、掛路由、掛靜態前端、lifespan 初始化 agent
├── agents/
│   ├── app.py               Agent 核心：model、middleware 鏈、vision 辨識、_stream_agent、對外介面
│   ├── router.py            tool_router middleware：關鍵字比對，動態縮小每輪可用工具集
│   ├── tools.py              私廚工具定義 + ChefState + store namespace
│   └── lessons.py            教訓記憶（procedural memory）：確定性失敗訊號 → 常駐行為規則
├── api/v1/
│   ├── chat.py               /chat/stream /chat/resume /chat/messages /chat/upload-image
│   └── oss.py                 GCS 預簽名 URL（上傳 + 給 LLM 讀取）
├── models/schemas.py         ChatRequest / ResumeRequest pydantic 模型
├── common/
│   ├── image_utils.py        圖片 bytes → Base64 Data URL
│   └── logger.py             logging 設定
└── static/                   前端（index.html / app.js / style.css）
```

## 請求流程

```
瀏覽器 (static/)
   │ POST /api/v1/chat/stream {message, image_url?, thread_id, user_id?}
   ▼
main.py（FastAPI，lifespan 已呼叫 init_agent_infra() 建好連線池/agent）
   ▼
api/v1/chat.py::chat_endpoint → await call_agent(...) → StreamingResponse(...)
   ▼
agents/app.py::call_agent
   │  若帶圖片：先 await _describe_image()（獨立 vision 模型，單次呼叫、無工具）
   │  把辨識結果轉成純文字包進 <tool_output>，跟 <user_input> 一起組成訊息
   ▼
_stream_agent()
   ▼
agent.astream(..., stream_mode=["messages","updates"])   ← create_agent 建出的 LangGraph
   │   （四道防護：步數 / 重複偵測 / 整體逾時 / HITL 中斷）
   ▼
逐字 yield 文字 給前端；工具呼叫時 yield 系統提示字串
```

HITL 中斷時回傳 `{"type":"interrupt","action_requests":[...]}`，前端要呼叫
`/chat/resume` 並帶 `decisions:[{"type":"approve"}]` 或 `reject` 才會繼續。

## 三個模型的分工（`agents/app.py`）

各自獨立、互不掛載對方的 tools，刻意不疊加在同一次生成裡：

| 模型 | 用途 | Provider | 備註 |
|---|---|---|---|
| `model` | 主 ReAct agent，tool-calling 對話 | NVIDIA NIM `openai/gpt-oss-120b` | 實測 Groq 版本（`llama-3.3-70b-versatile`／`openai/gpt-oss-120b`）皆有各自的 tool-calling 穩定性問題，見下方「模型選型記錄」 |
| `summary_model` | 摘要壓縮／web_search 蒸餾／過敏原安全判定等輕量子任務 | Groq `llama-3.1-8b-instant` | 延遲低、便宜，未觀察到問題 |
| `vision_model` | 圖片辨識（僅描述食材，不掛任何工具） | Groq `qwen/qwen3.6-27b` | Groq 上唯一支援圖片輸入的模型；回覆含 `<think>...</think>` 推理區塊需自行濾除 |

**Vision 與 tool-calling 解耦**：`call_agent` 收到圖片時，先用 `vision_model` 單次 `ainvoke`
把圖片轉成純文字食材描述（`_describe_image()`），再包進 `<tool_output>` 標籤跟隨純文字訊息一起
送進主 agent；圖片本身（`image_url` content block）**不會**進入主 agent 的訊息歷史。這樣主模型
只需要處理「選工具、填 schema」，vision 模型只需要處理「看圖」，兩者互不疊加失敗率。
`_describe_image()` 失敗時回傳空字串而非拋例外，訊息退化成「請忽略圖片內容」，不中斷整輪對話。

**⚠️ 連動限制**：`agents/router.py` 的 `tool_router` 靠掃訊息裡有沒有 `image_url` content block
或固定字串 `_IMAGE_MARKER`（"系統自動辨識圖片中的食材"）判斷「這輪有圖片」以放行 `inventory_add`。
`_describe_image()` 的辨識結果字串、以及包住它的 `<tool_output>` 文案若被改動，務必同步檢查
`router.py` 的 `_IMAGE_MARKER` 是否還抓得到，否則會出現「模型宣稱已存檔，實際上 `inventory_add`
從未被呼叫」的靜默失敗（曾在實測中發生過一次）。

## Agent 核心

- **State**：`ChefState`（`agents/tools.py`）在 `AgentState` 之上加
  `cooking_steps` / `current_step` / `current_recipe` / `original_goal`，隨 checkpointer
  持久化到該 thread。
- **middleware 鏈**（由外而內執行順序即下列順序）：
  1. `DietarySafetyGuard`——零信任過敏原檢查：模型回覆後直接呼叫 `summary_model`
     `_find_recommended_allergen()`，一次將所有過敏原清單傳入 LLM 判定是否推薦了其中任何一種。
     命中則把違規回覆連同安全提示送回模型重試（最多 2 次），仍命中則直接攔截
     改寫成安全提示。不信任模型自稱「已避開」；判定器故障時 fail closed（一律視為違規）。
  2. `tool_router`（`agents/router.py`）——關鍵字規則比對最近 8 則訊息文字，動態決定
     這一輪 model call 能看到哪些工具（`_CORE_TOOLS` 保底 + 命中規則追加）。目的是縮小
     工具選單降低模型選錯工具機率。
  3. `ContextShaping`——① 把同一 thread 內已被新呼叫取代的舊工具快照
     （`inventory_get`/`profiles_get`/`web_search`）內容替換成佔位字串（不刪除整則訊息，
     避免破壞 OpenAI API 的 tool_call_id 配對要求）；② 把 `set_goal` 工具寫入 state 的
     `original_goal` 每輪注入 system prompt，防止長對話中任務漂移，並聲明衝突時以使用者
     最新訊息為準；③ 把 `lessons.py` 累積的常駐教訓規則注入 system prompt。
  4. `SoftLanding`——工具呼叫次數逼近上限（`MAX_STEPS-2`）時卸除全部工具、要求模型只憑
     現有資訊收尾並聲明「可能不完整」，比硬性拒答更優雅。
  5. `ReflectingSummarization`（繼承 `SummarizationMiddleware`）——對話超過約 3000 token
     觸發摘要壓縮；壓縮前先把「即將被壓掉的訊息」丟給 LangMem 的 `ReflectionExecutor`
     做背景反思萃取（更新使用者長期 profile），避免資訊隨摘要遺失。
  6. `ModelCallLimitMiddleware(run_limit=MAX_STEPS)`——硬性步數上限（`MAX_STEPS=15`），
     超過拋 `ModelCallLimitExceededError`，由 `_stream_agent` 攔截轉成中文提示。
  7. `HumanInTheLoopMiddleware`——攔截會修改持久化狀態的工具：`inventory_remove`、
     `diet_profile_manage`、`kitchen_profile_manage`、`household_profile_manage`。
     觸發時 graph 產生 `__interrupt__`，需呼叫 `resume_agent` 帶 approve/reject 才繼續。
- **`_stream_agent()` 的四道防護機制**：
  1. 最大步數：`ModelCallLimitMiddleware` + `recursion_limit=MAX_STEPS*2+1`（後者是最後防線）
  2. 重複工具偵測：用 `deque(maxlen=MAX_REPEAT=3)` 記錄 `(name, args_json)`，連續 3 次
     完全相同的工具呼叫視為卡住，強制終止（跨 HITL resume 持久化於 `_thread_recent_calls`）
  3. 整體逾時：`asyncio.timeout(MAX_TIMEOUT=120)` 包住整個 streaming 迴圈
  4. HITL：見上
- **教訓記憶（`agents/lessons.py`）**：只從程式可驗證的確定性訊號寫入（重複呼叫、
  HITL 被拒、過敏原誤推薦、軟著陸觸發、搜尋策略反覆），累積達門檻（`hits >= 2`）才升格
  為常駐規則注入 system prompt，附 debounce 與 TTL，避免單次失誤誤判成行為模式。
- **持久化**：
  - `AsyncPostgresSaver`（checkpointer）——短期記憶，對話歷程按 `thread_id` 還原
  - `AsyncPostgresStore`（store）——長期記憶，`(namespace, key) → value`：
    - `("inventory", user_id)` → 冰箱食材清單
    - `("memories", user_id, "diet"/"kitchen"/"household")` → 三個 LangMem typed-memory
      profile（由 `create_manage_memory_tool` 自動萃取/更新/刪除）
    - `("lessons", user_id)` → 教訓記憶
- **對外介面**：`call_agent()`（新訊息，`async def`）、`resume_agent()`（HITL 續行）、
  `get_checkpointer()` / `delete_checkpointer()`（讀/刪對話歷史）。

## 工具清單（`agents/tools.py` → `ALL_TOOLS`）

| 工具 | 用途 |
|---|---|
| `set_goal` | 記錄/更新使用者當前任務目標（寫入 state.original_goal，供 ContextShaping 防漂移） |
| `web_search` | Tavily 網路搜尋，含逾時保護（20s，獨立 thread pool）、prompt-injection guardrail（LLM 偵測+regex 模糊移除）、結果過長時 LLM 蒸餾 |
| `nutrition_lookup` | 固定表查詢食材熱量/蛋白質/脂肪/碳水，零幻覺 |
| `inventory_get/add/remove` | 冰箱庫存讀寫（store）。`add` 由圖片辨識或文字觸發，`remove` 走 HITL |
| `shopping_list_generate` | 比對食譜所需食材 vs 冰箱庫存，產出缺料清單 |
| `profiles_get` | 讀取 diet/kitchen/household 三個長期 profile |
| `diet_profile_manage` / `kitchen_profile_manage` / `household_profile_manage` | LangMem 自動管理記憶工具，寫入對應 profile（HITL 攔截） |
| `step_tracker_start/next/current` | 逐步引導食譜步驟，用 `Command` 同時更新 messages 與 state |

## 安全設計重點

- system_prompt 明確要求 `<user_input>`／`<tool_output>` XML 標籤隔離：標籤內一律視為
  資料而非指令，防止 prompt injection（來自使用者、網路搜尋結果、或圖片辨識結果——
  vision 模型的輸出也可能被圖片裡藏的文字操控，同樣不當可信指令處理）。
- `web_search` 的 guardrail：小模型判斷搜尋結果是否含惡意指令，命中則用 fuzzy regex
  切除該片段（容忍全形/半形/大小寫差異），而非整篇丟棄。
- `DietarySafetyGuard`：不信任模型自稱「已避開過敏原」，模型回覆後直接呼叫小模型
  `_find_recommended_allergen()`，一次判斷所有過敏原，判定器故障時 fail closed。
- 寫入類工具一律走 HITL，人工 approve 才真正落地。

## Windows 部署踩坑（已解決，寫在 main.py / app.py 開頭註解）

- uvicorn 預設 `ProactorEventLoop` 與 psycopg async 不相容 → `main.py` 自訂
  `selector_loop_factory`，啟動時加 `--loop app.main:selector_loop_factory`。
- Windows 終端機 cp950 無法輸出 emoji（agent 內部 print 用了 emoji）→ `agents/app.py`
  import 階段就把 stdout/stderr `reconfigure(encoding="utf-8")`。
- **`print()` 預設是 block-buffered（非 TTY 導向檔案時）**：背景執行 uvicorn 並把
  輸出導向檔案時，若不加 `-u`（unbuffered），錯誤訊息可能延遲很久才寫進 log，
  排查問題時容易誤以為「沒有印出例外」。
- **Port 占用 / 殘留行程排查**：服務異常中斷後 Port 未釋放時，用
  `netstat -ano | findstr :8001` 尋找占用 PID，再用 `taskkill /PID <pid> /F` 釋放埠口。
- **共用 venv 時，其他服務占用會讓 `uv sync` 中途失敗**：若同一個 venv 底下還有其他
  服務在跑（例如本專案的 `deepresearch/`），它載入記憶體的 `.pyd`／DLL 會被 Windows
  鎖住，`uv sync` 升級套件寫到一半失敗會留下版本混亂的殘破安裝（多個 dist-info
  並存、套件目錄缺檔）。需先停掉占用該 venv 的其他行程，再重跑 `uv sync` 修復。
- **`load_dotenv()` 必須在會讀取環境變數的 import 之前執行**：例如 `TavilySearch()`
  在 import 階段就會讀 `TAVILY_API_KEY`，若 `load_dotenv()` 太晚執行會拋出
  `ValidationError`。目前放置於 `agents/app.py` 頂端確保最早載入。

## 模型設定注意事項

- **`init_chat_model` 參數放置規則**：`reasoning_effort`、`request_timeout` 等必須作為
  `init_chat_model()` 的直接命名參數傳入，不能塞進 `model_kwargs` dict，否則
  pydantic 驗證會在 runtime 拋 `ValidationError`。
- **`parallel_tool_calls`**：若透過 `model_kwargs` 間接傳入非直接支援的參數，
  `init_chat_model` 只會印 `UserWarning` 而非報錯（實際仍會生效，但代表這個 provider
  的 SDK 封裝沒有把它當一等公民參數看待，行為以官方文件為準，不要只看有沒有報錯）。
- **NVIDIA NIM provider**：`model_provider="openai"` + `base_url="https://integrate.api.nvidia.com/v1"`，
  讀取對應的 `NVIDA_*` API key（注意變數名拼寫沿用既有慣例，非筆誤）。
- **Groq provider**：`model_provider="groq"`，不需要 `base_url`，讀取 `GROQ_API_KEY`。
- **同一個 Groq 帳號底下的多個模型，TPD（tokens per day）額度是各模型獨立計算**，
  但同一個 org 底下換不同 API key **不會**得到獨立額度——TPD 是 org 層級共用，
  額度打滿時换 key 沒用，只能等滾動窗口釋放或換到真正不同 org 的帳號。

## 模型選型記錄（實測結論，2026-07-19）

主模型（tool-calling 為核心）在不同 provider / model 上的實測表現：

| 模型 | Provider | 觀察到的問題 |
|---|---|---|
| `llama-3.3-70b-versatile` | Groq | 複雜巢狀 schema（如 `DietProfile` 多個 list/optional 欄位）下偶發 `Failed to call a function` 格式錯誤；且免費層 TPD 僅 100K/日，容易被測試用量打滿 |
| `openai/gpt-oss-120b` | Groq | 對「全新使用者、diet namespace 尚無既有記錄」的情境，`diet_profile_manage` 系統性誤選 `action="update"` 且不帶必要的 `id` 參數（實測可重現），導致 LangMem 直接拋 `ValueError` 炸穿整個 run |
| `openai/gpt-oss-120b` | NVIDIA NIM | 目前採用版本，tool-calling 穩定、schema 生成正確、多工具序列（`set_goal` → `profiles_get` → `web_search` → `diet_profile_manage`）符合 system_prompt 規則 |

**延伸設計原則**：`create_manage_memory_tool` 這類「update 需要 id」的工具，若呼叫時
action/id 不合法，LangMem 目前是直接拋例外而非回傳可讓模型自我修正的錯誤訊息，
會被 `_stream_agent` 的 catch-all 接住變成整輪對話失敗。之後若要進一步強化，可考慮
在呼叫路徑上包一層防禦，把這類錯誤轉成 ToolMessage 回饋給模型重試，而不是讓整個
run 失敗。

## 多模態（圖片輸入）

- 圖片上傳流程：`/chat/upload-image` 收到圖片 → 轉 Base64 Data URL →
  `call_agent` 收到非 null 的 `imageUrl` → 呼叫 `_describe_image()`（`vision_model`
  單次辨識，見上方「三個模型的分工」）→ 辨識結果以純文字包進 `<tool_output>` →
  正常走主 agent 的 ReAct 迴圈（此時訊息裡已無 `image_url` block）。
- 圖片本身只會被 `vision_model` 看到一次，不會進入主 agent 的訊息歷史、不會被
  checkpointer 持久化，也不會在後續對話輪次中重複佔用 context。

## 環境變數

`DB_URI`（Postgres 連線字串）、`GROQ_API_KEY`（Groq，`summary_model`／`vision_model` 用）、
`NVIDA_API_KEY`（NVIDIA NIM，主模型 `model` 用，注意變數名拼寫）、
`TAVILY_API_KEY`（`web_search` 用）、
`GCS_BUCKET` + `GOOGLE_APPLICATION_CREDENTIALS`（`oss.py` 用）。
