"""LLM tool router：用「Query Rewrite → Hybrid Search（dense + BM25）→ RRF 融合」
選出動態工具，再與基底工具聯集，覆寫本輪送進主模型的工具集。

規模說明：本專案只有 14 個工具，其實把全部工具丟給模型就夠了；這套檢索管線是為
「工具多到塞不進 context」的規模設計的，此處完整實作純為練習業界標準管線。檢索後端
（Qdrant 索引與 Hybrid+RRF 查詢）在 tool_index.py。

分層設計：
- 基底工具（Base Tools）：每輪都在場。set_goal 必須隨時可呼叫（換任務時記錄目標）；
  profiles_get / inventory_get / web_search 是推薦類任務的骨幹，且檢索召回率非 100%
  （實測「推薦晚餐」會漏撈 web_search），用基底兜住這種漏撈。
- 動態工具（Dynamic Tools）：對「重寫後的查詢」做 Hybrid Search 取 Top-K。
- 圖片：偵測到本輪含圖片時，強制補 inventory_add（沿用原關鍵字 router 的行為）。
"""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.tools import ALL_TOOLS
from app.agents.tool_index import retrieve_tool_names

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# 基底工具：不管檢索結果如何都保留（理由見模組 docstring）。
_BASE_TOOLS: set[str] = {"profiles_get", "inventory_get", "web_search", "set_goal"}

# 每輪 Hybrid Search 取幾個動態工具（spec 的 Top 5）
_RETRIEVE_LIMIT = 5

# 送進 query rewrite 的最近訊息數（含使用者輸入、AI 文字、tool 輸出），
# 讓重寫助理能解析代名詞與延續性指令（「下一步」「那雞胸肉呢」）。
_CONTEXT_WINDOW = 6

# call_agent 收到圖片時會把 vision 描述以固定字串組進訊息（見 app.py），
# 主 agent 收到的訊息已不含 image_url block，靠這個標記判斷「本輪含圖片」。
_IMAGE_MARKER = "系統自動辨識圖片中的食材"

# 由 app.py init_agent_infra 注入 summary_model（Groq llama-3.1-8b-instant，低延遲）。
# None 表示尚未啟用 → 跳過重寫，直接用原文檢索。沿用 tools.py 的注入慣例。
_rewrite_model = None

_REWRITE_PROMPT = """你是一個「意圖翻譯官」，只翻譯使用者當下想做的「動作」，用來檢索最合適的工具。

【嚴格禁忌】
1. 絕對不要預測答案、不要接續對話、不要幫使用者把下一步的具體內容編出來！
2. 絕對不可以遺漏使用者提到的「關鍵實體、食材、過敏原或限制條件」！

【核心原則】
- 如果使用者是在「陳述狀況、過敏原、飲食限制或個人偏好」，請 100% 保留所有名詞與條件，不可簡化為抽象句。

【範例 1：代名詞與主詞還原】
歷史：使用者：幫我查 A 專案的進度 / 助理：已完成 80%
使用者最後一句：那 B 專案呢
輸出：<query>查詢 B 專案的進度</query>

【範例 2：系統流程指令 (純動作，不預測內容)】
歷史：助理：第一步：請輸入舊密碼
使用者最後一句：下一步
輸出：<query>前進到流程的下一個步驟</query>
（❌ 錯誤示範，不要這樣寫：<query>輸入新密碼</query>——這是在編造下一步內容）

【範例 3：對話確認/否定 (僅在助理主動詢問時套用)】
歷史：助理：確認要刪除這筆紀錄嗎？
使用者最後一句：對，都刪了吧
輸出：<query>確認執行刪除操作</query>

【範例 4：陳述個人狀況/過敏限制 (關鍵詞全留)】
歷史：（無）
使用者最後一句：我對花生過敏，不能吃蝦
輸出：<query>記錄對花生過敏且不能吃蝦的飲食限制</query>
（❌ 錯誤示範，不要這樣寫：<query>確認不能吃蝦</query>——這遺漏了過敏原名詞）

規則：
- 消除代名詞與上下文依賴，但只還原「動作」，不要補充猜測內容。
- 必須完整保留所有的「過敏原、食材、數量、偏好」。

【輸出格式】
你必須且只能將最終的重寫結果包在 <query> 與 </query> 標籤中，不管前面加了什麼說明文字都沒關係，
只有標籤內的內容會被採用。例如：<query>查詢台北天氣</query>

<對話歷史>
{history}
</對話歷史>

<使用者最後一句>
{latest}
</使用者最後一句>"""


def _clean(text: str) -> str:
    """去掉 <user_input>/<tool_output> 這類隔離標籤，讓重寫助理看到乾淨文字。"""
    for tag in ("<user_input>", "</user_input>", "<tool_output>", "</tool_output>"):
        text = text.replace(tag, "")
    return text.strip()


def _msg_text(msg) -> str:
    """取一則訊息的純文字內容（list 型 content 只取 text 片段）。"""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""


def _extract_context(messages) -> tuple[str, str, bool]:
    """回傳 (對話歷史文字, 使用者最後一句, 是否含圖片)。

    歷史 = 最近 _CONTEXT_WINDOW 則的角色標記文字；最後一句 = 最後一則 HumanMessage。
    """
    window = messages[-_CONTEXT_WINDOW:]
    has_image = False
    history_lines: list[str] = []
    for m in window:
        text = _clean(_msg_text(m))
        if _IMAGE_MARKER in text:
            has_image = True
        if not text:
            continue
        if isinstance(m, HumanMessage):
            history_lines.append(f"使用者：{text}")
        elif isinstance(m, AIMessage):
            history_lines.append(f"助理：{text}")

    latest = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            latest = _clean(_msg_text(m))
            break

    return "\n".join(history_lines), latest, has_image


async def _rewrite_query(history: str, latest: str) -> str:
    """Step 1 Query Rewrite：小模型消代名詞。失敗或未啟用時退回原文。

    callbacks=[] 切斷與外層 run 的 callback 鏈，避免這次側呼叫的輸出被 astream
    的 messages 模式當成正常串流丟給使用者（同 DietarySafetyGuard 的做法）。
    """
    if _rewrite_model is None or not latest:
        return latest
    try:
        resp = await _rewrite_model.ainvoke(
            _REWRITE_PROMPT.format(history=history or "（無）", latest=latest),
            config={"callbacks": []},
        )
        raw = resp.content or ""
        # 用「最後一個」<query> 開始位置定位，不用 re.search 抓第一個匹配：
        # 8B 小模型常會先用自己的話複誦一次指令（含「包在 <query> 標籤中」這種
        # 字面提及），若複誦文字裡剛好只有一組 <query>...</query> 閉合，
        # search 會誤把「指令說明本身」當成要擷取的內容，把中間所有雜訊
        # 都吞進去。真正的答案幾乎必定是模型最後才給的，故取最後一次出現。
        last_open = raw.rfind("<query>")
        cleaned = ""
        if last_open != -1:
            remainder = raw[last_open + len("<query>"):]
            close_idx = remainder.find("</query>")
            content = remainder[:close_idx] if close_idx != -1 else remainder
            cleaned = content.strip()
        if cleaned:
            return cleaned
        # 沒抓到標籤或標籤內是空的，代表這次生成壞掉了，保底退回原文，
        # 不讓垃圾字串靜默流進檢索（同本專案零信任的一貫原則）。
        print(f"⚠️ [ROUTER] 未解析出 <query> 標籤，改用原文。原始輸出：{raw!r}")
        return latest
    except Exception as exc:
        print(f"⚠️ [ROUTER] query rewrite 失敗，改用原文：{type(exc).__name__}: {exc}")
        return latest


@wrap_model_call
async def tool_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    history, latest, has_image = _extract_context(request.messages)

    # Step 1：查詢重寫（消代名詞、補意圖）
    query = await _rewrite_query(history, latest)

    # Step 2-3：Hybrid Search（dense + BM25）+ RRF 融合，取動態工具
    dynamic = await retrieve_tool_names(query, limit=_RETRIEVE_LIMIT)

    # Step 4：基底工具 + 動態工具（+ 圖片強制補 inventory_add）
    selected: set[str] = set(_BASE_TOOLS)
    selected.update(dynamic)
    if has_image:
        selected.add("inventory_add")

    subset = [_TOOL_MAP[name] for name in selected if name in _TOOL_MAP]
    print(f"🧭 [ROUTER] rewrite={query!r} dynamic={dynamic} → {sorted(t.name for t in subset)}")
    return await handler(request.override(tools=subset))
