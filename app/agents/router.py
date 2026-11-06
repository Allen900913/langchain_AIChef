"""LLM tool router：用「Query Rewrite → Hybrid Search（dense + BM25）→ RRF 融合」
選出動態工具，再與基底工具聯集，覆寫本輪送進主模型的工具集。

規模說明：本專案只有 12 個工具，其實把全部工具丟給模型就夠了；這套檢索管線是為
「工具多到塞不進 context」的規模設計的，此處完整實作純為練習業界標準管線。檢索後端
（Qdrant 索引與 Hybrid+RRF 查詢）在 tool_index.py。

分層設計：
- 基底工具（Base Tools）：每輪都在場。set_goal 必須隨時可呼叫（換任務時記錄目標）；
  inventory（查冰箱）/ web_search 是推薦類任務的骨幹，且檢索召回率非 100%
  （實測「推薦晚餐」會漏撈 web_search），用基底兜住這種漏撈。
  profiles_get 不放基底：ContextShaping 每輪都會把偏好預注入 system prompt。
- 動態工具（Dynamic Tools）：對「重寫後的查詢」與「使用者原話」各做 Hybrid Search 取 Top-K 後聯集，
  基底工具不參與檢索、不佔名額。排名第一的是偏好寫入工具時，補成完整的工具族群（見 _TOOL_FAMILIES）。
- 同類的動作合併成一個工具、用 action 參數區分（inventory、step_tracker），不會互搶 Top-K 名額。
- 圖片：偵測到本輪含圖片時，強制補 inventory（存入辨識出的食材）。
"""

from collections import OrderedDict
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.tool_index import retrieve_tool_names
from app.agents.prompt_manager import (
    PROMPT_REWRITE,
    DEFAULT_REWRITE_PROMPT,
    compile_prompt as _compile_lf_prompt,
)

# 基底工具：不管檢索結果如何都保留（理由見模組 docstring）。
_BASE_TOOLS: set[str] = {"inventory", "web_search", "set_goal"}

# 每輪 Hybrid Search 取幾個動態工具（spec 的 Top 5）
_RETRIEVE_LIMIT = 5

# 工具族群：族裡的工具排名第一時，整族一起帶進來。同一族的工具 description 很像、彼此搶名次，
# Top-K 常只撈到其中幾個（2026-11-01 實測：「我對蝦過敏」被改寫成「不要有蝦」，撈到 kitchen /
# household 的 manage 卻漏了 diet，模型沒有工具可用、只能口頭說「已記住」）。
# 步驟引導、冰箱當時也是同樣問題，已經各自合併成一個工具（step_tracker、inventory）；
# 三個偏好工具的參數 schema 各不相同（LangMem 每個領域一個 schema），沒有合併，靠族群展開。
# 只看排名第一：這族共約 2000 token，主模型每分鐘只有 8000。只要撈到就展開的話，41 題標註問題裡
# 40 題都會帶入；只看第一名時帶入 14 題，召回率不變（40/41，2026-11-06 離線評估）。
_TOOL_FAMILIES: list[frozenset[str]] = [
    frozenset({"diet_profile_manage", "kitchen_profile_manage", "household_profile_manage"}),
]


def _expand_families(names: list[str]) -> set[str]:
    """names 是依分數排序的檢索結果：第一名屬於某個族群時補成完整的族群，其他工具原樣保留。"""
    expanded = set(names)
    for family in _TOOL_FAMILIES:
        if names and names[0] in family:
            expanded |= family
    return expanded

# 送進 query rewrite 的最近訊息數（含使用者輸入、AI 文字、tool 輸出），
# 讓重寫助理能解析代名詞與延續性指令（「下一步」「那雞胸肉呢」）。
_CONTEXT_WINDOW = 6

# call_agent 收到圖片時會把 vision 描述以固定字串組進訊息（見 app.py），
# 主 agent 收到的訊息已不含 image_url block，靠這個標記判斷「本輪含圖片」。
_IMAGE_MARKER = "系統自動辨識圖片中的食材"

# 由 app.py init_agent_infra 注入 summary_model（Groq gpt-oss-20b，低延遲）。
# None 表示尚未啟用 → 跳過重寫，直接用原文檢索。沿用 tools.py 的注入慣例。
_rewrite_model = None


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

    歷史 = 到最後一則 HumanMessage 為止、最近 _CONTEXT_WINDOW 則的角色標記文字；最後一句 = 最後一則
    HumanMessage。不算這一輪之後的訊息：同一輪每次呼叫模型的輸入都一樣，rewrite 才快取得到。
    """
    end = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            end = i + 1
            break
    window = messages[:end][-_CONTEXT_WINDOW:]
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


# 一輪對話會呼叫主模型好幾次（工具呼叫前後），每次都經過 router。輸入一樣就不再花一次 rewrite。
# 只存在這個行程裡：resume 換到別台 worker 時只是多呼叫一次。失敗不快取，下次還有機會成功。
_REWRITE_CACHE_SIZE = 256
_rewrite_cache: OrderedDict[tuple[str, str], str] = OrderedDict()


async def _rewrite_query(history: str, latest: str) -> str:
    """Query Rewrite，同樣的輸入直接回上次的結果。"""
    key = (history, latest)
    if key in _rewrite_cache:
        _rewrite_cache.move_to_end(key)
        return _rewrite_cache[key]
    query, ok = await _call_rewrite(history, latest)
    if ok:
        _rewrite_cache[key] = query
        if len(_rewrite_cache) > _REWRITE_CACHE_SIZE:
            _rewrite_cache.popitem(last=False)
    return query


async def _call_rewrite(history: str, latest: str) -> tuple[str, bool]:
    """Step 1 Query Rewrite：小模型消代名詞。回傳 (查詢, 是否成功)；失敗或未啟用時退回原文。

    callbacks=[] 切斷與外層 run 的 callback 鏈，避免這次側呼叫的輸出被 astream
    的 messages 模式當成正常串流丟給使用者（同 DietarySafetyGuard 的做法）。
    """
    if _rewrite_model is None or not latest:
        return latest, False
    try:
        # 動態取得並編譯 Prompt（prompt_manager 帶 300s 快取，支援 Langfuse 與本地 fallback）
        prompt_text = _compile_lf_prompt(
            PROMPT_REWRITE,
            DEFAULT_REWRITE_PROMPT,
            history=history or "（無）",
            latest=latest,
        )
        resp = await _rewrite_model.ainvoke(
            prompt_text,
            config={"callbacks": []},
        )
        raw = resp.content or ""
        # 用「最後一個」<query> 開始位置定位，不用 re.search 抓第一個匹配：
        # 小模型常會先用自己的話複誦一次指令（含「包在 <query> 標籤中」這種
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
            return cleaned, True
        # 沒抓到標籤或標籤內是空的，代表這次生成壞掉了，保底退回原文，
        # 不讓垃圾字串靜默流進檢索（同本專案零信任的一貫原則）。
        print(f"⚠️ [ROUTER] 未解析出 <query> 標籤，改用原文。原始輸出：{raw!r}")
        return latest, False
    except Exception as exc:
        print(f"⚠️ [ROUTER] query rewrite 失敗，改用原文：{type(exc).__name__}: {exc}")
        return latest, False


@wrap_model_call
async def tool_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    history, latest, has_image = _extract_context(request.messages)

    # Step 1：查詢重寫（消代名詞、補意圖）
    query = await _rewrite_query(history, latest)

    # Step 2-3：Hybrid Search（dense + BM25）+ RRF 融合，取動態工具。
    # 改寫後的查詢與使用者原話各檢索一次取聯集：改寫偶爾會丟掉決定用哪個工具的字
    # （2026-11-01 實測：「我在之前的對話裡請你教我做哪一道菜」→「請教我做哪一道菜的步驟」，
    # 回想的意思沒了，conversation_search 撈不到）。embedding 在本機，多一次只多幾十毫秒。
    retrieved = [await retrieve_tool_names(query, limit=_RETRIEVE_LIMIT, exclude=_BASE_TOOLS)]
    if latest and latest != query:
        retrieved.append(await retrieve_tool_names(latest, limit=_RETRIEVE_LIMIT, exclude=_BASE_TOOLS))
    dynamic = list(dict.fromkeys(n for names in retrieved for n in names))

    # Step 4：基底工具 + 動態工具（補成完整族群）（+ 圖片強制補 inventory）
    selected: set[str] = set(_BASE_TOOLS)
    for names in retrieved:
        selected.update(_expand_families(names))
    if has_image:
        selected.add("inventory")

    # 只從「這個 agent 有註冊的工具」裡挑：訪客 agent 只有不碰長期記憶的工具（tools.GUEST_TOOLS），
    # 檢索結果裡的其他工具直接略過，不會把沒註冊的工具交給模型
    available = {getattr(t, "name", None): t for t in request.tools}
    subset = [available[name] for name in selected if name in available]
    print(f"🧭 [ROUTER] rewrite={query!r} dynamic={dynamic} → {sorted(t.name for t in subset)}")
    return await handler(request.override(tools=subset))
