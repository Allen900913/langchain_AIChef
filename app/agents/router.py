from collections.abc import Awaitable, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from app.agents.tools import ALL_TOOLS

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# 不管命中哪條規則都保留的核心工具，避免 ReAct 中途想用卻被鎖死拿不到。
# set_goal 必須每圈在場：LLM 判斷使用者換任務時要能隨時呼叫它記錄新目標。
_CORE_TOOLS: set[str] = {"profiles_get", "inventory_get", "web_search", "set_goal"}

# 每輪納入判斷的最近訊息數（含 AI 文字、tool 輸出），讓 router 能看到「目前執行狀態」
# 而非只看最初那句話，否則 step 1 查完冰箱、step 2 才發現需要的工具會永遠呼叫不到
_CONTEXT_WINDOW = 8

# call_agent 收到圖片時，會先呼叫 vision_model 把圖片轉成純文字描述再組進訊息
# （見 app.py call_agent），送進主 agent 的訊息已不含 image_url content block，
# 下面 _extract_recent_context 的 image_url 偵測抓不到。故額外用這個固定字串
# 當「本輪含圖片」的訊號——沿用本檔既有的關鍵字比對風格，而非額外傳遞旗標。
_IMAGE_MARKER = "系統自動辨識圖片中的食材"

_RULES: list[tuple[list[str], list[str]]] = [
    (
        ["我有", "買了", "我買", "冰箱有", "還剩", "剛買"],
        ["inventory_add"],
    ),
    (
        ["用完", "沒了", "過期", "丟掉", "吃完", "用掉"],
        ["inventory_remove"],
    ),
    (
        ["過敏", "不吃", "吃素", "忌口", "討厭", "素食", "vegan", "vegetarian", "halal", "keto",
         "重口味", "清淡", "不辣", "偏甜", "偏鹹", "台式", "日式", "韓式", "義式", "快炒",
         "減重", "增肌", "控糖", "低鈉", "瘦身", "健康飲食"],
        ["diet_profile_manage"],
    ),
    (
        ["氣炸鍋", "烤箱", "電鍋", "瓦斯爐", "廚具", "鍋", "新手", "不會煮", "沒時間", "分鐘"],
        ["kitchen_profile_manage"],
    ),
    (
        ["家人", "家裡", "女兒", "兒子", "老婆", "老公", "小孩", "孩子", "懷孕", "幾人份", "全家"],
        ["household_profile_manage"],
    ),
    (
        ["下一步", "然後呢", "第幾步", "接下來", "繼續", "下步"],
        ["step_tracker_next", "step_tracker_current"],
    ),
    (
        ["缺", "要買", "購物", "採買", "需要買"],
        ["shopping_list_generate", "inventory_get"],
    ),
    (
        ["熱量", "卡路里", "營養", "蛋白質", "脂肪", "碳水"],
        ["nutrition_lookup"],
    ),
    (
        ["推薦", "想吃", "食譜", "怎麼做", "教我", "料理", "做法", "煮什麼", "吃什麼", "學做"],
        ["profiles_get", "inventory_get", "web_search",
         "shopping_list_generate", "nutrition_lookup", "step_tracker_start"],
    ),
    (
        ["冰箱", "庫存", "有什麼食材", "有哪些食材"],
        ["inventory_get"],
    ),
]


def _extract_recent_context(messages) -> tuple[str, bool]:
    """回傳 (最近 _CONTEXT_WINDOW 則訊息合併文字, 是否含圖片)。

    掃描 HumanMessage / AIMessage / ToolMessage 的文字內容，讓 router 能看到
    「目前執行狀態」（含上一輪 tool 輸出），而非只看最初那句使用者輸入。
    訊息數少於 window 時，切片自然回傳全部，不會出錯。
    """
    text_parts: list[str] = []
    has_image = False
    for msg in messages[-_CONTEXT_WINDOW:]:
        content = msg.content
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        has_image = True
                elif isinstance(item, str):
                    text_parts.append(item)
    return " ".join(text_parts), has_image


@wrap_model_call
async def tool_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    text, has_image = _extract_recent_context(request.messages)

    selected: set[str] = set(_CORE_TOOLS)  # 保底：核心工具每輪都在場

    if has_image or _IMAGE_MARKER in text:
        selected.add("inventory_add")

    for keywords, tools in _RULES:
        if any(kw in text for kw in keywords):
            selected.update(tools)

    subset = [_TOOL_MAP[name] for name in selected if name in _TOOL_MAP]
    print(f"🧭 [ROUTER] {sorted(name for t in subset for name in [t.name])}")
    return await handler(request.override(tools=subset))
