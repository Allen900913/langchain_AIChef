"""反思（偏好萃取）改前 / 改後對照：python -m evals.reflection_eval [--reps N] [--variants A,B,C,D]

只測 LangMem 的 memory store manager 本身：給一份「既有偏好」和一段對話，看反思寫回什麼。
不跑 agent、不碰正式資料庫（每次都用全新的 InMemoryStore），所以結果只反映反思這一步。

四組設定，用來拆開每次改動各自的影響：
  A 原始       舊指令 + gpt-oss-120b（主模型）
  B 第二版     第二版指令 + gpt-oss-120b（只換指令）
  C 第二版正式 第二版指令 + gpt-oss-20b（summary_model）+ ProfileGuardStore（修正誤存的 patch、過敏原只增不減）
  D 目前正式   C + 窄版矛盾值規則（= tools.PROFILE_REFLECTION_INSTRUCTIONS）
  （寬版矛盾值規則的結果見 evals/results/reflection_20261016_15*.json，因一次性情境退步已收窄）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from langmem import create_memory_store_manager  # noqa: E402

from app.agents.memory_guard import ProfileGuardStore, ToolNameFixingChatGroq  # noqa: E402
from app.agents.tools import (  # noqa: E402
    PROFILE_REFLECTION_INSTRUCTIONS,
    _PROFILE_DOMAINS,
    _profile_ns,
)

USER_ID = "eval"

# 改動前的反思指令（凍結一份當基準，正式程式碼已改成 tools.PROFILE_REFLECTION_INSTRUCTIONS）
BASELINE_INSTRUCTIONS: dict[str, str] = {
    "diet": (
        "掃描這段完整對話，萃取飲食相關資訊到對應欄位。"
        "重點補即時工具漏掉的：隱含偏好（三次都選清淡 → flavor_preferences 加清淡）、"
        "對話後期才提到的限制、間接表達的好惡（『上次那道太辣』→ dislikes 加辣）。"
        "只存可觀察的事實，不存推論定論；欄位：allergies / dislikes / diet_type / "
        "flavor_preferences / cuisine_preferences / health_goals。"
    ),
    "kitchen": (
        "掃描這段完整對話，萃取廚房條件到對應欄位。"
        "補即時工具漏掉的：間接提到的廚具（包含抱怨沒有）、"
        "從做菜時間限制推斷的 time_budget、從提問難度推斷的 skill_level。"
        "欄位：equipment / skill_level / time_budget。"
    ),
    "household": (
        "掃描這段完整對話，萃取家庭成員資訊到對應欄位。"
        "members 只記錄使用者以外的其他同住或同桌成員（女兒、老婆、小孩、家人等）的飲食需求；"
        "使用者本人的過敏、忌口、飲食型態屬於 diet namespace，household 不重複存。"
        "補即時工具漏掉的：間接提到的成員（『我媽不吃辣』）、幾人份需求。"
        "欄位：members / usual_portions。"
    ),
}

# 第二版指令（加入一次性情境 / 替別人煮 / 過敏原只增不減），凍結一份當「矛盾值規則」的對照基準
_V2_RULES = (
    "偏好檔案只記錄長期、穩定的事實，之後每次推薦都會參考它，寫錯會一直影響下去。\n"
    "一律不寫入、也不修改既有記錄的情況：\n"
    "- 一次性情境：這次、今天、這週、這餐、臨時、剛好、心情、想換換口味。\n"
    "- 替別人煮：朋友、同事、客人、聚餐、宴客。他們的需求不是使用者的偏好，也不是家庭成員。\n"
    "- 單純詢問：問某道菜怎麼做，不代表喜歡它。\n"
    "會寫入的情況：使用者明確宣告（以後、從現在開始、我都、我一直、我不吃），"
    "或同一個選擇在對話中重複出現（三次都選清淡）。\n"
    "新資訊與舊記錄衝突時：只有使用者明確表示長期改變（以後、最近開始都、醫生要我、我已經改成、"
    "我買了 / 我丟了）才更新舊值；一次性的例外保留舊值。"
    "例如舊記錄不吃辣、使用者說「今天想吃辣」→ 維持不吃辣。\n"
    "這段對話沒有新的長期資訊時，原樣保留既有記錄，不要為了有輸出而改寫。\n\n"
)
V2_INSTRUCTIONS: dict[str, str] = {
    "diet": _V2_RULES + (
        "掃描這段完整對話，萃取飲食相關資訊到對應欄位。"
        "重點補即時工具漏掉的：隱含偏好（三次都選清淡 → flavor_preferences 加清淡）、"
        "對話後期才提到的限制、間接表達的好惡（『上次那道太辣』→ dislikes 加辣）。"
        "allergies 只能新增，不能移除或改寫：使用者替別人處理該食材、或說過敏好了，都保留原本的過敏原"
        "（移除要由使用者在對話中明確要求，走另一個有人工審核的流程）。"
        "只存可觀察的事實，不存推論定論；欄位：allergies / dislikes / diet_type / "
        "flavor_preferences / cuisine_preferences / health_goals。"
    ),
    "kitchen": _V2_RULES + (
        "掃描這段完整對話，萃取廚房條件到對應欄位。"
        "補即時工具漏掉的：間接提到的廚具（包含抱怨沒有）、"
        "從做菜時間限制推斷的 time_budget、從提問難度推斷的 skill_level。"
        "借來的、這次才用的廚具不算擁有；只有這幾天比較忙不改 time_budget。"
        "欄位：equipment / skill_level / time_budget。"
    ),
    "household": _V2_RULES + (
        "掃描這段完整對話，萃取家庭成員資訊到對應欄位。"
        "members 只記錄使用者以外的其他同住或同桌成員（女兒、老婆、小孩、家人等）的飲食需求；"
        "使用者本人的過敏、忌口、飲食型態屬於 diet namespace，household 不重複存。"
        "補即時工具漏掉的：間接提到的成員（『我媽不吃辣』）、幾人份需求。"
        "來訪的朋友、客人不是成員；請客時的人數不改 usual_portions。"
        "欄位：members / usual_portions。"
    ),
}

# 與 app/agents/app.py 的 model / summary_model 相同設定
MODELS = {
    "120b": lambda: init_chat_model(
        "openai/gpt-oss-120b", model_provider="groq", api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2, request_timeout=50.0, max_retries=3,
    ),
    # 模型設定同 summary_model，但預設用 KEY3（--key20b 可換）：Groq 每把 key 每個模型每天有 token 上限（20b 為 20 萬），
    # 評測跑一次就十幾萬，用正式 summary_model 的 KEY2 會把當天的摘要 / 反思額度吃光
    # 跟正式的 reflection_model 一樣用 ToolNameFixingChatGroq（去掉工具名稱的 functions. 前綴）
    "20b": lambda: ToolNameFixingChatGroq(
        model="openai/gpt-oss-20b", api_key=os.getenv(os.getenv("EVAL_KEY_20B", "GROQ_API_KEY3")),
        temperature=0, request_timeout=50.0,
    ),
    # 同上，推理強度降成 low（預設 medium）：看反思能不能跟 summary_model 一樣省掉大部分推理 token
    "20b-low": lambda: ToolNameFixingChatGroq(
        model="openai/gpt-oss-20b", api_key=os.getenv(os.getenv("EVAL_KEY_20B", "GROQ_API_KEY3")),
        temperature=0, reasoning_effort="low", request_timeout=50.0,
    ),
}

# 目前的指令拿掉「新值跟舊值相反時取代」那一段，用來判斷窄版矛盾值規則要不要留
_CONFLICT_RULE = (
    "只有在上面這種明確的長期改變裡，新值又跟同一欄位的某個舊值意思相反時，"
    "才用新值取代那個舊值，不讓兩個相反的值並存。"
    "例如 flavor_preferences 是 [偏甜, 台式]、使用者說醫生要他以後控糖 → [少糖, 台式]。"
    "同一欄位的其他舊項目不動；一次性情境照前面的規則，不取代任何舊值。\n"
)
assert all(_CONFLICT_RULE in v for v in PROFILE_REFLECTION_INSTRUCTIONS.values())
NO_CONFLICT_INSTRUCTIONS = {d: v.replace(_CONFLICT_RULE, "") for d, v in PROFILE_REFLECTION_INSTRUCTIONS.items()}


@dataclass
class Variant:
    key: str
    label: str
    model: str
    instructions: dict[str, str]
    guard: bool


VARIANTS = {
    "A": Variant("A", "原始（舊指令 + 120b）", "120b", BASELINE_INSTRUCTIONS, False),
    "B": Variant("B", "第二版指令 + 120b", "120b", V2_INSTRUCTIONS, False),
    "C": Variant("C", "第二版指令 + 20b + 保護", "20b", V2_INSTRUCTIONS, True),
    "D": Variant("D", "目前（窄版矛盾值規則 + 20b + 保護）", "20b", PROFILE_REFLECTION_INSTRUCTIONS, True),
    # 以下三組都含「只記使用者本人的過敏」與 schema forbid（2026-10-18）。
    # 注意 schema 從 tools 匯入，所以 A–D 現在跑也是 forbid 版，跟 10/18 以前的結果不完全可比
    "E": Variant("E", "本人過敏 + forbid + 窄版矛盾值規則", "20b", PROFILE_REFLECTION_INSTRUCTIONS, True),
    "F": Variant("F", "本人過敏 + forbid，無矛盾值規則", "20b", NO_CONFLICT_INSTRUCTIONS, True),
    "G": Variant("G", "E + 推理 low", "20b-low", PROFILE_REFLECTION_INSTRUCTIONS, True),
}


# ──────────────────────────────────────────────────────────────────────────
# 測試案例
# check 格式：(欄位, "has" | "lacks" | "lacks_item", 關鍵字或關鍵字 tuple) 或 (欄位, "eq", 值)
# has：任一筆記錄的該欄位含任一關鍵字；lacks：所有記錄都不含
# lacks_item：所有記錄的該欄位都沒有「完全等於」關鍵字的項目（舊值「辣」被移除，但允許新值「不辣」）
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: str
    category: str
    title: str
    existing: dict[str, dict]          # domain → 既有 content（沒列的 domain 視為沒有記錄）
    turns: list[tuple[str, str]]       # ("user" | "ai", 文字)
    checks: dict[str, list[tuple]]     # domain → checks；只跑有列出的 domain
    note: str = ""


def _diet(**kw):
    base = {"allergies": [], "dislikes": [], "diet_type": None, "flavor_preferences": [],
            "cuisine_preferences": [], "health_goals": []}
    return {**base, **kw}


def _kitchen(**kw):
    return {**{"equipment": [], "skill_level": None, "time_budget": None}, **kw}


def _household(**kw):
    return {**{"members": [], "usual_portions": None}, **kw}


CASES: list[Case] = [
    # ---- 一次性情境：不該寫進檔案 ----
    Case("once_seafood", "一次性情境", "幫朋友煮海鮮",
         {"diet": _diet(flavor_preferences=["清淡"])},
         [("user", "明天朋友要來，他超愛海鮮，幫我想三道海鮮料理"),
          ("ai", "可以做蒜蒸蝦、蛤蜊絲瓜湯、鹽烤鯖魚，三道都不難。"),
          ("user", "好，那就做蒜蒸蝦跟蛤蜊湯，給我步驟")],
         {"diet": [("cuisine_preferences", "lacks", "海鮮"), ("flavor_preferences", "has", "清淡")]}),
    Case("once_vegan_friend", "一次性情境", "朋友吃全素，這次煮素",
         {"diet": _diet()},
         [("user", "這週末有朋友吃全素要來我家，有什麼推薦的菜"),
          ("ai", "可以做麻婆豆腐（不放肉）、香菇炒青江菜、番茄燉鷹嘴豆。")],
         {"diet": [("diet_type", "eq", None)]}),
    Case("once_spicy", "一次性情境", "平常不吃辣，今天想吃辣",
         {"diet": _diet(dislikes=["辣"])},
         [("user", "今天心情不好，想吃辣一點的，來個麻辣鍋的做法"),
          ("ai", "麻辣鍋湯底：牛油、花椒、乾辣椒、豆瓣醬……")],
         {"diet": [("dislikes", "has", "辣")]}),
    Case("once_busy", "一次性情境", "這週加班，先要 15 分鐘的菜",
         {"kitchen": _kitchen(time_budget="1小時內")},
         [("user", "這週加班很忙，這幾天先給我 15 分鐘能做好的晚餐"),
          ("ai", "可以做番茄炒蛋、蒜炒高麗菜、蔥爆牛肉，都在 15 分鐘內。")],
         {"kitchen": [("time_budget", "has", ("1小時", "一小時", "60")),
                      ("time_budget", "lacks", "15")]}),
    Case("once_borrowed", "一次性情境", "借朋友的氣炸鍋用一次",
         {"kitchen": _kitchen(equipment=["電鍋"])},
         [("user", "今天跟朋友借了一台氣炸鍋，用完就要還他，有什麼氣炸鍋料理"),
          ("ai", "氣炸雞翅、氣炸地瓜條、氣炸鮭魚都很適合。")],
         {"kitchen": [("equipment", "lacks", "氣炸鍋"), ("equipment", "has", "電鍋")]}),
    Case("once_guests", "一次性情境", "請 6 個朋友來吃飯",
         {"household": _household(usual_portions=2)},
         [("user", "週六要請 6 個朋友來家裡吃飯，幫我規劃菜單"),
          ("ai", "6 人份可以準備：滷肉、三杯雞、炒時蔬、玉米濃湯。")],
         {"household": [("usual_portions", "eq", 2), ("members", "lacks", "朋友")]}),
    Case("once_colleague_allergy", "一次性情境", "同事對花生過敏，明天來吃飯",
         {"diet": _diet(), "household": _household()},
         [("user", "明天同事要來吃飯，他對花生過敏，菜裡要注意什麼"),
          ("ai", "避開宮保雞丁、花生醬拌麵，也要確認醬料沒有花生成分。")],
         {"diet": [("allergies", "lacks", "花生")],
          "household": [("members", "lacks", ("同事", "花生"))]}),

    # ---- 長期改變：應該更新 ----
    Case("perm_vegetarian", "長期改變", "從這個月開始吃素",
         {"diet": _diet()},
         [("user", "我從這個月開始吃素了，以後都幫我推薦素食料理就好"),
          ("ai", "好的，之後都以素食為主。")],
         {"diet": [("diet_type", "has", ("vegetarian", "素"))]}),
    Case("perm_coriander", "長期改變", "討厭香菜，以後都不要放",
         {"diet": _diet()},
         [("user", "我很討厭香菜，以後所有料理都不要放香菜"),
          ("ai", "了解，之後的食譜都不會放香菜。")],
         {"diet": [("dislikes", "has", "香菜")]}),
    Case("perm_low_salt", "長期改變", "重口味改清淡（醫生要求）",
         {"diet": _diet(flavor_preferences=["重口味"])},
         [("user", "最近健檢血壓偏高，醫生要我以後都吃清淡、少鹽"),
          ("ai", "好的，之後會以清淡少鹽為主。")],
         {"diet": [("flavor_preferences", "has", "清淡"), ("flavor_preferences", "lacks", "重口味")]}),
    Case("perm_new_job", "長期改變", "換工作，以後平日只有 20 分鐘",
         {"kitchen": _kitchen(time_budget="1小時內")},
         [("user", "我換工作了，以後平日下班只剩 20 分鐘能煮飯"),
          ("ai", "了解，平日會推薦 20 分鐘內能完成的料理。")],
         {"kitchen": [("time_budget", "has", "20")]}),
    Case("perm_bought", "長期改變", "買了一台氣炸鍋",
         {"kitchen": _kitchen(equipment=["電鍋"])},
         [("user", "我昨天買了一台氣炸鍋，之後想多用它"),
          ("ai", "太好了，之後會多推薦氣炸鍋料理。")],
         {"kitchen": [("equipment", "has", "氣炸鍋"), ("equipment", "has", "電鍋")]}),
    Case("perm_daughter", "長期改變", "女兒最近開始不吃辣",
         {"household": _household()},
         [("user", "我女兒最近開始不吃辣了，之後煮飯要注意一下"),
          ("ai", "好的，之後會提供不辣的版本。")],
         {"household": [("members", "has", "女兒")]}),
    Case("perm_repeated", "長期改變", "連續三次都選清淡（隱含偏好）",
         {"diet": _diet()},
         [("user", "晚餐想吃什麼好？"), ("ai", "可以選：A 清蒸魚、B 麻婆豆腐、C 紅燒肉。"),
          ("user", "清蒸魚好了，我比較喜歡清淡的"), ("ai", "那明天午餐：A 涼拌雞絲、B 咖哩飯。"),
          ("user", "涼拌雞絲，一樣清淡一點"), ("ai", "湯品要哪個：A 冬瓜湯、B 酸辣湯。"),
          ("user", "冬瓜湯")],
         {"diet": [("flavor_preferences", "has", "清淡")]}),

    # ---- 矛盾值替換：長期改變時，舊值要拿掉，不能新舊並存；無關的舊值要留著 ----
    # 案例刻意跟指令裡的舉例不同，避免只是背答案
    Case("conflict_beef", "矛盾值替換", "以前不吃牛，現在開始吃了",
         {"diet": _diet(dislikes=["牛肉", "香菜"])},
         [("user", "我以前不吃牛肉，最近開始吃了，之後可以推薦牛肉料理"),
          ("ai", "好的，之後會加入牛肉料理。")],
         {"diet": [("dislikes", "lacks_item", "牛肉"), ("dislikes", "has", "香菜")]}),
    Case("conflict_spicy", "矛盾值替換", "原本愛吃辣，醫生要我以後不能吃辣",
         {"diet": _diet(flavor_preferences=["辣", "鹹香"])},
         [("user", "我有胃食道逆流，醫生要我以後都不能吃辣"),
          ("ai", "了解，之後的料理都會做成不辣的版本。")],
         {"diet": [("flavor_preferences", "lacks_item", ("辣", "重辣", "麻辣")),
                   ("flavor_preferences", "has", "鹹香")]}),
    Case("conflict_omnivore", "矛盾值替換", "不吃素了，現在葷素都吃",
         {"diet": _diet(diet_type="vegetarian", dislikes=["苦瓜"])},
         [("user", "我已經沒有吃素了，現在葷素都吃"),
          ("ai", "好的，之後葷食料理也會推薦。")],
         {"diet": [("diet_type", "lacks", ("vegetarian", "素食", "vegan")),
                   ("dislikes", "has", "苦瓜")]}),
    Case("conflict_oven", "矛盾值替換", "烤箱壞了丟掉，不會再買",
         {"kitchen": _kitchen(equipment=["電鍋", "烤箱"])},
         [("user", "家裡的烤箱壞了，已經丟掉，之後也不打算再買"),
          ("ai", "了解，之後不會推薦需要烤箱的料理。")],
         {"kitchen": [("equipment", "lacks_item", "烤箱"), ("equipment", "has", "電鍋")]}),
    Case("conflict_daughter", "矛盾值替換", "女兒長大了，現在可以吃辣",
         {"household": _household(members=["女兒不吃辣", "老婆不吃牛"], usual_portions=3)},
         [("user", "女兒現在長大了，辣的也敢吃了，以後不用特別幫她做不辣的"),
          ("ai", "好的，之後不用另外準備不辣的版本。")],
         {"household": [("members", "lacks", "女兒不吃辣"), ("members", "has", "老婆")]}),

    # ---- 過敏原安全：背景反思不能移除 ----
    Case("allergy_own_new", "過敏原安全", "使用者說自己對奇異果過敏",
         {"diet": _diet()},
         [("user", "跟你說一下，我對奇異果過敏，吃了嘴巴會癢，甜點不要放"),
          ("ai", "了解，之後甜點都會避開奇異果。")],
         {"diet": [("allergies", "has", "奇異果")]},
         note="「只記本人過敏」的規則不能讓本人的過敏也漏記"),
    Case("allergy_cook_for_friend", "過敏原安全", "對蝦過敏，幫朋友煮胡椒蝦",
         {"diet": _diet(allergies=["蝦"])},
         [("user", "朋友很愛吃蝦，今天幫他做胡椒蝦，我自己不吃，給我步驟"),
          ("ai", "胡椒蝦：蝦子剪鬚、熱鍋下奶油與蒜末……處理時記得戴手套。")],
         {"diet": [("allergies", "has", "蝦")]}),
    Case("allergy_cured", "過敏原安全", "說蝦過敏好了（政策：背景反思也不移除）",
         {"diet": _diet(allergies=["蝦"])},
         [("user", "醫生說我的蝦過敏已經好了，之後可以正常吃蝦了"),
          ("ai", "太好了，恭喜！")],
         {"diet": [("allergies", "has", "蝦")]},
         note="依新政策，移除過敏原要走 diet_profile_manage 並經人工審核，背景反思一律保留"),
]


# ──────────────────────────────────────────────────────────────────────────

def _messages(turns):
    # 正式流程裡使用者文字包在 <user_input>（見 app.py 組訊息的步驟）
    return [HumanMessage(f"<user_input>{t}</user_input>") if who == "user" else AIMessage(t)
            for who, t in turns]


def _contains(value, keywords) -> bool:
    keywords = keywords if isinstance(keywords, tuple) else (keywords,)
    items = value if isinstance(value, list) else [value]
    return any(isinstance(v, str) and k.lower() in v.lower() for v in items for k in keywords)


def _equals_item(value, keywords) -> bool:
    keywords = keywords if isinstance(keywords, tuple) else (keywords,)
    items = value if isinstance(value, list) else [value]
    return any(isinstance(v, str) and v.strip().lower() == k.lower() for v in items for k in keywords)


def _evaluate(records: list[dict], checks: list[tuple]) -> list[dict]:
    out = []
    for field_name, op, *arg in checks:
        values = [r.get(field_name) for r in records] or [None]
        if op == "has":
            ok = any(_contains(v, arg[0]) for v in values)
        elif op == "lacks":
            ok = not any(_contains(v, arg[0]) for v in values)
        elif op == "lacks_item":
            ok = not any(_equals_item(v, arg[0]) for v in values)
        else:  # eq
            ok = all(v == arg[0] for v in values)
        out.append({"check": f"{field_name} {op} {arg[0] if arg else ''}".strip(), "ok": ok})
    return out


async def _with_retry(coro_factory, label: str):
    for attempt in range(8):
        try:
            return await coro_factory()
        except Exception as exc:  # 429 以外的錯誤直接記錄
            text = f"{type(exc).__name__}: {exc}"
            if "429" not in text and "rate_limit" not in text.lower():
                raise
            wait = 20 + attempt * 10
            print(f"    ⏳ {label} 被限流，{wait} 秒後重試")
            await asyncio.sleep(wait)
    raise RuntimeError(f"{label} 重試 8 次仍被限流")


async def run_one(variant: Variant, case: Case, model, usage: UsageMetadataCallbackHandler) -> dict:
    inner = InMemoryStore()
    store = ProfileGuardStore(inner, _PROFILE_DOMAINS) if variant.guard else inner
    config = {"configurable": {"user_id": USER_ID}, "callbacks": [usage]}
    result = {"variant": variant.key, "case": case.id, "domains": {}}
    for domain, checks in case.checks.items():
        ns = tuple(p.replace("{user_id}", USER_ID) for p in _profile_ns(domain))
        if domain in case.existing:
            await inner.aput(ns, "profile", {"kind": _PROFILE_DOMAINS[domain].__name__,
                                            "content": case.existing[domain]})
        mgr = create_memory_store_manager(
            model, namespace=_profile_ns(domain), schemas=[_PROFILE_DOMAINS[domain]],
            instructions=variant.instructions[domain], store=store, enable_deletes=True,
        )
        started = time.perf_counter()
        try:
            await _with_retry(lambda: mgr.ainvoke({"messages": _messages(case.turns)}, config=config),
                              f"{variant.key}/{case.id}/{domain}")
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        records = [it.value.get("content", {}) for it in await inner.asearch(ns, limit=20)]
        # 執行失敗（例如每日額度用完：LangMem 內部會吞掉 429 一直重試到 GraphRecursionError）
        # 不算進分數，另外列出次數，避免把額度問題當成指令變差
        result["domains"][domain] = {
            "records": records,
            "checks": _evaluate(records, checks) if error is None else [],
            "error": error,
            "seconds": round(time.perf_counter() - started, 1),
        }
    return result


async def run_lane(variants: list[Variant], reps: int, usages: dict) -> list[dict]:
    """同一個模型的設定排在同一條線依序跑（同模型共用每分鐘 token 額度）；不同模型的線並行。"""
    model = MODELS[variants[0].model]()
    rows = []
    for rep in range(1, reps + 1):
        for case in CASES:
            for v in variants:  # 同一案例 A、B 緊接著跑，時間因素對兩者一致
                row = await run_one(v, case, model, usages[v.key])
                row["rep"] = rep
                ok = sum(c["ok"] for d in row["domains"].values() for c in d["checks"])
                total = sum(len(d["checks"]) for d in row["domains"].values())
                errors = [d["error"] for d in row["domains"].values() if d["error"]]
                status = f"執行失敗：{errors[0][:80]}" if errors else f"{ok}/{total}"
                print(f"  [{v.key}] 第{rep}輪 {case.title:<22} {status}")
                rows.append(row)
    return rows


def _report(rows: list[dict], variant_keys: list[str], usages: dict) -> dict:
    def score(filter_fn):
        out = {}
        for k in variant_keys:
            checks = [c for r in rows if r["variant"] == k and filter_fn(r)
                      for d in r["domains"].values() for c in d["checks"]]
            out[k] = (sum(c["ok"] for c in checks), len(checks))
        return out

    head = "".join(f"{VARIANTS[k].key:>14}" for k in variant_keys)
    print("\n" + "=" * 78)
    for k in variant_keys:
        print(f"  {k} = {VARIANTS[k].label}")
    print("=" * 86)
    print(f"{'案例':<30}{head}")
    by_case = {}
    for case in CASES:
        s = score(lambda r, cid=case.id: r["case"] == cid)
        by_case[case.id] = s
        cells = "".join(f"{f'{a}/{b}':>14}" for a, b in s.values())
        print(f"{case.title:<30}{cells}")
    print("-" * 86)
    by_cat = {}
    for cat in dict.fromkeys(c.category for c in CASES):
        ids = {c.id for c in CASES if c.category == cat}
        s = score(lambda r, ids=ids: r["case"] in ids)
        by_cat[cat] = s
        cells = "".join(f"{f'{a}/{b} ({a / max(b, 1):.0%})':>14}" for a, b in s.values())
        print(f"{cat:<30}{cells}")
    total = score(lambda r: True)
    print(f"{'總計':<30}" + "".join(f"{f'{a}/{b} ({a / max(b, 1):.0%})':>14}" for a, b in total.values()))
    failed = {k: sum(1 for r in rows if r["variant"] == k for d in r["domains"].values() if d["error"])
              for k in variant_keys}
    print(f"{'執行失敗（不計分）':<30}" + "".join(f"{n:>14}" for n in failed.values()))

    print("\ntoken 用量（含 LangMem 內部的所有模型呼叫）")
    tokens = {}
    for k in variant_keys:
        u = usages[k].usage_metadata
        inp = sum(m.get("input_tokens", 0) for m in u.values())
        outp = sum(m.get("output_tokens", 0) for m in u.values())
        n = sum(1 for r in rows if r["variant"] == k for _ in r["domains"])
        tokens[k] = {"input": inp, "output": outp, "per_call": round((inp + outp) / max(n, 1))}
        print(f"  {k}: 輸入 {inp:,}、輸出 {outp:,}，平均每次反思 {tokens[k]['per_call']:,}")
    return {"by_case": by_case, "by_category": by_cat, "total": total, "tokens": tokens}


async def main(reps: int, variant_keys: list[str], case_ids: list[str] | None = None):
    if case_ids:
        CASES[:] = [c for c in CASES if c.id in case_ids]
    usages = {k: UsageMetadataCallbackHandler() for k in variant_keys}
    lanes: dict[str, list[Variant]] = {}
    for k in variant_keys:
        lanes.setdefault(VARIANTS[k].model, []).append(VARIANTS[k])
    print(f"案例 {len(CASES)} 個 × 設定 {variant_keys} × {reps} 輪")
    results = await asyncio.gather(*(run_lane(vs, reps, usages) for vs in lanes.values()))
    rows = [r for lane in results for r in lane]
    summary = _report(rows, variant_keys, usages)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"reflection_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"variants": {k: VARIANTS[k].label for k in variant_keys},
                   "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n✔ 明細已存：{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=2, help="每個案例跑幾輪（看結果穩不穩定）")
    parser.add_argument("--variants", default="A,B,C,D")
    parser.add_argument("--cases", default="", help="只跑這些案例 id（逗號分隔），預設全部")
    parser.add_argument("--key20b", default="GROQ_API_KEY3",
                        help="20b 用哪個環境變數的 key（每把 key 每個模型每天各有額度，用完可換一把）")
    args = parser.parse_args()
    os.environ["EVAL_KEY_20B"] = args.key20b
    asyncio.run(main(args.reps, [v.strip() for v in args.variants.split(",")],
                     [c.strip() for c in args.cases.split(",") if c.strip()]))
