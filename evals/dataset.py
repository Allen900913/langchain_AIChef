"""評測資料集。

每個 case 描述一次完整對話輪，欄位：
- id           : 唯一識別
- query        : 使用者輸入文字
- user_id      : 該 case 使用的使用者（用於隔離 profile / inventory）
- seed_diet    : 跑案前要寫入該 user diet profile 的內容（dict）；None 表不種
- seed_inventory: 跑案前要寫入該 user 冰箱的食材清單（list[str]）；None 表不種
- expected_tools: 這個 query「理應」呼叫到的工具集合（算 Tool Selection Accuracy 用）
- allergens    : 該 user 已知過敏原；最終回答若推薦含這些食材即算違反（算過敏原違反率用）

備註：Faithfulness / ContextRelevance 需要 retrieved_contexts，只有實際呼叫 web_search 的
case 才會有；純查冰箱 / 純算營養的 case 沒有檢索內容，這兩項指標會被標為 N/A 而非 0。
"""
from __future__ import annotations

from typing import Any

EVAL_CASES: list[dict[str, Any]] = [
    {
        "id": "rec_allergen_peanut",
        "query": "幫我推薦一道今晚可以做的家常菜",
        "user_id": "eval_peanut",
        "seed_diet": {"allergies": ["花生", "花生醬"], "dislikes": [], "diet_type": None},
        "seed_inventory": ["雞胸肉", "青椒", "洋蔥", "白米"],
        "expected_tools": ["profiles_get", "inventory_get", "web_search"],
        "allergens": ["花生"],
    },
    {
        "id": "rec_vegetarian",
        "query": "我吃素，推薦一道有蛋白質的午餐",
        "user_id": "eval_veg",
        "seed_diet": {"allergies": [], "dislikes": [], "diet_type": "vegetarian"},
        "seed_inventory": ["豆腐", "花椰菜", "番茄", "雞蛋"],
        "expected_tools": ["profiles_get", "inventory_get", "web_search"],
        "allergens": ["豬肉", "牛肉", "雞肉", "鮭魚"],
    },
    {
        "id": "nutrition_chicken",
        "query": "雞胸肉每100克的熱量跟蛋白質大概多少？",
        "user_id": "eval_nutri",
        "seed_diet": None,
        "seed_inventory": None,
        "expected_tools": ["nutrition_lookup"],
        "allergens": [],
    },
    {
        "id": "inventory_query",
        "query": "我冰箱現在有什麼食材？",
        "user_id": "eval_inv",
        "seed_diet": None,
        "seed_inventory": ["雞蛋", "高麗菜", "豆腐"],
        "expected_tools": ["inventory_get"],
        "allergens": [],
    },
    {
        "id": "shopping_list",
        "query": "我想做番茄炒蛋，需要番茄、雞蛋、蔥、鹽，幫我看還缺什麼要買",
        "user_id": "eval_shop",
        "seed_diet": None,
        "seed_inventory": ["雞蛋", "鹽"],
        "expected_tools": ["shopping_list_generate"],
        "allergens": [],
    },
    {
        "id": "learn_recipe_steps",
        "query": "教我怎麼做蒜香雞胸肉，給我步驟",
        "user_id": "eval_steps",
        "seed_diet": None,
        "seed_inventory": ["雞胸肉", "蒜頭"],
        "expected_tools": ["web_search", "step_tracker_start"],
        "allergens": [],
    },
    {
        "id": "rec_seafood_allergy",
        "query": "推薦一道適合我的晚餐，要簡單一點",
        "user_id": "eval_seafood",
        "seed_diet": {"allergies": ["蝦", "螃蟹", "帶殼海鮮"], "dislikes": [], "diet_type": None},
        "seed_inventory": ["雞蛋", "番茄", "洋蔥", "白米"],
        "expected_tools": ["profiles_get", "inventory_get", "web_search"],
        "allergens": ["蝦", "螃蟹"],
    },
    {
        "id": "nutrition_unknown",
        "query": "酪梨的營養成分是多少？",
        "user_id": "eval_avocado",
        "seed_diet": None,
        "seed_inventory": None,
        # 本地營養庫沒有酪梨，理應 nutrition_lookup 查不到後改用 web_search 補
        "expected_tools": ["nutrition_lookup", "web_search"],
        "allergens": [],
    },
]
