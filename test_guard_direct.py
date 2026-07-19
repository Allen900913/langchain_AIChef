import asyncio
from app.agents.app import _find_recommended_allergen

async def test():
    allergens = ["花生", "蝦"]
    # Test 1: Recommend peanut
    resp1 = "您可以試試看花生厚片吐司，香濃好吃！"
    hit1 = await _find_recommended_allergen(allergens, resp1)
    print(f"Test 1 (Recommend peanut): hit={hit1}")

    # Test 2: Safe
    resp2 = "這是一道不辣的雞胸肉料理，使用氣炸鍋，減脂、30分鐘內完成。"
    hit2 = await _find_recommended_allergen(allergens, resp2)
    print(f"Test 2 (Safe): hit={hit2}")

    # Test 3: Avoid peanut
    resp3 = "因為您對花生過敏，所以我幫您換成了無花果醬吐司，絕對沒有花生！"
    hit3 = await _find_recommended_allergen(allergens, resp3)
    print(f"Test 3 (Avoid peanut): hit={hit3}")

if __name__ == "__main__":
    asyncio.run(test())
