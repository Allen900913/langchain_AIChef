# -*- coding: utf-8 -*-
"""E2B Code Interpreter 連線測試腳本"""

import os
import sys

# 強制 stdout 使用 utf-8 編碼（Windows 環境需要）
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox

load_dotenv()

api_key = os.environ.get("E2B_API_KEY", "")
print(f"API Key 前 10 碼: {api_key[:10]}...")
print("正在啟動 E2B 沙箱...")

with Sandbox.create() as sandbox:
    execution = sandbox.run_code("""
import sys
import math
print(f"沙箱內的 Python 版本: {sys.version}")
print(f"計算結果: pi * 2 = {math.pi * 2}")
print("E2B 沙箱連線成功!")
""")

    print("--- 沙箱輸出 ---")

    # 實測：此版本 execution.text 回傳 None，
    # 真正的 stdout 在 execution.logs.stdout（list[str]）
    stdout = "".join(execution.logs.stdout)
    print(stdout)

    if execution.error:
        print(f"錯誤: {execution.error.name} - {execution.error.value}")

print("沙箱已自動關閉，測試完成 ✅")
