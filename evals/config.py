"""評測用的 judge LLM 與 embeddings 建構（全程 NVIDIA，但與受測模型隔離）。

⚠️ 這個模組必須在 import 任何 ragas 之前先被 import，因為它會注入
   `langchain_community.chat_models.vertexai` 的相容 shim。
"""
from __future__ import annotations

import os
import sys
import types

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────
# ragas 0.4.3 在 import 階段硬寫 `from langchain_community.chat_models.vertexai
# import ChatVertexAI`，但新版（sunset 的）langchain-community 0.4.x 已移除該模組。
# 我們用 NVIDIA、根本不碰 vertexai，所以塞一個 stub 讓 import 通過即可。
# 放在這裡（import ragas 之前）注入，不動 site-packages、重裝套件也不會被洗掉。
# ──────────────────────────────────────────────────────────────────────────
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        _shim = types.ModuleType("langchain_community.chat_models.vertexai")
        _shim.ChatVertexAI = type("ChatVertexAI", (), {})  # stub，永不使用
        sys.modules["langchain_community.chat_models.vertexai"] = _shim

from openai import AsyncOpenAI  # noqa: E402

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv("NVIDA_API_KEY")  # 註：env 變數名沿用專案既有拼字 NVIDA_API_KEY

# judge：受測 agent 用 qwen3-next-80b，這裡刻意換不同模型避免自己改自己考卷
JUDGE_MODEL = "meta/llama-3.3-70b-instruct"
# embeddings：baai/bge-m3 為 symmetric 模型，NVIDIA OpenAI 相容端點不需 input_type（nv-embedqa 需要）
EMBED_MODEL = "baai/bge-m3"

# 各指標的及格門檻（0~1），run_eval 以此標記 PASS/FAIL 並算通過率
THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "context_relevance": 0.50,
    "tool_selection_accuracy": 0.80,
    "allergen_violation_rate": 0.0,  # 過敏原違反率：越低越好，>0 即視為不及格
    "task_success_rate": 0.80,
}

# NVIDIA NIM 計費（USD / 每 1M tokens）。預設 None＝不知道你的實際合約價，
# 報表只會列出 token 用量，不會編造金額；要換算成本時自行填入這兩個數字即可。
COST_PER_1M_INPUT_TOKENS: float | None = None
COST_PER_1M_OUTPUT_TOKENS: float | None = None


def _nvidia_client() -> AsyncOpenAI:
    if not NVIDIA_API_KEY:
        raise RuntimeError("缺少環境變數 NVIDA_API_KEY，無法建立 judge / embeddings client")
    # 沒設 timeout 會吃 openai SDK 預設的 600 秒；跟 app.py 主模型的 request_timeout=50.0
    # 對齊，避免單次 judge / embeddings 呼叫卡住拖慢整批 eval。
    return AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=60.0)


def build_judge_and_embeddings():
    """回傳 (judge_llm, embeddings)，皆為 ragas 新版 collections API 所需型別。"""
    from ragas.embeddings import OpenAIEmbeddings as RagasEmbeddings
    from ragas.llms import llm_factory

    client = _nvidia_client()
    judge = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    embeddings = RagasEmbeddings(client=client, model=EMBED_MODEL)
    return judge, embeddings
