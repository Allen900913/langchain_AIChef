"""訊息判讀小工具：API 伺服器與 agent worker 共用。

刻意只依賴 langchain_core——API 伺服器讀歷史時會用到，不能為此 import 整個 agent
（模型、fastembed、E2B 等重依賴只該出現在 worker）。
"""

import re

from langchain_core.messages import HumanMessage

# call_agent 包裝的 <user_input> 標籤與 vision 附加的 <tool_output> 區塊
_USER_INPUT_TAG_RE = re.compile(r"</?user_input>")
_TOOL_OUTPUT_SECTION_RE = re.compile(r"<tool_output>.*?</tool_output>", re.DOTALL)


# 不可信的文字（使用者訊息、圖片辨識結果、搜尋結果、偏好內容）包進標籤之前，先把看起來像標籤的
# 字串「<」換成全形「＜」：模型讀起來意思不變，但它不再是標籤，關不掉外層的 <user_input>，
# 也冒充不了 <system>。涵蓋我們自己用的分隔標籤、常見的角色標記，以及 <|im_start|> 這類特殊記號。
_SPOOFABLE_TAG_RE = re.compile(
    r"<(?=\s*/?\s*(?:user_input|tool_output|user_profile|system|developer|assistant|user|tool)\b)|<(?=\|)",
    re.IGNORECASE,
)


def neutralize_tags(text: str) -> str:
    """讓不可信文字裡的標籤失效（「<system>」→「＜system>」），其他內容原樣保留。"""
    return _SPOOFABLE_TAG_RE.sub("＜", text)


def is_summary_msg(msg) -> bool:
    """判斷一則訊息是否為 SummarizationMiddleware 產生的摘要訊息。"""
    return (
        isinstance(msg, HumanMessage)
        and msg.additional_kwargs.get("lc_source") == "summarization"
    )


# SummarizationMiddleware 組摘要訊息時加的固定開頭（langchain/agents/middleware/summarization.py）
_SUMMARY_PREFIX = "Here is a summary of the conversation to date:"


def summary_text(msg) -> str:
    """摘要訊息去掉 LangChain 加的英文開頭，只留摘要本文。"""
    content = msg.content if isinstance(msg.content, str) else ""
    return content.removeprefix(_SUMMARY_PREFIX).strip()


def clean_user_text(content: str) -> str:
    """去除 <user_input> 包裝與 <tool_output> 附加區塊，還原使用者原話。"""
    content = _USER_INPUT_TAG_RE.sub("", content)
    return _TOOL_OUTPUT_SECTION_RE.sub("", content).strip()
