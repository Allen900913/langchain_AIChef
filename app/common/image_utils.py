"""
圖片工具模組 - 提供 Base64 圖片編碼功能
由於沒有 OSS 服務，圖片透過 Base64 編碼直接傳送給模型
"""
import base64
from typing import Optional


def encode_image_to_base64_url(image_bytes: bytes, content_type: str = "image/png") -> str:
    """
    將圖片的 bytes 轉換為 Base64 Data URL，可直接傳給 LLM 的 image_url。

    Args:
        image_bytes: 圖片的原始 bytes 資料
        content_type: 圖片的 MIME 型別，例如 "image/png", "image/jpeg", "image/webp"

    Returns:
        Base64 Data URL 字串，格式為 "data:{content_type};base64,{encoded_data}"
    """
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def get_content_type(filename: str) -> str:
    """
    根據檔案副檔名判斷 Content-Type

    Args:
        filename: 檔案名稱

    Returns:
        對應的 MIME 型別
    """
    content_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
    }
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    return content_type_map.get(ext, "image/png")
