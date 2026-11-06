"""
圖片工具模組：檔案類型判斷、Base64 Data URL 編碼、訊息附圖網址檢查。

檔案類型一律看檔頭（magic bytes），不看副檔名或瀏覽器送的 Content-Type：
那兩個都是使用者可以隨便填的，例如把 HTML 改名成 .png 上傳。
"""
import base64
import binascii
import re
from urllib.parse import urlsplit

# 登入使用者上傳到 GCS 的上限（簽名 URL 帶 x-goog-content-length-range，GCS 會擋超過的）
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# 訪客圖片轉成 Base64 直接放進請求送給 Groq，Groq 對「含 Base64 圖片的請求」上限 4MB，
# Base64 會膨脹 4/3 倍，原圖 2.5MB → 約 3.3MB，加上提示詞仍在上限內
MAX_DATA_URL_IMAGE_BYTES = int(2.5 * 1024 * 1024)
MAX_DATA_URL_LEN = 64 + (MAX_DATA_URL_IMAGE_BYTES + 2) // 3 * 4

ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
GCS_HOST = "storage.googleapis.com"

_DATA_URL_RE = re.compile(r"^data:(image/(?:jpeg|png|gif|webp));base64,")


def sniff_image_type(head: bytes) -> str | None:
    """依檔頭判斷圖片類型，只認 JPEG / PNG / GIF / WebP；其他（含 SVG、HTML）回傳 None。"""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def encode_image_to_base64_url(image_bytes: bytes, content_type: str) -> str:
    """將圖片的 bytes 轉換為 Base64 Data URL，可直接傳給 LLM 的 image_url。"""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def check_message_image_url(url: str, user_id: str | None, bucket: str | None) -> str | None:
    """檢查訊息附帶的圖片網址；合格回傳 None，不合格回傳要給使用者看的錯誤訊息。

    這個網址會原封不動交給視覺模型去讀，所以只接受我們自己產生的兩種：
    - Base64 Data URL（/chat/upload-image 產生）：宣告的類型和實際檔頭都必須是允許的圖片
    - 我們 GCS bucket 裡、這個使用者自己目錄（uploads/<user_id>/）底下的簽名 URL
      （/gcs/presign 產生）。user_id 為 None（訪客）時不接受，訪客不能上傳到雲端
    其他任意網址一律拒絕，避免把服務當成「替任何人去抓任何網址」的跳板。
    """
    if url.startswith("data:"):
        m = _DATA_URL_RE.match(url)
        if not m:
            return "只接受 jpg / png / gif / webp 圖片"
        if len(url) > MAX_DATA_URL_LEN:
            return f"圖片大小不可超過 {MAX_DATA_URL_IMAGE_BYTES / 1024 / 1024:g}MB"
        try:
            head = base64.b64decode(url[m.end():m.end() + 24], validate=True)
        except (binascii.Error, ValueError):
            return "圖片資料格式錯誤"
        if sniff_image_type(head) != m.group(1):
            return "圖片內容與宣告的類型不符"
        return None

    parts = urlsplit(url)
    if (
        user_id is None
        or not bucket
        or parts.scheme != "https"
        or parts.hostname != GCS_HOST
        or parts.port is not None
        or ".." in parts.path
        or not parts.path.startswith(f"/{bucket}/uploads/{user_id}/")
    ):
        return "圖片網址無效，請重新上傳"
    return None
