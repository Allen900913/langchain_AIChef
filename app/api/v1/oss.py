import os
from datetime import timedelta
from fastapi import APIRouter
from google.cloud import storage

# /api/v1
router = APIRouter()

# 從環境變數獲取你的 Bucket 名稱
GCS_BUCKET = os.getenv("GCS_BUCKET")

# 初始化 GCS 客戶端
# 注意：這裡會自動抓取系統環境變數 GOOGLE_APPLICATION_CREDENTIALS 所指向的 JSON 金鑰路徑
client = storage.Client()

@router.get("/gcs/presign")
def chat_endpoint(filename: str):
    # 根據檔案副檔名判斷 Content-Type
    content_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
    content_type = content_type_map.get(ext, "application/octet-stream")

    # 取得 Bucket 與 Blob (檔案物件) 例項
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(filename)

    # 產生預先簽名的上傳 URL (PUT 請求)
    upload_url = blob.generate_signed_url(
        version="v4", # 強烈建議使用 v4 簽名機制
        expiration=timedelta(seconds=3600), # 有效期 1 小時
        method="PUT", # 設定為上傳
        content_type=content_type, # 必須與前端上傳時的 Content-Type 完全一致
    )

    # 👇 2. 新增：產生讓 LLM 模型「讀取」用的 URL (帶有 GET 簽名)
    read_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=3600), # 1小時內 LLM 都可以讀取
        method="GET", # 👈 注意這裡是 GET
    )

    # 返回這兩個 URL
    return {
        "uploadUrl": upload_url,
        "contentType": content_type,
        # 👇 3. 將原本的公開網址，換成這把帶有讀取簽名的網址
        "accessUrl": read_url 
    }