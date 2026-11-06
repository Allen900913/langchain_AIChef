import os
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from google.cloud import storage

from app.api.deps import Principal, require_user
from app.common.image_utils import MAX_IMAGE_BYTES

router = APIRouter()


def get_gcs_client() -> storage.Client:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        # 若為相對路徑，轉為相對於專案根目錄的絕對路徑
        if not os.path.isabs(cred_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            candidate = os.path.join(base_dir, cred_path)
            if os.path.exists(candidate):
                cred_path = candidate
        if os.path.exists(cred_path):
            return storage.Client.from_service_account_json(cred_path)
    return storage.Client()


@router.get("/gcs/presign")
def chat_endpoint(filename: str, principal: Principal = Depends(require_user)):
    """只給登入使用者（訪客不留任何檔案，改走 /chat/upload-image 的 Base64）。

    filename 只用來判斷副檔名；物件路徑由伺服器產生（uploads/<user_id>/<隨機>.<副檔名>），
    使用者無法指定路徑去覆寫別人的檔案。

    上傳 URL 綁了 Content-Type 與檔案大小範圍（x-goog-content-length-range），兩者都納入簽章：
    前端 PUT 時必須原樣帶上 uploadHeaders，超過大小 GCS 直接拒絕，不會有人拿這把 URL 塞大檔進 bucket。
    """
    gcs_bucket = os.getenv("GCS_BUCKET")
    if not gcs_bucket:
        raise HTTPException(
            status_code=500,
            detail="GCS_BUCKET 環境變數尚未設定，請在 .env 中填寫 GCS Bucket 名稱",
        )
    # 根據檔案副檔名判斷 Content-Type
    content_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
    if ext not in content_type_map:
        raise HTTPException(status_code=400, detail="只接受 jpg / png / gif / webp 圖片")
    content_type = content_type_map[ext]
    object_name = f"uploads/{principal.id}/{uuid.uuid4().hex}.{ext}"

    # 取得 Bucket 與 Blob (檔案物件) 例項
    client = get_gcs_client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(object_name)

    # 產生預先簽名的上傳 URL (PUT 請求)
    size_range = f"0,{MAX_IMAGE_BYTES}"
    upload_url = blob.generate_signed_url(
        version="v4", # 強烈建議使用 v4 簽名機制
        expiration=timedelta(minutes=10), # 前端拿到就立刻上傳，不需要長效期
        method="PUT", # 設定為上傳
        content_type=content_type, # 必須與前端上傳時的 Content-Type 完全一致
        headers={"x-goog-content-length-range": size_range},
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
        # 前端 PUT 時要原樣帶上（都在簽章裡，少帶或改值都會被 GCS 拒絕）
        "uploadHeaders": {"Content-Type": content_type, "x-goog-content-length-range": size_range},
        # 👇 3. 將原本的公開網址，換成這把帶有讀取簽名的網址
        "accessUrl": read_url 
    }