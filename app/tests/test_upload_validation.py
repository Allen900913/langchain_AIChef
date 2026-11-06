"""測試：圖片上傳與訊息附圖網址的檢查（app/common/image_utils.py、app/api/v1/chat.py、app/api/v1/oss.py）。

測試項目：
1. sniff_image_type：只認 JPEG / PNG / GIF / WebP 的檔頭，SVG、HTML 一律不認
2. check_message_image_url：只接受合格的 Data URL，或自己 GCS 目錄底下的簽名 URL
3. /chat/upload-image：類型看檔頭不看副檔名、超過大小回 413；/chat/tasks 帶不合格網址回 400
4. （加 --gcs 才跑，會真的上傳到 GCS）簽名 URL 綁了大小範圍：照 uploadHeaders 上傳成功、
   少帶 header 被拒、超過 10MB 被拒；產生的讀取 URL 通得過 check_message_image_url

執行方式：python app/tests/test_upload_validation.py [--gcs]
"""

import asyncio
import base64
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from app.api import deps  # noqa: E402
from app.api.v1 import chat  # noqa: E402
from app.common.image_utils import (  # noqa: E402
    MAX_DATA_URL_IMAGE_BYTES,
    MAX_IMAGE_BYTES,
    check_message_image_url,
    sniff_image_type,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 64
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
HTML = b"<!DOCTYPE html><script>alert(1)</script>"


def data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def test_sniff():
    print("\n[1] 檔頭判斷")
    for raw, expect in [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif"), (WEBP, "image/webp"),
                        (SVG, None), (HTML, None), (b"", None), (b"RIFF\x00\x00\x00\x00WAVE", None)]:
        got = sniff_image_type(raw[:16])
        check(f"{raw[:10]!r} → {expect}", got == expect, f"got {got}")


def test_check_url():
    print("\n[2] 訊息附圖網址")
    uid, bucket = "11111111-2222-3333-4444-555555555555", "my-bucket"
    own = f"https://storage.googleapis.com/{bucket}/uploads/{uid}/abc.png?X-Goog-Signature=x"
    ok_cases = [
        ("PNG Data URL", data_url("image/png", PNG), None),
        ("WebP Data URL（訪客）", data_url("image/webp", WEBP), None),
        ("自己目錄的 GCS 簽名 URL", own, uid),
    ]
    for name, url, user in ok_cases:
        err = check_message_image_url(url, user, bucket)
        check(f"接受：{name}", err is None, err or "")
    bad_cases = [
        ("SVG Data URL", data_url("image/svg+xml", SVG), uid),
        ("宣告 PNG、實際是 HTML", data_url("image/png", HTML), uid),
        ("宣告 PNG、實際是 JPEG", data_url("image/png", JPEG), uid),
        ("Base64 亂碼", "data:image/png;base64,!!!!@@@@####$$$$%%%%^^^^", uid),
        ("超過大小的 Data URL", data_url("image/png", PNG + b"\x00" * MAX_DATA_URL_IMAGE_BYTES), uid),
        ("任意外部網址", "https://evil.example.com/a.png", uid),
        ("內網網址", "http://169.254.169.254/latest/meta-data/", uid),
        ("http（非 https）的 GCS", own.replace("https://", "http://"), uid),
        ("別人目錄的 GCS", f"https://storage.googleapis.com/{bucket}/uploads/other-user/abc.png", uid),
        ("別的 bucket", f"https://storage.googleapis.com/other-bucket/uploads/{uid}/abc.png", uid),
        ("路徑穿越", f"https://storage.googleapis.com/{bucket}/uploads/{uid}/../other/abc.png", uid),
        ("偽裝主機名稱", f"https://storage.googleapis.com.evil.com/{bucket}/uploads/{uid}/a.png", uid),
        ("帶 port", f"https://storage.googleapis.com:8443/{bucket}/uploads/{uid}/a.png", uid),
        ("帶帳密的網址", f"https://x@evil.com/{bucket}/uploads/{uid}/a.png", uid),
        ("訪客用 GCS 網址", own, None),
    ]
    for name, url, user in bad_cases:
        err = check_message_image_url(url, user, bucket)
        check(f"拒絕：{name}", err is not None, "沒被擋下")
    check("沒設 GCS_BUCKET 時拒絕 GCS 網址", check_message_image_url(own, uid, None) is not None)


async def test_endpoints():
    print("\n[3] API 端點")
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1")
    guest = deps.Principal(id="guest:" + uuid.uuid4().hex, is_guest=True)
    app.dependency_overrides[deps.principal_or_new_guest] = lambda: guest
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async def upload(name: str, raw: bytes, mime: str = "image/png"):
            return await client.post("/api/v1/chat/upload-image", files={"file": (name, raw, mime)})

        r = await upload("photo.png", PNG)
        check("PNG 上傳成功", r.status_code == 200 and r.json()["content_type"] == "image/png", r.text[:100])
        r = await upload("photo.png", JPEG)
        check("副檔名 .png、實際是 JPEG → 以檔頭為準", r.status_code == 200 and r.json()["content_type"] == "image/jpeg",
              r.text[:100])
        r = await upload("evil.png", SVG, "image/png")
        check("SVG 改名成 .png → 415", r.status_code == 415, f"{r.status_code} {r.text[:100]}")
        r = await upload("page.png", HTML, "image/png")
        check("HTML 改名成 .png → 415", r.status_code == 415, f"{r.status_code} {r.text[:100]}")
        r = await upload("big.png", PNG + b"\x00" * MAX_DATA_URL_IMAGE_BYTES)
        check("超過大小 → 413", r.status_code == 413, f"{r.status_code} {r.text[:100]}")

        for name, url in [("外部網址", "https://evil.example.com/a.png"),
                          ("SVG Data URL", data_url("image/svg+xml", SVG))]:
            r = await client.post("/api/v1/chat/tasks", json={"message": "這是什麼", "image_url": url})
            check(f"/chat/tasks 帶{name} → 400（沒進佇列）", r.status_code == 400, f"{r.status_code} {r.text[:100]}")


def test_gcs():
    print("\n[4] GCS 簽名 URL（真的上傳）")
    import os

    from app.api.v1 import oss

    uid = f"test-{uuid.uuid4().hex[:8]}"
    user = deps.Principal(id=uid, is_guest=False)
    created = []

    def presign():
        res = oss.chat_endpoint("x.png", user)
        created.append(res["uploadUrl"].split("?")[0].split(f"/{os.environ['GCS_BUCKET']}/", 1)[1])
        return res

    res = presign()
    r = httpx.put(res["uploadUrl"], headers=res["uploadHeaders"], content=PNG, timeout=60)
    check("照 uploadHeaders 上傳 → 成功", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
    err = check_message_image_url(res["accessUrl"], uid, os.environ["GCS_BUCKET"])
    check("產生的讀取 URL 通得過網址檢查", err is None, err or "")
    r = httpx.get(res["accessUrl"], timeout=60)
    check("讀取 URL 拿得到剛上傳的檔案", r.status_code == 200 and r.content == PNG, str(r.status_code))

    res = presign()
    r = httpx.put(res["uploadUrl"], headers={"Content-Type": res["contentType"]}, content=PNG, timeout=60)
    check("少帶大小範圍 header → 被拒", r.status_code in (400, 403), f"{r.status_code}")

    res = presign()
    r = httpx.put(res["uploadUrl"], headers=res["uploadHeaders"], content=PNG + b"\x00" * MAX_IMAGE_BYTES, timeout=300)
    check("超過 10MB → 被拒", r.status_code in (400, 413), f"{r.status_code} {r.text[:200]}")

    # 清掉測試檔（service account 沒有刪除權限就略過，檔案在 uploads/test-* 底下）
    try:
        bucket = oss.get_gcs_client().bucket(os.environ["GCS_BUCKET"])
        for name in created:
            blob = bucket.blob(name)
            if blob.exists():
                blob.delete()
        print("  🧹 已刪除測試檔")
    except Exception as exc:
        print(f"  ⚠️ 無法刪除測試檔 uploads/{uid}/（{type(exc).__name__}），請手動清理")


def main():
    test_sniff()
    test_check_url()
    asyncio.run(test_endpoints())
    if "--gcs" in sys.argv:
        test_gcs()
    print(f"\n結果：{passed} 通過，{failed} 失敗")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
