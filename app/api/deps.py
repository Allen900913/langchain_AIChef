"""認證相關的 FastAPI 依賴與 cookie 工具。

身分只從伺服器簽發的 cookie 來，請求本文裡不收 user_id：前端送什麼都不影響「你是誰」。

判定順序：
1. 有 access cookie：驗得過 → 登入使用者；驗不過（過期 / 被竄改）→ 401，前端用 refresh 換新的。
   不會退回訪客身分，否則 token 一過期，使用者送出的訊息就會在不知情的情況下變成不保存的訪客訊息。
2. 沒有 access cookie、有有效的 guest cookie → 訪客。
3. 都沒有 → 沒有身分（送訊息時才會自動發訪客身分）。

訪客不留任何紀錄：guest cookie 只是一把「暫時的 session 鑰匙」，用來確認事件串流與停止
請求是不是訊息的送出者本人，伺服器端沒有任何資料綁在它上面。
"""

import ipaddress
import os
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response

from app.services import auth, threads

ACCESS_COOKIE = "chef_access"
REFRESH_COOKIE = "chef_refresh"
GUEST_COOKIE = "chef_guest"
# refresh token 只送到認證 API，平常的請求不帶它，降低外洩面
REFRESH_COOKIE_PATH = "/api/v1/auth"

# 正式環境走 HTTPS 一定要 Secure。Chrome / Firefox 把 http://localhost 視為安全來源，
# 所以本機開發也能用；用區網 IP 走 http 測試時才需要設 COOKIE_SECURE=false。
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() != "false"


@dataclass
class Principal:
    id: str        # 使用者 uuid 或 guest:<hex>；就是傳給 agent 的 user_id
    is_guest: bool


def _set_cookie(response: Response, name: str, value: str, max_age: int, path: str = "/") -> None:
    response.set_cookie(
        name, value, max_age=max_age, path=path,
        httponly=True, secure=COOKIE_SECURE, samesite="lax",
    )


def set_login_cookies(response: Response, user_id: str, refresh_token: str | None) -> None:
    """refresh_token 為 None（寬限期內重送）時只更新 access cookie，瀏覽器沿用已經拿到的新 refresh cookie。"""
    # access cookie 留得跟 refresh token 一樣久，有效期限由 JWT 裡的 exp（15 分鐘）決定。
    # cookie 若也只留 15 分鐘，時間一到瀏覽器就把它刪掉，伺服器看到的是「沒帶身分」而不是
    # 「token 過期」：/auth/me 回未登入、送訊息被當成訪客（對話不保存），前端也不會去換新。
    # 留著過期的 token 則是回 401，前端照常用 refresh cookie 換一張再重試。
    _set_cookie(response, ACCESS_COOKIE, auth.create_access_token(user_id), int(auth.REFRESH_TTL.total_seconds()))
    if refresh_token is not None:
        _set_cookie(response, REFRESH_COOKIE, refresh_token, int(auth.REFRESH_TTL.total_seconds()), REFRESH_COOKIE_PATH)
    response.delete_cookie(GUEST_COOKIE, path="/")


def clear_login_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=401, detail="登入已過期，請重新登入", headers={"WWW-Authenticate": "Cookie"})


def _parse_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    return getattr(ip, "ipv4_mapped", None) or ip


def client_ip(request: Request) -> str:
    """使用者的真實 IP（速率限制用）。

    路徑是 Caddy → nginx → API，API 直接看到的是 nginx 的內部位址。Caddy 會覆蓋使用者送來的
    X-Forwarded-For，nginx 原樣轉送，所以這個值可信；但只有連線來自內部位址時才採用，
    外部直接連進來時它可以被任意偽造。IPv6 整段 /64 算同一個來源：一個使用者手上常有一整段位址。
    """
    peer = request.client.host if request.client else ""
    ip = _parse_ip(peer)
    if ip is not None and ip.is_private:
        forwarded = request.headers.getlist("x-forwarded-for")
        if forwarded:
            ip = _parse_ip(forwarded[-1].rsplit(",", 1)[-1].strip()) or ip
    if ip is None:
        return peer or "unknown"
    return str(ipaddress.ip_network((ip, 64), strict=False)) if ip.version == 6 else str(ip)


async def optional_principal(request: Request) -> Principal | None:
    access = request.cookies.get(ACCESS_COOKIE)
    if access:
        claims = auth.decode_token(access, "access")
        if claims is None:
            raise _unauthorized()
        return Principal(id=claims.subject, is_guest=False)
    guest = request.cookies.get(GUEST_COOKIE)
    if guest:
        claims = auth.decode_token(guest, "guest")
        if claims is not None:
            return Principal(id=claims.subject, is_guest=True)
    return None


async def require_principal(request: Request) -> Principal:
    principal = await optional_principal(request)
    if principal is None:
        raise _unauthorized()
    return principal


async def require_user(request: Request) -> Principal:
    """只有登入使用者能用的功能（例如把圖片上傳到雲端儲存：訪客不能留下檔案）。"""
    principal = await require_principal(request)
    if principal.is_guest:
        raise HTTPException(status_code=403, detail="請先登入才能使用這個功能")
    return principal


async def principal_or_new_guest(request: Request, response: Response) -> Principal:
    """送訊息用：沒有身分就發一個訪客 session（只是一把鑰匙，伺服器不存任何訪客資料）。"""
    principal = await optional_principal(request)
    if principal is not None:
        return principal
    guest_id, token, _ = auth.create_guest_token()
    _set_cookie(response, GUEST_COOKIE, token, int(auth.GUEST_TTL.total_seconds()))
    return Principal(id=guest_id, is_guest=True)


async def owned_thread(thread_id: str, principal: Principal) -> dict:
    """取得自己的 thread；不存在、不是自己的、訪客一律 404（不洩漏是否存在）。"""
    row = None if principal.is_guest else await threads.get_owned(thread_id, principal.id)
    if row is None:
        raise HTTPException(status_code=404, detail="對話不存在")
    return row
