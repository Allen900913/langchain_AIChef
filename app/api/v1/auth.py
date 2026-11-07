"""認證 API：註冊 / 登入 / 換發 / 登出 / 目前使用者。

訪客不留紀錄，所以登入時沒有東西要轉移：直接換成使用者身分、清掉訪客 cookie。
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api import deps
from app.common import task_queue as tq
from app.services import auth

router = APIRouter()

# 同一個 email 15 分鐘內最多失敗 10 次，擋線上暴力猜密碼
_LOGIN_FAIL_LIMIT = 10
_LOGIN_FAIL_WINDOW = 15 * 60


# 長度在這裡先擋：這兩支 API 不用登入，超長的 email 不該進到 regex，也不該變成 Redis 鍵。
# 太短、格式不對由 services/auth.py 檢查，回傳可以直接顯示給使用者的中文訊息。
class RegisterRequest(BaseModel):
    email: str = Field(max_length=auth.EMAIL_MAX)
    password: str = Field(max_length=auth.PASSWORD_MAX)
    display_name: str | None = Field(default=None, max_length=auth.DISPLAY_NAME_MAX)


class LoginRequest(BaseModel):
    email: str = Field(max_length=auth.EMAIL_MAX)
    password: str = Field(max_length=auth.PASSWORD_MAX)


def _fail_key(email: str) -> str:
    return f"chef:auth:fail:{email.strip().lower()}"


async def _finish_login(response: Response, user: dict) -> dict:
    deps.set_login_cookies(response, str(user["id"]), await auth.issue_refresh_token(user["id"]))
    return {"user": auth.public_user(user)}


@router.post("/auth/register", status_code=201)
async def register(body: RegisterRequest, response: Response):
    try:
        user = await auth.create_user(body.email, body.password, body.display_name)
    except auth.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    return await _finish_login(response, user)


@router.post("/auth/login")
async def login(body: LoginRequest, response: Response):
    r = tq.get_redis()
    key = _fail_key(body.email)
    if int(await r.get(key) or 0) >= _LOGIN_FAIL_LIMIT:
        raise HTTPException(status_code=429, detail="登入失敗次數過多，請 15 分鐘後再試")
    user = await auth.authenticate(body.email, body.password)
    if user is None:
        async with r.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, _LOGIN_FAIL_WINDOW, nx=True)
            await pipe.execute()
        raise HTTPException(status_code=401, detail="Email 或密碼錯誤")
    await r.delete(key)
    return await _finish_login(response, user)


def _refresh_failed(message: str, status_code: int = 401) -> JSONResponse:
    """換發失敗要清掉 cookie：否則過期的 access cookie 還在，之後每個請求都回 401。

    直接回 JSONResponse（不是 raise HTTPException），清 cookie 的 Set-Cookie 才會跟著送出。
    """
    res = JSONResponse(status_code=status_code, content={"detail": message})
    deps.clear_login_cookies(res)
    return res


@router.post("/auth/refresh")
async def refresh(request: Request, response: Response):
    raw = request.cookies.get(deps.REFRESH_COOKIE)
    if not raw:
        return _refresh_failed("尚未登入")
    try:
        user, new_raw = await auth.rotate_refresh_token(raw)
    except auth.AuthError as exc:
        return _refresh_failed(str(exc), exc.status_code)
    deps.set_login_cookies(response, str(user["id"]), new_raw)
    return {"user": auth.public_user(user)}


@router.post("/auth/logout", status_code=204)
async def logout(request: Request, response: Response):
    raw = request.cookies.get(deps.REFRESH_COOKIE)
    if raw:
        await auth.revoke_refresh_token(raw)
    deps.clear_login_cookies(response)
    response.delete_cookie(deps.GUEST_COOKIE, path="/")


@router.get("/auth/me")
async def me(request: Request, principal: deps.Principal | None = Depends(deps.optional_principal)):
    """已登入回傳使用者；訪客回傳 guest=true；access token 過期回 401（前端會先 refresh）。

    沒有 access cookie 但有 refresh cookie（舊版把 access cookie 設成 15 分鐘就過期，瀏覽器已經刪掉）
    也回 401 讓前端換新，不把還能續期的使用者當成未登入。refresh cookie 的 path 是 /api/v1/auth，這裡收得到。
    """
    if principal is None:
        if request.cookies.get(deps.REFRESH_COOKIE):
            raise HTTPException(status_code=401, detail="登入已過期，請重新登入")
        return {"user": None, "guest": False}
    if principal.is_guest:
        return {"user": None, "guest": True}
    user = await auth.get_user(principal.id)
    if user is None:
        raise HTTPException(status_code=401, detail="帳號不存在")
    return {"user": auth.public_user(user), "guest": False}
