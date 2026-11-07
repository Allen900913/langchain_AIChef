"""帳號、密碼與 token。

三種 token，全部放在 httpOnly cookie（JavaScript 讀不到，XSS 偷不走；EventSource 也會自動帶上）：

- access token：JWT，15 分鐘。每個請求用它認人，驗簽即可，不查資料庫。
- refresh token：隨機字串。資料庫只存 SHA-256 雜湊；每用一次就換一張新的（rotation），登出時撤銷。
  兩個期限：閒置 7 天沒換發就過期（每次換發往後推）；從輸入密碼算起 30 天一定過期（換發只繼承、不延長），
  所以 token 被偷後，就算一直換發也撐不過這個期限，時間到必須重新輸入密碼。舊 token 在寬限期內重送只換 access token（不開第二條鏈）；超過寬限期又出現 → 視為外洩，
  撤銷該使用者全部 refresh token。
- guest token：JWT，訪客 session（sub = guest:<hex>），沒有密碼、不進 users 表、伺服器不存任何
  訪客資料。它只用來確認「看這個事件串流 / 停止這個任務的人，就是送出訊息的人」。
"""

import asyncio
import hashlib
import os
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash
from psycopg.errors import UniqueViolation

from app.services import db

JWT_ALGORITHM = "HS256"
ACCESS_TTL = timedelta(minutes=int(os.getenv("ACCESS_TOKEN_MINUTES", "15")))
REFRESH_TTL = timedelta(days=int(os.getenv("REFRESH_TOKEN_DAYS", "7")))  # 閒置期限
REFRESH_ABSOLUTE_TTL = timedelta(days=int(os.getenv("REFRESH_ABSOLUTE_DAYS", "30")))  # 從登入算起的絕對期限
GUEST_TTL = timedelta(hours=int(os.getenv("GUEST_TTL_HOURS", "24")))
# 多分頁同時用同一張 refresh token 換發時，第二個請求不算外洩。
# 只需要涵蓋「請求還在路上」的時間：回應一回來，瀏覽器之後送的就是新 cookie，所以不用太長。
REFRESH_REUSE_GRACE = timedelta(seconds=10)

GUEST_PREFIX = "guest:"
PASSWORD_MIN, PASSWORD_MAX = 8, 128
EMAIL_MAX = 254
DISPLAY_NAME_MAX = 50
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_hasher = PasswordHash.recommended()  # Argon2id
# 帳號不存在時也跑一次雜湊比對，讓回應時間看不出「這個 email 有沒有註冊」
_DUMMY_HASH = _hasher.hash(secrets.token_hex(16))


class AuthError(Exception):
    """登入 / 註冊 / 換發失敗；message 可直接回給前端。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TokenClaims:
    subject: str
    expires_at: datetime


def _secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET 未設定或太短（至少 32 字元），請在 .env 設定一組隨機字串")
    return secret


def check_config() -> None:
    """API 啟動時呼叫：設定錯誤要在啟動時就失敗，而不是第一個登入請求才炸。"""
    _secret()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ==============================================================================
# JWT
# ==============================================================================

def _encode(sub: str, typ: str, ttl: timedelta) -> tuple[str, datetime]:
    now = _now()
    exp = now + ttl
    token = jwt.encode({"sub": sub, "typ": typ, "iat": now, "exp": exp}, _secret(), algorithm=JWT_ALGORITHM)
    return token, exp


def create_access_token(user_id: str) -> str:
    return _encode(user_id, "access", ACCESS_TTL)[0]


def create_guest_token() -> tuple[str, str, datetime]:
    """回傳 (guest_id, token, 到期時間)。"""
    guest_id = f"{GUEST_PREFIX}{uuid.uuid4().hex}"
    token, exp = _encode(guest_id, "guest", GUEST_TTL)
    return guest_id, token, exp


def decode_token(token: str, typ: str) -> TokenClaims | None:
    """驗簽 + 驗期限 + 驗類型（避免拿 guest token 冒充 access token）。失敗回傳 None。"""
    try:
        claims = jwt.decode(
            token, _secret(), algorithms=[JWT_ALGORITHM], options={"require": ["sub", "exp", "typ"]}
        )
    except jwt.PyJWTError:
        return None
    if claims.get("typ") != typ:
        return None
    sub = claims["sub"]
    if typ == "guest" and not sub.startswith(GUEST_PREFIX):
        return None
    if typ == "access" and sub.startswith(GUEST_PREFIX):
        return None
    return TokenClaims(subject=sub, expires_at=datetime.fromtimestamp(claims["exp"], timezone.utc))


# ==============================================================================
# 帳號
# ==============================================================================

def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    # 先檢查長度：這個 regex 遇到特製的長字串（例如大量連續的點）耗時會隨長度平方成長，
    # 而且跑在 event loop 上，卡住時所有請求都要等
    if len(email) > EMAIL_MAX or not _EMAIL_RE.match(email):
        raise AuthError("Email 格式不正確")
    return email


def validate_password(password: str) -> None:
    if not (PASSWORD_MIN <= len(password or "") <= PASSWORD_MAX):
        raise AuthError(f"密碼長度需介於 {PASSWORD_MIN} 到 {PASSWORD_MAX} 字元")


def public_user(row: dict) -> dict:
    return {"id": str(row["id"]), "email": row["email"], "display_name": row.get("display_name")}


async def create_user(email: str, password: str, display_name: str | None = None) -> dict:
    email = normalize_email(email)
    validate_password(password)
    display_name = (display_name or "").strip()[:DISPLAY_NAME_MAX] or None
    # Argon2 一次約數十毫秒的 CPU，丟到執行緒池，不卡住 event loop 上的 SSE
    password_hash = await asyncio.to_thread(_hasher.hash, password)
    try:
        async with db.get_pool().connection() as conn:
            cur = await conn.execute(
                "INSERT INTO users (id, email, password_hash, display_name) VALUES (%s, %s, %s, %s)"
                " RETURNING id, email, display_name",
                (uuid.uuid4(), email, password_hash, display_name),
            )
            return await cur.fetchone()
    except UniqueViolation:
        raise AuthError("這個 Email 已經註冊過了", status_code=409) from None


async def authenticate(email: str, password: str) -> dict | None:
    """帳密正確回傳使用者，否則 None（不區分「帳號不存在」與「密碼錯誤」）。"""
    try:
        email = normalize_email(email)
    except AuthError:
        return None
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT id, email, display_name, password_hash FROM users WHERE email = %s", (email,)
        )
        row = await cur.fetchone()
    if row is None:
        await asyncio.to_thread(_hasher.verify, password or "", _DUMMY_HASH)
        return None
    ok, new_hash = await asyncio.to_thread(_hasher.verify_and_update, password or "", row["password_hash"])
    if not ok:
        return None
    if new_hash:  # 雜湊參數升級過：順手換成新參數
        async with db.get_pool().connection() as conn:
            await conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, row["id"]))
    return row


async def get_user(user_id: str) -> dict | None:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return None
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("SELECT id, email, display_name FROM users WHERE id = %s", (uid,))
        return await cur.fetchone()


# ==============================================================================
# Refresh token
# ==============================================================================

def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def _insert_refresh_token(conn, user_id, absolute_expires_at: datetime) -> str:
    """閒置期限不能超過絕對期限：快到期時發的 token 只活到絕對期限為止。"""
    raw = secrets.token_urlsafe(48)
    await conn.execute(
        "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, absolute_expires_at)"
        " VALUES (%s, %s, %s, %s, %s)",
        (uuid.uuid4(), user_id, _hash_token(raw), min(_now() + REFRESH_TTL, absolute_expires_at), absolute_expires_at),
    )
    return raw


async def issue_refresh_token(user_id) -> str:
    """登入用：輸入過密碼，絕對期限從現在重新起算。"""
    async with db.get_pool().connection() as conn:
        return await _insert_refresh_token(conn, user_id, _now() + REFRESH_ABSOLUTE_TTL)


async def rotate_refresh_token(raw: str) -> tuple[dict, str | None]:
    """用 refresh token 換發，回傳 (使用者, 新 refresh token)。失敗拋 AuthError(401)。

    - 超過閒置期限或絕對期限：拒絕。只影響這條鏈，其他裝置的登入不受影響。
    - 第一次使用：換一張新的，繼承原本的絕對期限。
    - 寬限期內重送（例如多分頁同時換發）：新 token 回 None，呼叫端只發 access token。
      不再發第二張 refresh token，同一個登入就不會分叉成兩條互不相干的鏈；
      就算先換發的是駭客，使用者之後再用這張舊 token 時也一定會被偵測到。
    - 超過寬限期又出現：視為外洩，撤銷這個使用者全部 refresh token。
    - 已被撤銷（登出或偵測到外洩）：一律拒絕，沒有寬限。
    """
    reused = False
    new_raw = None
    async with db.get_pool().connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                "SELECT id, user_id, expires_at, absolute_expires_at, rotated_at, revoked_at FROM refresh_tokens"
                " WHERE token_hash = %s FOR UPDATE",
                (_hash_token(raw),),
            )
            row = await cur.fetchone()
            if row is None or min(row["expires_at"], row["absolute_expires_at"]) <= _now():
                raise AuthError("登入已過期，請重新登入", status_code=401)
            if row["revoked_at"] is not None:
                raise AuthError("登入已失效，請重新登入", status_code=401)
            rotated_at = row["rotated_at"]
            if rotated_at is not None and _now() - rotated_at > REFRESH_REUSE_GRACE:
                # 已換發過的 token 又出現：可能被偷了，讓這個使用者所有裝置都重新登入。
                # 這裡不能直接 raise：例外會讓交易回滾，撤銷就白做了
                await conn.execute(
                    "UPDATE refresh_tokens SET revoked_at = now()"
                    " WHERE user_id = %s AND revoked_at IS NULL",
                    (row["user_id"],),
                )
                reused = True
            else:
                cur = await conn.execute(
                    "SELECT id, email, display_name FROM users WHERE id = %s", (row["user_id"],)
                )
                user = await cur.fetchone()
                if user is None:
                    raise AuthError("帳號不存在", status_code=401)
                if rotated_at is None:
                    # 寫入和比對都用應用程式的時間，寬限期才不會受資料庫時鐘差影響
                    await conn.execute(
                        "UPDATE refresh_tokens SET rotated_at = %s WHERE id = %s", (_now(), row["id"])
                    )
                    new_raw = await _insert_refresh_token(conn, user["id"], row["absolute_expires_at"])
    if reused:
        raise AuthError("登入已失效，請重新登入", status_code=401)
    return user, new_raw


async def revoke_refresh_token(raw: str) -> None:
    async with db.get_pool().connection() as conn:
        await conn.execute(
            "UPDATE refresh_tokens SET revoked_at = now() WHERE token_hash = %s AND revoked_at IS NULL",
            (_hash_token(raw),),
        )


async def purge_expired_refresh_tokens() -> int:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM refresh_tokens WHERE expires_at < now() - interval '1 day'")
        return cur.rowcount
