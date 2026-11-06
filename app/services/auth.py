"""帳號、密碼與 token。

三種 token，全部放在 httpOnly cookie（JavaScript 讀不到，XSS 偷不走；EventSource 也會自動帶上）：

- access token：JWT，15 分鐘。每個請求用它認人，驗簽即可，不查資料庫。
- refresh token：隨機字串，30 天。資料庫只存 SHA-256 雜湊；每用一次就換一張新的（rotation），
  登出時撤銷。舊 token 在寬限期後又被拿來用 → 視為外洩，撤銷該使用者全部 refresh token。
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
REFRESH_TTL = timedelta(days=int(os.getenv("REFRESH_TOKEN_DAYS", "30")))
GUEST_TTL = timedelta(hours=int(os.getenv("GUEST_TTL_HOURS", "24")))
# 多分頁同時用同一張 refresh token 換發時，第二個請求不算外洩
REFRESH_REUSE_GRACE = timedelta(seconds=30)

GUEST_PREFIX = "guest:"
PASSWORD_MIN, PASSWORD_MAX = 8, 128
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
    if not _EMAIL_RE.match(email) or len(email) > 254:
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
    display_name = (display_name or "").strip()[:50] or None
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


async def issue_refresh_token(user_id) -> str:
    raw = secrets.token_urlsafe(48)
    async with db.get_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at) VALUES (%s, %s, %s, %s)",
            (uuid.uuid4(), user_id, _hash_token(raw), _now() + REFRESH_TTL),
        )
    return raw


async def rotate_refresh_token(raw: str) -> tuple[dict, str]:
    """用舊 refresh token 換一張新的，回傳 (使用者, 新 token)。失敗拋 AuthError(401)。"""
    reused = False
    async with db.get_pool().connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                "SELECT id, user_id, expires_at, revoked_at FROM refresh_tokens"
                " WHERE token_hash = %s FOR UPDATE",
                (_hash_token(raw),),
            )
            row = await cur.fetchone()
            if row is None or row["expires_at"] <= _now():
                raise AuthError("登入已過期，請重新登入", status_code=401)
            if row["revoked_at"] is not None:
                if _now() - row["revoked_at"] > REFRESH_REUSE_GRACE:
                    # 已換發過的 token 又出現：可能被偷了，讓這個使用者所有裝置都重新登入。
                    # 這裡不能直接 raise：例外會讓交易回滾，撤銷就白做了
                    await conn.execute(
                        "UPDATE refresh_tokens SET revoked_at = now()"
                        " WHERE user_id = %s AND revoked_at IS NULL",
                        (row["user_id"],),
                    )
                    reused = True
            else:
                await conn.execute("UPDATE refresh_tokens SET revoked_at = now() WHERE id = %s", (row["id"],))
            if not reused:
                cur = await conn.execute(
                    "SELECT id, email, display_name FROM users WHERE id = %s", (row["user_id"],)
                )
                user = await cur.fetchone()
                if user is None:
                    raise AuthError("帳號不存在", status_code=401)
                new_raw = secrets.token_urlsafe(48)
                await conn.execute(
                    "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at) VALUES (%s, %s, %s, %s)",
                    (uuid.uuid4(), user["id"], _hash_token(new_raw), _now() + REFRESH_TTL),
                )
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
