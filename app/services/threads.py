"""對話擁有權（threads 表）。只有登入使用者的對話會出現在這裡；訪客不留任何紀錄。

checkpointer 只用 thread_id 當鍵，不知道對話屬於誰；「誰可以讀寫這個 thread」以這張表為準。
每個碰到 thread 的 API 都先經過 get_owned()：不是擁有者就當作不存在（404），
不洩漏「這個 thread_id 存在、只是不是你的」。

對話清單（list_owned）經過 Row-Level Security 讀取：見 app/common/rls.py。
"""

from app.common import rls
from app.services import db

DEFAULT_TITLE = "新對話"
TITLE_MAX = 20

_COLUMNS = "thread_id, owner_id, title, created_at, updated_at"


def make_title(text: str) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return DEFAULT_TITLE
    return f"{clean[:TITLE_MAX]}..." if len(clean) > TITLE_MAX else clean


def public_thread(row: dict) -> dict:
    return {
        "thread_id": row["thread_id"],
        "title": row["title"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


async def create(thread_id: str, owner_id: str, title: str) -> dict:
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            f"INSERT INTO threads (thread_id, owner_id, title) VALUES (%s, %s, %s) RETURNING {_COLUMNS}",
            (thread_id, owner_id, title),
        )
        return await cur.fetchone()


async def get_owned(thread_id: str, owner_id: str) -> dict | None:
    """擁有者是 owner_id 才回傳，否則 None。"""
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_COLUMNS} FROM threads WHERE thread_id = %s AND owner_id = %s",
            (thread_id, owner_id),
        )
        return await cur.fetchone()


async def list_owned(owner_id: str, limit: int = 200) -> list[dict]:
    """側邊欄清單。在 RLS 角色下讀取：就算 WHERE 寫錯，資料庫也只會回傳 owner_id 自己的列。"""
    async with rls.as_user(db.get_pool(), owner_id) as conn:
        cur = await conn.execute(
            f"SELECT {_COLUMNS} FROM threads WHERE owner_id = %s ORDER BY updated_at DESC LIMIT %s",
            (owner_id, limit),
        )
        return await cur.fetchall()


async def touch_owned(thread_id: str, owner_id: str) -> dict | None:
    """驗擁有者並更新最後活動時間（一次查詢）；不是自己的回傳 None。"""
    async with db.get_pool().connection() as conn:
        cur = await conn.execute(
            f"UPDATE threads SET updated_at = now() WHERE thread_id = %s AND owner_id = %s RETURNING {_COLUMNS}",
            (thread_id, owner_id),
        )
        return await cur.fetchone()


async def delete(thread_id: str) -> None:
    async with db.get_pool().connection() as conn:
        await conn.execute("DELETE FROM threads WHERE thread_id = %s", (thread_id,))
