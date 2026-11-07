"""API 端共用的 Postgres 連線池，以及本專案自己的資料表遷移（users / refresh_tokens / threads）。

LangGraph 的 checkpoint / store 資料表由 worker 的 init_agent_infra 呼叫 setup() 建立
（store 的 RLS 也由 worker 在 setup 之後建立）；這裡只管我們自己的資料表。

遷移以 pg_advisory_xact_lock 序列化：多個 API 行程同時啟動時只有一個會真的執行，
其他的等它完成後看到版本已套用就跳過。
"""

import os

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.common import rls

_pool: AsyncConnectionPool | None = None

# 所有 API / worker 行程搶同一把 advisory lock（worker 建 store 的 RLS 時也用它）
_MIGRATION_LOCK_ID = rls.MIGRATION_LOCK_ID

# 依序套用；已上線的版本不要改內容，要變更就新增一個版本
_MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE users (
        id            uuid PRIMARY KEY,
        email         text NOT NULL UNIQUE,          -- 一律存小寫（應用層正規化）
        password_hash text NOT NULL,
        display_name  text,
        created_at    timestamptz NOT NULL DEFAULT now()
    );

    -- refresh token 只存 SHA-256 雜湊：資料庫外洩也拿不到可用的 token
    CREATE TABLE refresh_tokens (
        id          uuid PRIMARY KEY,
        user_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash  text NOT NULL UNIQUE,
        expires_at  timestamptz NOT NULL,
        revoked_at  timestamptz,
        created_at  timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX refresh_tokens_user_idx ON refresh_tokens (user_id);

    -- 對話擁有權：checkpointer 只認 thread_id，「誰可以讀這個 thread」記在這裡。
    -- owner_id 是使用者 uuid 或 guest:<hex>（訪客），所以不設 FK。
    CREATE TABLE threads (
        thread_id   text PRIMARY KEY,
        owner_id    text NOT NULL,
        title       text NOT NULL,
        is_guest    boolean NOT NULL DEFAULT false,
        expires_at  timestamptz,                     -- 只有訪客對話有期限
        created_at  timestamptz NOT NULL DEFAULT now(),
        updated_at  timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX threads_owner_updated_idx ON threads (owner_id, updated_at DESC);
    CREATE INDEX threads_guest_expiry_idx ON threads (expires_at) WHERE is_guest;
    """),
    # v2：訪客不再留任何紀錄（只看 SSE 串流），threads 只屬於登入使用者；
    #     對話清單的讀取改走 Row-Level Security（見 app/common/rls.py）
    (2, """
    DELETE FROM threads WHERE is_guest;
    DROP INDEX IF EXISTS threads_guest_expiry_idx;
    ALTER TABLE threads DROP COLUMN is_guest, DROP COLUMN expires_at;
    """ + rls.THREADS_RLS_SQL),
    # v3：把「換發過」和「被撤銷」分開記。之前都記在 revoked_at，寬限期又只看這個欄位，
    #     導致登出或被全部撤銷的 token 在寬限期內還能換發。
    #     舊資料分不出是哪一種，一律維持撤銷（比較安全），所以不回填 rotated_at。
    (3, """
    ALTER TABLE refresh_tokens ADD COLUMN rotated_at timestamptz;
    """),
    # v4：絕對期限。expires_at 每次換發都會往後推（閒置期限），一直換發就永遠不會過期；
    #     absolute_expires_at 在登入時決定，換發只繼承不延長。
    #     舊資料不知道當初何時登入，以目前的到期時間當期限：不會立刻踢人，之後也延長不了。
    (4, """
    ALTER TABLE refresh_tokens ADD COLUMN absolute_expires_at timestamptz;
    UPDATE refresh_tokens SET absolute_expires_at = expires_at;
    ALTER TABLE refresh_tokens ALTER COLUMN absolute_expires_at SET NOT NULL;
    """),
]


async def init_pool() -> None:
    global _pool
    if _pool is not None:
        return
    _pool = AsyncConnectionPool(
        conninfo=os.environ["DB_URI"],
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    )
    await _pool.open()


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
    _pool = None


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("資料庫連線池尚未初始化（init_pool）")
    return _pool


async def migrate() -> None:
    async with get_pool().connection() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS app_migrations ("
                " version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            cur = await conn.execute("SELECT version FROM app_migrations")
            applied = {row["version"] for row in await cur.fetchall()}
            for version, sql in _MIGRATIONS:
                if version in applied:
                    continue
                await conn.execute(sql)
                await conn.execute("INSERT INTO app_migrations (version) VALUES (%s)", (version,))
                print(f"🗄️ [DB] 已套用遷移 v{version}")
    await rls.verify_threads_rls(get_pool())
