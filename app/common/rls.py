"""Row-Level Security：只用在兩種「讀取」上——對話清單（threads）與使用者偏好（store 的 memories.*）。

做法跟 Supabase / PostgREST 一樣：應用程式照常用自己的帳號連線，要讀這兩種資料時，
在交易裡 `SET LOCAL ROLE chef_rls_reader` 並設定 `app.user_id`，Postgres 依 policy 只回傳
該使用者的列。交易結束（commit / rollback）角色與設定自動還原，連線池裡的連線不會被汙染。

為什麼只用在讀取：
- 寫入走 LangGraph 的 store / checkpointer，它們的 SQL 不歸我們管，也無法逐請求設定 session 變數。
- 這兩種讀取是「把資料交給使用者或模型看」的出口，是洩漏時傷害最大的地方。

應用層的 WHERE 條件仍然保留：RLS 是第二道防線，不是唯一一道。
即使哪天查詢忘了加 WHERE、或 user_id 組錯，資料庫也只會回傳空結果，不會回傳別人的資料。

policy 只套在 chef_rls_reader 這個角色上（`TO chef_rls_reader`）：應用程式本身的連線角色是
資料表擁有者 / superuser，不受影響，LangGraph 的讀寫行為完全不變。
"""

from contextlib import asynccontextmanager

RLS_ROLE = "chef_rls_reader"
USER_SETTING = "app.user_id"

# 與 app/services/db.py 遷移共用同一把 advisory lock：API 與 worker 同時啟動時，DDL 不會互撞
MIGRATION_LOCK_ID = 7_412_001

_ROLE_SQL = f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{RLS_ROLE}') THEN
        CREATE ROLE {RLS_ROLE} NOLOGIN;
    END IF;
END $$;
GRANT USAGE ON SCHEMA public TO {RLS_ROLE};
-- 讓應用程式的連線角色可以 SET ROLE 成它（superuser 本來就可以，非 superuser 部署時需要）
DO $$ BEGIN EXECUTE format('GRANT {RLS_ROLE} TO %I', current_user); END $$;
"""

THREADS_RLS_SQL = _ROLE_SQL + f"""
GRANT SELECT ON threads TO {RLS_ROLE};
ALTER TABLE threads ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS threads_owner_read ON threads;
CREATE POLICY threads_owner_read ON threads FOR SELECT TO {RLS_ROLE}
    USING (owner_id = current_setting('{USER_SETTING}', true));
"""

# store.prefix 是 namespace 以「.」串起來的字串，例如 memories.<user_id>.diet、inventory.<user_id>。
# LangGraph 規定 namespace 每一段不能含「.」，所以第二段一定就是 user_id。
STORE_RLS_SQL = _ROLE_SQL + f"""
GRANT SELECT ON store TO {RLS_ROLE};
ALTER TABLE store ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS store_owner_read ON store;
CREATE POLICY store_owner_read ON store FOR SELECT TO {RLS_ROLE}
    USING (split_part(prefix, '.', 2) = current_setting('{USER_SETTING}', true));
"""


async def _assert_enabled(conn, table: str, policy: str) -> None:
    """RLS 沒開或 policy 不見時，讀取會直接回傳整張表：寧可啟動失敗也不要帶著這個狀態上線。"""
    cur = await conn.execute(
        "SELECT c.relrowsecurity AS on, EXISTS (SELECT 1 FROM pg_policies p"
        " WHERE p.tablename = c.relname AND p.policyname = %s) AS has_policy"
        " FROM pg_class c WHERE c.relname = %s AND c.relnamespace = 'public'::regnamespace",
        (policy, table),
    )
    row = await cur.fetchone()
    on, has_policy = (row["on"], row["has_policy"]) if isinstance(row, dict) else row
    if not (on and has_policy):
        raise RuntimeError(f"{table} 的 Row-Level Security 未正確設定（policy {policy}）")


async def ensure_threads_rls(conn) -> None:
    """在 threads 表上建立 RLS（遷移時於同一個交易內呼叫，呼叫端已持有 advisory lock）。"""
    await conn.execute(THREADS_RLS_SQL)


async def ensure_store_rls(pool) -> None:
    """在 LangGraph 的 store 表上建立 RLS（worker 於 store.setup() 之後呼叫；冪等）。"""
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
            await conn.execute(STORE_RLS_SQL)
            await _assert_enabled(conn, "store", "store_owner_read")


async def verify_threads_rls(pool) -> None:
    async with pool.connection() as conn:
        await _assert_enabled(conn, "threads", "threads_owner_read")


@asynccontextmanager
async def as_user(pool, user_id: str):
    """取得一條「只看得到 user_id 自己資料」的連線（交易內有效）。"""
    if not user_id:
        raise ValueError("as_user 需要 user_id")
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
            await conn.execute("SELECT set_config(%s, %s, true)", (USER_SETTING, user_id))
            yield conn


def _row_get(row, key: str, index: int):
    return row[key] if isinstance(row, dict) else row[index]


async def read_profiles(pool, user_id: str, domains) -> dict[str, list]:
    """讀取使用者三個偏好領域（diet / kitchen / household）的內容，經過 RLS。

    回傳 {domain: [content, ...]}，每個 domain 都有 key（沒資料時是空 list）。
    """
    domains = list(domains)
    prefixes = {f"memories.{user_id}.{d}": d for d in domains}
    result: dict[str, list] = {d: [] for d in domains}
    async with as_user(pool, user_id) as conn:
        cur = await conn.execute(
            "SELECT prefix, value FROM store WHERE prefix = ANY(%s)"
            " AND (expires_at IS NULL OR expires_at > now()) ORDER BY updated_at DESC",
            (list(prefixes),),
        )
        for row in await cur.fetchall():
            domain = prefixes.get(_row_get(row, "prefix", 0))
            value = _row_get(row, "value", 1) or {}
            if domain is not None:
                result[domain].append(value.get("content"))
    return result
