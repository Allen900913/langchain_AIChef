"""整合測試：登入系統、資料隔離與 Row-Level Security。真 Postgres + 真 Redis（DB 14）+ 真 FastAPI app，不啟動 worker。

執行（需要 docker compose 的 redis / postgres 已啟動，且 worker 至少啟動過一次建好 store 資料表；
會清空 Redis DB 14，測試帳號、對話與偏好資料結束時刪除）：
    PYTHONPATH=. .venv/Scripts/python.exe app/tests/integration_auth.py

涵蓋：訪客不留紀錄（不建對話、沒有清單、不能上傳到雲端）、訪客只看得到自己的任務、
別人的對話 / 任務一律 404、refresh token 輪替與重用偵測、過期 / 偽造 / 類型錯誤的 token、
CSRF Origin 檢查、登入失敗次數限制、對話清單與偏好讀取的 RLS（直接在資料庫層驗證）。
"""
import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
# 沿用 .env 的 Redis 位址與密碼，只換成測試專用的 DB 14
os.environ["REDIS_URL"] = os.getenv("REDIS_URL", "redis://localhost:6389/0").rsplit("/", 1)[0] + "/14"
sys.stdout.reconfigure(encoding="utf-8")

import httpx  # noqa: E402
import jwt  # noqa: E402
from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402
from psycopg.errors import InsufficientPrivilege  # noqa: E402

from app.api import deps  # noqa: E402
from app.common import rls  # noqa: E402
from app.common import task_queue as tq  # noqa: E402
from app.main import app  # noqa: E402
from app.services import auth, db, history  # noqa: E402

BASE = "https://t"
RUN = uuid.uuid4().hex[:8]
PASSWORD = "correct horse battery"
DOMAINS = ["diet", "kitchen", "household"]


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        raise SystemExit(1)


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE, timeout=30)


def email(name: str) -> str:
    return f"itest-{RUN}-{name}@example.com"


def as_cookie(name, value):
    return {"Cookie": f"{name}={value}"}


async def main():
    auth.check_config()
    await db.init_pool()
    await db.migrate()
    await history.init_history()
    pool = db.get_pool()
    await rls.ensure_store_rls(pool)  # 正式環境由 worker 啟動時建立
    store = AsyncPostgresStore(pool)
    r = tq.get_redis()
    await r.flushdb()
    users: list[str] = []

    async with client() as a, client() as b, client() as anon:
        # ---- 1. 沒有身分 ----
        check((await anon.get("/api/v1/auth/me")).json() == {"user": None, "guest": False}, "no cookie: anonymous")
        check((await anon.get("/api/v1/chat/threads")).json() == {"threads": []}, "no cookie: empty thread list")
        check((await anon.get("/api/v1/chat/threads/whatever")).status_code == 401, "no cookie: read thread -> 401")

        # ---- 2. 訪客：只拿 SSE 輸出，不留任何紀錄 ----
        res = await a.post("/api/v1/chat/tasks", json={"message": "訪客A的問題"})
        body = res.json()
        check(res.status_code == 202 and body["thread_id"] is None and body["ephemeral"],
              "guest A: message accepted, no thread created")
        check(deps.GUEST_COOKIE in a.cookies, "guest A: gets a guest session cookie (key only, no data)")
        a_task = body["task_id"]
        task_hash = await r.hgetall(tq.task_key(a_task))
        check('"ephemeral": true' in task_hash["payload"], "guest task is marked ephemeral for the worker")
        check((await a.get("/api/v1/auth/me")).json() == {"user": None, "guest": True}, "guest A: /auth/me -> guest")
        check((await a.get("/api/v1/chat/threads")).json() == {"threads": []}, "guest A: no thread list")
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM threads WHERE owner_id LIKE 'guest:%%'")
            check((await cur.fetchone())["n"] == 0, "guest A: nothing written to threads table")
        status = await a.get(f"/api/v1/chat/tasks/{a_task}")
        check(status.status_code == 200 and "user_id" not in status.json() and "payload" not in status.json(),
              "guest A: can read own task status (user_id and payload hidden)")
        check((await a.get(f"/api/v1/chat/threads/{task_hash['thread_id']}")).status_code == 404,
              "guest A: the Redis queue key is not a readable thread")
        check((await a.post("/api/v1/chat/tasks", json={"message": "x", "thread_id": "thread_abc"})).status_code == 404,
              "guest A: cannot post into a thread")
        check((await a.get("/api/v1/gcs/presign?filename=x.png")).status_code == 403,
              "guest A: cannot upload to cloud storage (no files left behind)")

        # ---- 3. 訪客 B 碰 A 的任務：404 ----
        await b.post("/api/v1/chat/tasks", json={"message": "B 的訊息"})
        check(a.cookies.get(deps.GUEST_COOKIE) != b.cookies.get(deps.GUEST_COOKIE), "guest B: different session")
        for label, coro in {
            "read A's task": b.get(f"/api/v1/chat/tasks/{a_task}"),
            "stream A's task": b.get(f"/api/v1/chat/tasks/{a_task}/events"),
            "cancel A's task": b.post(f"/api/v1/chat/tasks/{a_task}/cancel"),
        }.items():
            check((await coro).status_code == 404, f"guest B: {label} -> 404")
        check(await r.hget(tq.task_key(a_task), "status") != "cancelled", "A's task was not cancelled by B")

        # ---- 4. 註冊：訪客沒有東西要轉移；登入後的對話才會保存 ----
        check((await a.post("/api/v1/auth/register", json={"email": "bad", "password": PASSWORD})).status_code == 400,
              "register: invalid email -> 400")
        check((await a.post("/api/v1/auth/register", json={"email": email("a"), "password": "short"})).status_code == 400,
              "register: short password -> 400")
        res = await a.post("/api/v1/auth/register", json={"email": email("A").upper(), "password": PASSWORD,
                                                           "display_name": "小A"})
        check(res.status_code == 201 and res.json()["user"]["email"] == email("a"), "register: email normalized")
        user_a = res.json()["user"]["id"]
        users.append(user_a)
        check(deps.GUEST_COOKIE not in a.cookies and deps.ACCESS_COOKIE in a.cookies, "register: guest cookie replaced")
        check((await a.get("/api/v1/chat/threads")).json() == {"threads": []}, "register: guest message not saved")
        check((await anon.post("/api/v1/auth/register", json={"email": email("a"), "password": PASSWORD})).status_code
              == 409, "register: duplicate email -> 409")
        res = await a.post("/api/v1/chat/tasks", json={"message": "使用者A的對話，這句話超過二十個字所以標題要被截短"})
        check(res.status_code == 202 and res.json()["thread_id"] and not res.json()["ephemeral"],
              "user A: message creates a saved thread")
        a_thread, a_user_task = res.json()["thread_id"], res.json()["task_id"]
        lst = (await a.get("/api/v1/chat/threads")).json()["threads"]
        check([t["thread_id"] for t in lst] == [a_thread] and lst[0]["title"].endswith("..."),
              "user A: thread listed, title from server")

        # ---- 5. 使用者 B：A 的對話與任務一律 404 ----
        res = await b.post("/api/v1/auth/register", json={"email": email("b"), "password": PASSWORD})
        user_b = res.json()["user"]["id"]
        users.append(user_b)
        res = await b.post("/api/v1/chat/tasks", json={"message": "使用者B的對話"})
        b_thread = res.json()["thread_id"]
        for label, coro in {
            "read A's thread": b.get(f"/api/v1/chat/threads/{a_thread}"),
            "delete A's thread": b.delete(f"/api/v1/chat/threads/{a_thread}"),
            "post into A's thread": b.post("/api/v1/chat/tasks", json={"message": "x", "thread_id": a_thread}),
            "resume A's thread": b.post("/api/v1/chat/tasks/resume",
                                        json={"thread_id": a_thread, "decisions": [{"type": "approve"}]}),
            "read A's task": b.get(f"/api/v1/chat/tasks/{a_user_task}"),
            "stream A's task": b.get(f"/api/v1/chat/tasks/{a_user_task}/events"),
            "cancel A's task": b.post(f"/api/v1/chat/tasks/{a_user_task}/cancel"),
        }.items():
            check((await coro).status_code == 404, f"user B: {label} -> 404")
        check([t["thread_id"] for t in (await b.get("/api/v1/chat/threads")).json()["threads"]] == [b_thread],
              "user B: list shows only own thread")
        res = await b.post("/api/v1/chat/tasks", json={"message": "x", "user_id": user_a})
        check(await r.hget(tq.task_key(res.json()["task_id"]), "user_id") == user_b,
              "user_id in request body is ignored (agent runs as cookie identity)")

        # ---- 6. 偽造 / 過期 / 類型錯誤的 token（直接帶 Cookie header）----
        async with client() as evil:
            soon = datetime.now(timezone.utc) + timedelta(hours=1)
            valid = auth.create_access_token(user_a)
            check((await evil.get(f"/api/v1/chat/threads/{a_thread}", headers=as_cookie(deps.ACCESS_COOKIE, valid)))
                  .status_code == 200, "control: valid access token -> 200")
            forged = jwt.encode({"sub": user_a, "typ": "access", "exp": soon}, "x" * 40, algorithm="HS256")
            check((await evil.get(f"/api/v1/chat/threads/{a_thread}", headers=as_cookie(deps.ACCESS_COOKIE, forged)))
                  .status_code == 401, "forged signature -> 401")
            expired = jwt.encode({"sub": user_a, "typ": "access", "exp": datetime.now(timezone.utc) - timedelta(seconds=5)},
                                 os.environ["JWT_SECRET"], algorithm="HS256")
            _, guest_tok, _ = auth.create_guest_token()
            res = await evil.get("/api/v1/chat/threads",
                                 headers={"Cookie": f"{deps.ACCESS_COOKIE}={expired}; {deps.GUEST_COOKIE}={guest_tok}"})
            check(res.status_code == 401, "expired access token -> 401 (no silent fallback to guest)")
            check((await evil.get("/api/v1/chat/threads", headers=as_cookie(deps.ACCESS_COOKIE, guest_tok)))
                  .status_code == 401, "guest token used as access token -> 401")
            me = (await evil.get("/api/v1/auth/me", headers=as_cookie(deps.GUEST_COOKIE, valid))).json()
            check(me == {"user": None, "guest": False}, "access token used as guest token is ignored")

        # ---- 7. refresh 輪替與重用偵測 ----
        old_refresh = a.cookies.get(deps.REFRESH_COOKIE)
        res = await a.post("/api/v1/auth/refresh")
        check(res.status_code == 200 and a.cookies.get(deps.REFRESH_COOKIE) != old_refresh, "refresh: rotated")
        async with client() as thief:
            res = await thief.post("/api/v1/auth/refresh", headers=as_cookie(deps.REFRESH_COOKIE, old_refresh))
            check(res.status_code == 200, "refresh: old token within grace period still works (concurrent tabs)")
            auth.REFRESH_REUSE_GRACE = timedelta(seconds=0)
            res = await thief.post("/api/v1/auth/refresh", headers=as_cookie(deps.REFRESH_COOKIE, old_refresh))
            check(res.status_code == 401, "refresh: old token reused after grace -> 401")
        res = await a.post("/api/v1/auth/refresh")
        check(res.status_code == 401 and deps.REFRESH_COOKIE not in a.cookies,
              "refresh: reuse detected -> every session of that user revoked, cookies cleared")
        auth.REFRESH_REUSE_GRACE = timedelta(seconds=30)

        # ---- 8. 登入 / 登出 / 失敗次數限制 ----
        a.cookies.clear()
        check((await a.post("/api/v1/auth/login", json={"email": email("a"), "password": "wrong-password"})).status_code
              == 401, "login: wrong password -> 401")
        check((await a.post("/api/v1/auth/login", json={"email": email("nobody"), "password": PASSWORD})).status_code
              == 401, "login: unknown email -> same 401")
        res = await a.post("/api/v1/auth/login", json={"email": email("a"), "password": PASSWORD})
        check(res.status_code == 200 and res.json()["user"]["display_name"] == "小A", "login: correct password")
        check([t["thread_id"] for t in (await a.get("/api/v1/chat/threads")).json()["threads"]] == [a_thread],
              "login: history is back")
        check((await a.post("/api/v1/auth/logout")).status_code == 204 and deps.ACCESS_COOKIE not in a.cookies,
              "logout: cookies cleared")
        check((await a.get(f"/api/v1/chat/threads/{a_thread}")).status_code == 401, "logout: thread no longer readable")
        for _ in range(10):
            await anon.post("/api/v1/auth/login", json={"email": email("b"), "password": "wrong-password"})
        res = await anon.post("/api/v1/auth/login", json={"email": email("b"), "password": PASSWORD})
        check(res.status_code == 429, "login: 10 failures -> 429 even with the right password")

        # ---- 9. CSRF：跨站寫入被擋 ----
        res = await b.post("/api/v1/chat/tasks", json={"message": "x"}, headers={"Origin": "https://evil.example"})
        check(res.status_code == 403, "cross-site POST (foreign Origin) -> 403")
        res = await b.post("/api/v1/chat/tasks", json={"message": "x"}, headers={"Origin": BASE})
        check(res.status_code == 202, "same-origin POST -> allowed")

    # ---- 10. Row-Level Security：直接在資料庫層驗證（不經過應用層的 WHERE）----
    async def visible_threads(user_id):
        async with rls.as_user(pool, user_id) as conn:
            cur = await conn.execute("SELECT owner_id FROM threads")  # 故意不加 WHERE
            return {row["owner_id"] for row in await cur.fetchall()}

    check(await visible_threads(user_a) == {user_a}, "RLS threads: query without WHERE sees only A's rows")
    check(await visible_threads(user_b) == {user_b}, "RLS threads: query without WHERE sees only B's rows")
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL ROLE {rls.RLS_ROLE}")  # 忘了設定 app.user_id
            cur = await conn.execute("SELECT count(*) AS n FROM threads")
            check((await cur.fetchone())["n"] == 0, "RLS threads: user id not set -> sees nothing (deny by default)")
    try:
        async with rls.as_user(pool, user_a) as conn:
            await conn.execute("UPDATE threads SET title = 'x'")
        check(False, "RLS role is read-only")
    except InsufficientPrivilege:
        check(True, "RLS role is read-only (UPDATE -> permission denied)")

    for uid, allergy in ((user_a, "花生"), (user_b, "蝦")):
        await store.aput(("memories", uid, "diet"), "m1", {"kind": "DietProfile", "content": {"allergies": [allergy]}})
        await store.aput(("memories", uid, "kitchen"), "m1", {"kind": "KitchenProfile", "content": {"equipment": ["烤箱"]}})
        await store.aput(("inventory", uid), "items", {"items": ["雞蛋"]})
    prof_a = await rls.read_profiles(pool, user_a, DOMAINS)
    check(prof_a["diet"] == [{"allergies": ["花生"]}] and prof_a["household"] == [],
          "RLS store: read_profiles returns A's preferences")
    async with rls.as_user(pool, user_a) as conn:
        cur = await conn.execute("SELECT prefix FROM store")  # 故意不加 WHERE
        prefixes = {row["prefix"] for row in await cur.fetchall()}
        check(prefixes and all(p.split(".")[1] == user_a for p in prefixes),
              "RLS store: query without WHERE sees only A's namespaces")
        cur = await conn.execute("SELECT count(*) AS n FROM store WHERE prefix = %s", (f"memories.{user_b}.diet",))
        check((await cur.fetchone())["n"] == 0, "RLS store: asking for B's prefix explicitly still returns nothing")

    # ---- 清理測試資料 ----
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT thread_id FROM threads WHERE owner_id = ANY(%s)", (users,))
        for row in await cur.fetchall():
            await history.delete_thread(row["thread_id"])
        await conn.execute("DELETE FROM threads WHERE owner_id = ANY(%s)", (users,))
        await conn.execute("DELETE FROM users WHERE email LIKE %s", (f"itest-{RUN}-%",))
        await conn.execute("DELETE FROM store WHERE split_part(prefix, '.', 2) = ANY(%s)", (users,))
    await r.flushdb()
    await tq.close_redis()
    await history.close_history()
    await db.close_pool()
    print("ALL PASSED")


asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
