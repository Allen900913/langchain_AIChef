"""整合測試：登入系統、資料隔離與 Row-Level Security。真 Postgres + 真 Redis（DB 14）+ 真 FastAPI app，不啟動 worker。

執行（需要 docker compose 的 redis / postgres 已啟動，且 worker 至少啟動過一次建好 store 資料表；
會清空 Redis DB 14，測試帳號、對話與偏好資料結束時刪除）：
    PYTHONPATH=. .venv/Scripts/python.exe app/tests/integration_auth.py

涵蓋：訪客不留紀錄（不建對話、沒有清單、不能上傳到雲端）、訪客只看得到自己的任務、
別人的對話 / 任務一律 404、refresh token 輪替與重用偵測、過期 / 偽造 / 類型錯誤的 token、
CSRF Origin 檢查、登入失敗次數限制、輸入長度限制、對話清單與偏好讀取的 RLS（直接在資料庫層驗證）。
"""
import asyncio
import os
import sys
import time
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
from app.common import rate_limit  # noqa: E402
from app.common import rls  # noqa: E402
from app.common import task_queue as tq  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schemas import (  # noqa: E402
    DECISION_MESSAGE_MAX_CHARS,
    DECISIONS_MAX,
    MESSAGE_MAX_CHARS,
    THREAD_ID_MAX_CHARS,
)
from app.services import auth, db, history  # noqa: E402

BASE = "https://t"
RUN = uuid.uuid4().hex[:8]
# 其他段落會送很多訊息，先關掉；9c 段才打開測試
GUEST_LIMITS, USER_LIMITS = rate_limit.GUEST_LIMITS, rate_limit.USER_LIMITS
rate_limit.GUEST_LIMITS = rate_limit.USER_LIMITS = ()
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
        grace = auth.REFRESH_REUSE_GRACE

        def sets_cookie(res, name):
            return any(h.startswith(f"{name}=") for h in res.headers.get_list("set-cookie"))

        async def refresh_with(token):
            async with client() as c:
                return await c.post("/api/v1/auth/refresh", headers=as_cookie(deps.REFRESH_COOKIE, token))

        async def login_a():
            async with client() as c:
                await c.post("/api/v1/auth/login", json={"email": email("a"), "password": PASSWORD})
                return c.cookies.get(deps.REFRESH_COOKIE)

        async def live_tokens():
            """還能往下換發的 token 數（沒被撤銷、也還沒換發過）。"""
            async with pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT count(*) AS n FROM refresh_tokens"
                    " WHERE user_id = %s AND revoked_at IS NULL AND rotated_at IS NULL",
                    (user_a,),
                )
                return (await cur.fetchone())["n"]

        old_refresh = a.cookies.get(deps.REFRESH_COOKIE)
        live = await live_tokens()
        res = await a.post("/api/v1/auth/refresh")
        check(res.status_code == 200 and a.cookies.get(deps.REFRESH_COOKIE) != old_refresh, "refresh: rotated")
        res = await refresh_with(old_refresh)
        check(res.status_code == 200 and sets_cookie(res, deps.ACCESS_COOKIE)
              and not sets_cookie(res, deps.REFRESH_COOKIE),
              "refresh: old token within grace -> new access token only, no second refresh token")
        check(await live_tokens() == live, "refresh: replay within grace does not fork the chain")

        # 兩個分頁同時換發：都成功，但只有一個拿到新的 refresh token
        token = await login_a()
        live = await live_tokens()
        both = await asyncio.gather(refresh_with(token), refresh_with(token))
        check([x.status_code for x in both] == [200, 200], "refresh: concurrent tabs -> both succeed")
        check(sum(sets_cookie(x, deps.REFRESH_COOKIE) for x in both) == 1, "refresh: concurrent tabs -> one new token")
        check(await live_tokens() == live, "refresh: concurrent tabs -> chain does not fork")

        # 超過寬限期才重放：判定外洩，全部撤銷
        auth.REFRESH_REUSE_GRACE = timedelta(seconds=0)
        check((await refresh_with(old_refresh)).status_code == 401, "refresh: old token reused after grace -> 401")
        auth.REFRESH_REUSE_GRACE = grace
        check(await live_tokens() == 0, "refresh: reuse detected -> every token of that user revoked")
        # 這張是最新的 token，撤銷才剛發生、還在寬限時間內，也不能再換發
        res = await a.post("/api/v1/auth/refresh")
        check(res.status_code == 401 and deps.REFRESH_COOKIE not in a.cookies,
              "refresh: newest token right after revocation -> 401 (no grace), cookies cleared")
        check(await live_tokens() == 0, "refresh: nothing new issued after revocation")

        # 登出後立刻重放：撤銷不適用寬限期
        token = await login_a()
        async with client() as c:
            res = await c.post("/api/v1/auth/logout", headers=as_cookie(deps.REFRESH_COOKIE, token))
            check(res.status_code == 204, "logout with refresh cookie -> 204")
        check((await refresh_with(token)).status_code == 401, "refresh: token right after logout -> 401 (no grace)")

        # 駭客先換發：使用者的舊 token 在寬限期內只拿到 access，超過寬限期再用就全部撤銷，駭客也被踢出去
        stolen = await login_a()
        async with client() as thief:
            res = await thief.post("/api/v1/auth/refresh", headers=as_cookie(deps.REFRESH_COOKIE, stolen))
            thief_token = thief.cookies.get(deps.REFRESH_COOKIE)
        check(res.status_code == 200 and thief_token and thief_token != stolen, "refresh: thief rotates first")
        res = await refresh_with(stolen)
        check(res.status_code == 200 and not sets_cookie(res, deps.REFRESH_COOKIE),
              "refresh: victim within grace -> access token only")
        auth.REFRESH_REUSE_GRACE = timedelta(seconds=0)
        check((await refresh_with(stolen)).status_code == 401, "refresh: victim after grace -> reuse detected")
        auth.REFRESH_REUSE_GRACE = grace
        check((await refresh_with(thief_token)).status_code == 401, "refresh: thief's token is dead too")
        check(await live_tokens() == 0, "refresh: nothing left alive after detection")

        # ---- 7b. 閒置期限與絕對期限 ----
        async def token_row(token):
            async with pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT expires_at, absolute_expires_at, rotated_at FROM refresh_tokens WHERE token_hash = %s",
                    (auth._hash_token(token),),
                )
                return await cur.fetchone()

        async def set_deadlines(token, expires_at, absolute_expires_at):
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE refresh_tokens SET expires_at = %s, absolute_expires_at = %s WHERE token_hash = %s",
                    (expires_at, absolute_expires_at, auth._hash_token(token)),
                )

        token = await login_a()
        row, now = await token_row(token), datetime.now(timezone.utc)
        check(abs(row["expires_at"] - now - auth.REFRESH_TTL) < timedelta(minutes=1), "lifetime: login -> idle deadline set")
        check(abs(row["absolute_expires_at"] - now - auth.REFRESH_ABSOLUTE_TTL) < timedelta(minutes=1),
              "lifetime: login -> absolute deadline set")

        # 不斷換發：閒置期限往後推，絕對期限完全不動
        deadline = row["absolute_expires_at"]
        await set_deadlines(token, now + timedelta(hours=1), deadline)
        for _ in range(20):
            res = await refresh_with(token)
            token = res.cookies.get(deps.REFRESH_COOKIE)
            check(res.status_code == 200 and token, "lifetime: rotation succeeds before the deadline")
        row = await token_row(token)
        check(row["absolute_expires_at"] == deadline, "lifetime: 20 rotations do not extend the absolute deadline")
        check(row["expires_at"] > now + timedelta(hours=1), "lifetime: rotation pushes the idle deadline forward")

        # 快到絕對期限時換發：新 token 不能活超過絕對期限
        soon = datetime.now(timezone.utc) + timedelta(hours=1)
        await set_deadlines(token, soon, soon)
        res = await refresh_with(token)
        token = res.cookies.get(deps.REFRESH_COOKIE)
        check(res.status_code == 200 and (await token_row(token))["expires_at"] == soon,
              "lifetime: token issued near the deadline expires at the deadline, not 7 days later")

        # 到了絕對期限：閒置期限還沒到也一樣失效，駭客下一次換發一定失敗
        other_device = await login_a()
        live = await live_tokens()
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        await set_deadlines(token, datetime.now(timezone.utc) + timedelta(days=1), past)
        res = await refresh_with(token)
        check(res.status_code == 401, "lifetime: past the absolute deadline -> 401 even if the idle deadline is ahead")
        check(any(h.startswith(f"{deps.REFRESH_COOKIE}=") and "Max-Age=0" in h for h in res.headers.get_list("set-cookie")),
              "lifetime: past the absolute deadline -> cookies cleared")
        check(await live_tokens() == live, "lifetime: past the absolute deadline -> no new token issued")
        res = await refresh_with(other_device)
        check(res.status_code == 200, "lifetime: another device's login is not affected")
        res = await refresh_with(res.cookies.get(deps.REFRESH_COOKIE))
        check(res.status_code == 200, "lifetime: reaching the deadline is not treated as token theft")

        # 閒置太久：絕對期限還沒到也失效
        token = await login_a()
        await set_deadlines(token, past, datetime.now(timezone.utc) + timedelta(days=1))
        check((await refresh_with(token)).status_code == 401, "lifetime: past the idle deadline -> 401")

        # 寬限期內重送也不能超過期限
        token = await login_a()
        res = await refresh_with(token)
        await set_deadlines(token, datetime.now(timezone.utc) + timedelta(days=1), past)
        check((await refresh_with(token)).status_code == 401, "lifetime: replay within grace past the deadline -> 401")

        # 重新輸入密碼：絕對期限重新起算
        token = await login_a()
        check((await token_row(token))["absolute_expires_at"] > deadline, "lifetime: new password login starts a new deadline")

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

        # ---- 9b. 輸入長度限制 ----
        async def send(**body):
            return (await b.post("/api/v1/chat/tasks", json=body)).status_code

        check(await send(message="字" * MESSAGE_MAX_CHARS) == 202, "message: exactly at the limit -> accepted")
        check(await send(message="字" * (MESSAGE_MAX_CHARS + 1)) == 422, "message: one over the limit -> 422")
        check(await send(message="字" * 2_000_000) == 422, "message: 2MB -> 422")
        check(await send(message=f"  {'字' * MESSAGE_MAX_CHARS}\n ") == 202, "message: surrounding whitespace not counted")
        for label, blank in {"empty": "", "spaces": "   ", "newlines and full-width spaces": "\n　\t"}.items():
            check(await send(message=blank) == 422, f"message: {label} -> 422")
        res = await b.post("/api/v1/chat/tasks", json={"message": "  前後有空白\n"})
        stored = await r.hget(tq.task_key(res.json()["task_id"]), "payload")
        check('"前後有空白"' in stored, "message: stored without surrounding whitespace")
        check((await anon.post("/api/v1/chat/tasks", json={"message": "字" * (MESSAGE_MAX_CHARS + 1)})).status_code
              == 422, "guest message: over the limit -> 422")
        check(await send(message="x", thread_id="t" * (THREAD_ID_MAX_CHARS + 1)) == 422, "thread_id: too long -> 422")

        async def resume(thread_id, decisions):
            res = await b.post("/api/v1/chat/tasks/resume", json={"thread_id": thread_id, "decisions": decisions})
            return res.status_code

        check(await resume(b_thread, [{"type": "approve"}] * (DECISIONS_MAX + 1)) == 422, "resume: too many decisions -> 422")
        check(await resume(b_thread, [{"type": "reject", "message": "字" * (DECISION_MESSAGE_MAX_CHARS + 1)}]) == 422,
              "resume: decision message too long -> 422")
        check(await resume(b_thread, [{"type": "maybe"}]) == 422, "resume: unknown decision type -> 422")
        check(await resume("t" * (THREAD_ID_MAX_CHARS + 1), [{"type": "approve"}]) == 422, "resume: thread_id too long -> 422")
        check(await resume(b_thread, [{"type": "reject", "message": "使用者拒絕執行此工具"}]) == 409,
              "resume: the frontend's reject shape passes validation (409: nothing pending)")

        async def register(**body):
            return (await anon.post("/api/v1/auth/register", json={"password": PASSWORD, **body})).status_code

        check(await register(email=email("long-name"), display_name="名" * auth.DISPLAY_NAME_MAX) == 201,
              "register: display_name at the limit -> accepted")
        anon.cookies.clear()
        check(await register(email=email("too-long-name"), display_name="名" * (auth.DISPLAY_NAME_MAX + 1)) == 422,
              "register: display_name over the limit -> 422")
        check((await anon.post("/api/v1/auth/register", json={"email": email("x"), "password": "p" * (auth.PASSWORD_MAX + 1)}))
              .status_code == 422, "register: password over the limit -> 422")
        long_email = "x" * auth.EMAIL_MAX + "@example.com"
        check(await register(email=long_email) == 422, "register: email over the limit -> 422")
        res = await anon.post("/api/v1/auth/login", json={"email": long_email, "password": PASSWORD})
        check(res.status_code == 422 and not await r.exists(f"chef:auth:fail:{long_email}"),
              "login: email over the limit -> 422, no Redis key created")
        check((await anon.post("/api/v1/auth/login", json={"email": email("a"), "password": "p" * (auth.PASSWORD_MAX + 1)}))
              .status_code == 422, "login: password over the limit -> 422")
        # 這個字串在長度檢查之前會讓 email regex 跑很久（耗時隨長度平方成長），而且卡住整個 event loop
        crafted = "a@" + "." * 200_000 + " x"
        started = time.perf_counter()
        for path in ("/api/v1/auth/login", "/api/v1/auth/register"):
            await anon.post(path, json={"email": crafted, "password": PASSWORD})
        try:
            auth.normalize_email(crafted)
            rejected = False
        except auth.AuthError:
            rejected = True
        check(rejected and time.perf_counter() - started < 2, "email regex: crafted long input rejected without running it")

        # ---- 9c. 速率限制：訪客看 IP，登入使用者看 user_id ----
        rate_limit.GUEST_LIMITS, rate_limit.USER_LIMITS = GUEST_LIMITS, USER_LIMITS
        guest_per_minute, user_per_minute = GUEST_LIMITS[0].count, USER_LIMITS[0].count

        async def reset_limits():
            for key in await r.keys("chef:rate:*"):
                await r.delete(key)

        async def guest_send(forwarded=None, peer="127.0.0.1"):
            """每次都是全新的 client（沒有 cookie），等於使用者開無痕或清掉 cookie。"""
            transport = httpx.ASGITransport(app=app, client=(peer, 123))
            async with httpx.AsyncClient(transport=transport, base_url=BASE, timeout=30) as c:
                headers = {"X-Forwarded-For": forwarded} if forwarded else {}
                return await c.post("/api/v1/chat/tasks", json={"message": "x"}, headers=headers)

        def codes(results):
            return sorted(res.status_code for res in results)

        # 訪客：同一個 IP，同時送也只有上限內的通過；清 cookie 沒有用
        ip_one, ip_two = "203.0.113.1", "203.0.113.2"
        burst = await asyncio.gather(*(guest_send(ip_one) for _ in range(20)))
        check(codes(burst) == [202] * guest_per_minute + [429] * (20 - guest_per_minute),
              "rate limit guest: 20 at once, new cookie each -> only the limit passes")
        blocked = next(res for res in burst if res.status_code == 429)
        check(0 < int(blocked.headers["Retry-After"]) <= 60 and "秒後再試" in blocked.json()["detail"],
              "rate limit guest: 429 has Retry-After and a readable message")
        check((await guest_send(ip_two)).status_code == 202, "rate limit guest: another IP has its own limit")

        # 偽造：只有最右邊（Caddy 寫入）才算數；外部直連時整個標頭都不信
        check((await guest_send(f"198.51.100.9, {ip_one}")).status_code == 429,
              "rate limit guest: forged left-hand X-Forwarded-For values are ignored")
        await reset_limits()
        for i in range(guest_per_minute):
            check((await guest_send(f"203.0.113.{10 + i}", peer="8.8.8.8")).status_code == 202,
                  "rate limit guest: direct connection counted by its own address")
        check((await guest_send("203.0.113.99", peer="8.8.8.8")).status_code == 429,
              "rate limit guest: X-Forwarded-For from a public peer cannot reset the limit")

        # IPv6：同一段 /64 算同一個來源
        await reset_limits()
        for i in range(guest_per_minute):
            await guest_send(f"2001:db8:1:1::{i + 1}")
        check((await guest_send("2001:db8:1:1::ffff")).status_code == 429, "rate limit guest: same IPv6 /64 shares a limit")
        check((await guest_send("2001:db8:1:2::1")).status_code == 202, "rate limit guest: another IPv6 /64 is separate")

        # 登入使用者：看 user_id，換 IP 也是同一份額度；被擋的訊息不會多一個對話
        await reset_limits()
        threads_before = len((await b.get("/api/v1/chat/threads")).json()["threads"])
        burst = await asyncio.gather(*(b.post("/api/v1/chat/tasks", json={"message": "x"}) for _ in range(20)))
        check(codes(burst) == [202] * user_per_minute + [429] * (20 - user_per_minute),
              "rate limit user: 20 at once -> only the limit passes")
        check(len((await b.get("/api/v1/chat/threads")).json()["threads"]) == threads_before + user_per_minute,
              "rate limit user: rejected messages create no thread")
        res = await b.post("/api/v1/chat/tasks", json={"message": "x"}, headers={"X-Forwarded-For": "198.51.100.77"})
        check(res.status_code == 429, "rate limit user: changing IP does not reset the limit")
        check((await guest_send()).status_code == 202, "rate limit user: does not use up the guest limit of that IP")
        async with client() as other:
            await other.post("/api/v1/auth/login", json={"email": email("a"), "password": PASSWORD})
            check((await other.post("/api/v1/chat/tasks", json={"message": "x"})).status_code == 202,
                  "rate limit user: another user has their own limit")

        # 冪等重送不扣額度；被擋下的 key 不會卡住
        await reset_limits()
        rate_limit.USER_LIMITS = (rate_limit.Limit(1, 60),)

        async def send_with_key(key):
            return await b.post("/api/v1/chat/tasks", json={"message": "同一則"}, headers={"Idempotency-Key": key})

        first, second = f"rate-{RUN}-first-0000", f"rate-{RUN}-second-000"
        check((await send_with_key(first)).status_code == 202, "rate limit: first message accepted")
        res = await send_with_key(first)
        check(res.status_code == 202 and res.json()["replayed"], "rate limit: idempotent replay passes at the limit")
        check((await send_with_key(second)).status_code == 429, "rate limit: a new message at the limit -> 429")
        await reset_limits()
        res = await send_with_key(second)
        check(res.status_code == 202 and not res.json().get("replayed"),
              "rate limit: a rejected Idempotency-Key is released and can be sent again")

        # 被擋下的請求不計數；時間窗過了就恢復；每日上限的提示不同
        await reset_limits()
        rate_limit.USER_LIMITS = (rate_limit.Limit(1, 1), rate_limit.Limit(2, 24 * 60 * 60))
        check((await b.post("/api/v1/chat/tasks", json={"message": "x"})).status_code == 202, "rate limit: first of the day")
        for _ in range(5):
            await b.post("/api/v1/chat/tasks", json={"message": "x"})
        await asyncio.sleep(1.2)
        check((await b.post("/api/v1/chat/tasks", json={"message": "x"})).status_code == 202,
              "rate limit: window expired, and rejected requests did not use up the daily limit")
        await asyncio.sleep(1.2)
        res = await b.post("/api/v1/chat/tasks", json={"message": "x"})
        check(res.status_code == 429 and int(res.headers["Retry-After"]) > 3600 and "今天" in res.json()["detail"],
              "rate limit: daily limit reached -> long Retry-After and a daily message")
        rate_limit.GUEST_LIMITS = rate_limit.USER_LIMITS = ()

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
