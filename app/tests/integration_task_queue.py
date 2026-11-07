"""整合測試：真 Redis（DB 15）+ 真 FastAPI app + 真 worker 迴圈，agent 換成假的。

執行（需要 docker compose 的 redis / postgres 已啟動；會清空 Redis DB 15）：
    PYTHONPATH=. .venv/Scripts/python.exe app/tests/integration_task_queue.py

涵蓋：enqueue 排隊順序、HITL 暫停佇列（hold）、略過審核、resume 插隊/衝突、
停止（排隊中 / 等 worker / 執行中 / 不聽話逾時）、冪等、崩潰回收、刪除對話、
訪客訊息（ephemeral：不建對話、依序執行、別人看不到）、
送訊息的冪等（重送不多跑、相同文字的新訊息照跑、同時重送、建立失敗、key 依使用者分開）。
主要情境以登入使用者執行（測試開頭註冊一個帳號，結束時刪除）。
"""
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
# 沿用 .env 的 Redis 位址與密碼，只換成測試專用的 DB 15
os.environ["REDIS_URL"] = os.getenv("REDIS_URL", "redis://localhost:6389/0").rsplit("/", 1)[0] + "/15"
os.environ["WORKER_CONCURRENCY"] = "2"
sys.stdout.reconfigure(encoding="utf-8")

import httpx  # noqa: E402

import app.worker as worker  # noqa: E402
from app.common import rate_limit  # noqa: E402
from app.common import task_queue as tq  # noqa: E402
from app.main import app  # noqa: E402
from app.services import db, history  # noqa: E402

# 這裡測的是佇列，要連續送很多訊息；速率限制由 integration_auth.py 測
rate_limit.GUEST_LIMITS = rate_limit.USER_LIMITS = ()

running_now = 0
max_running = 0
order: list[str] = []
saved_cancel: list[tuple] = []
fake_interrupts: dict[str, dict] = {}   # thread_id → pending interrupt（模擬 checkpointer）
stuck_threads: set[str] = set()
created_threads: set[str] = set()        # 測試建立的對話，結束時從 threads 表刪除
ephemeral_seen: dict[str, bool] = {}     # message → worker 傳給 call_agent 的 ephemeral 旗標          # 已進入「不理會取消」區段的 stubborn 任務


async def fake_call_agent(message, image_url, thread_id, user_id, ephemeral=False):
    ephemeral_seen[message] = ephemeral

    async def gen():
        global running_now, max_running
        running_now += 1
        max_running = max(max_running, running_now)
        order.append(message)
        try:
            if message.startswith("hitl"):
                # 真的 agent 走到 HITL 前至少要一次模型呼叫；停一下，讓「執行中又送出」的情境不靠 API 搶快
                await asyncio.sleep(0.3)
                yield ("tool_start", {"tool": "web_search", "args": {"query": "x"}, "id": "c1"})
                fake_interrupts[thread_id] = {"action_requests": [{"name": "web_search", "args": {}}],
                                              "review_configs": []}
                yield ("interrupt", {"type": "interrupt", "action_requests": [{"name": "web_search", "args": {}}]})
                return
            fake_interrupts.pop(thread_id, None)  # 新訊息讓 graph 往前走，中斷被丟棄
            if message.startswith("slow"):
                yield ("token", {"delta": "半段"})
                yield ("token", {"delta": "回答"})
                await asyncio.sleep(30)
                yield ("token", {"delta": "不該出現"})
                yield ("done", {"status": "complete"})
                return
            if message.startswith("stubborn"):
                # 不能先 yield：停止若剛好在 worker 發布這個 token 時送到，generator 停在 yield 上
                # 會被直接關掉，根本走不到下面「不理會取消」的區段，測到的就只是一般停止
                stuck_threads.add(thread_id)
                try:  # 不理會取消（模擬卡在無法中斷的地方），4 秒後才自己結束
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    await asyncio.sleep(4)
                yield ("token", {"delta": "late"})
                yield ("done", {"status": "complete"})
                return
            if message == "boom":
                yield ("token", {"delta": "half"})
                raise RuntimeError("agent exploded")
            for i in range(3):
                await asyncio.sleep(0.1)
                yield ("token", {"delta": f"{message}-{i}"})
            yield ("done", {"status": "complete"})
        finally:
            running_now -= 1
    return gen()


def fake_resume_agent(thread_id, decisions, user_id):
    async def gen():
        order.append(f"resume:{decisions[0]['type']}")
        fake_interrupts.pop(thread_id, None)
        await asyncio.sleep(0.2)
        yield ("token", {"delta": f"resumed:{decisions[0]['type']}"})
        yield ("done", {"status": "complete"})
    return gen()


async def fake_thread_state(thread_id):
    return [], fake_interrupts.get(thread_id)


async def noop(*a, **k):
    return None


async def fake_save_cancelled(thread_id, kind, boundary_id, partial):
    saved_cancel.append((thread_id, kind, partial))


worker.call_agent = fake_call_agent
worker.resume_agent = fake_resume_agent
worker.init_agent_infra = noop
worker.close_agent_infra = noop
worker.get_last_message_id = noop
worker.save_cancelled_turn = fake_save_cancelled
history.get_thread_state = fake_thread_state
# 停止逾時才強制收尾。須大於正常停止所需時間（實測約 1.5 秒），否則正常停止的任務也會被強制收尾、
# 拿不到部分回答；1.5 秒在機器忙的時候不夠
worker.CANCEL_DEADLINE = 3.0
worker.CANCEL_POLL_INTERVAL = 0.3
worker.REAP_INTERVAL = 0.3


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        ev = {}
        for line in block.split("\n"):
            if line.startswith("id: "):
                ev["id"] = line[4:]
            elif line.startswith("event: "):
                ev["event"] = line[7:]
            elif line.startswith("data: "):
                ev["data"] = json.loads(line[6:])
        if "event" in ev:
            events.append(ev)
    return events


async def wait_status(r, task_id, statuses, timeout=10):
    st = None
    for _ in range(int(timeout / 0.05)):
        st = await r.hget(tq.task_key(task_id), "status")
        if st in statuses:
            return st
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} stuck at {st}")


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        raise SystemExit(1)


async def main():
    r = tq.get_redis()
    await r.flushdb()
    await history.init_history()
    await db.migrate()  # threads 表（對話擁有權）
    worker_task = asyncio.create_task(worker.main())
    await asyncio.sleep(0.3)

    email = f"itest-queue-{uuid.uuid4().hex[:8]}@example.com"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://t", timeout=60) as c:
        res = await c.post("/api/v1/auth/register", json={"email": email, "password": "queue-test-password"})
        assert res.status_code == 201, res.text

        async def post(msg, tid=None):
            res = await c.post("/api/v1/chat/tasks", json={"message": msg, "thread_id": tid})
            assert res.status_code == 202, res.text
            created_threads.add(res.json()["thread_id"])
            return res.json()

        async def events(task_id, **kw):
            return parse_sse((await c.get(f"/api/v1/chat/tasks/{task_id}/events", **kw)).text)

        def tokens(evs):
            return "".join(e["data"].get("delta", "") for e in evs if e["event"] == "token")

        # ---- 1. enqueue：同一對話連發三則，依序執行 ----
        a = await post("m1")
        tid = a["thread_id"]
        check(a["status"] == "pending", "first message starts immediately")
        b = await post("m2", tid)
        c3 = await post("m3", tid)
        check(b["status"] == "queued" and b["position"] == 1 and c3["position"] == 2,
              f"follow-ups are queued in order (positions {b['position']}, {c3['position']})")
        await wait_status(r, a["task_id"], {"processing"})
        th = (await c.get(f"/api/v1/chat/threads/{tid}")).json()
        check(th["active_task"]["task_id"] == a["task_id"]
              and [q["task_id"] for q in th["queued_tasks"]] == [b["task_id"], c3["task_id"]]
              and th["queued_tasks"][0]["user_message"] == "m2",
              "thread view: active task + queued tasks with their messages")
        order.clear()
        evs_c = await events(c3["task_id"])  # 從排隊中就開始訂閱，一路等到它跑完
        check(order == ["m2", "m3"] or order == ["m1", "m2", "m3"][1:], f"queue drained in order ({order})")
        check(evs_c[0]["event"] == "started" and tokens(evs_c) == "m3-0m3-1m3-2" and evs_c[-1]["event"] == "done",
              "subscribing while queued waits, then streams started → tokens → done")
        check(not await tq.is_thread_busy(r, tid), "lock released after queue drained")

        # ---- 2. 撤回排隊中的訊息 ----
        s = await post("slow-1")
        tid2 = s["thread_id"]
        q1 = await post("q1", tid2)
        q2 = await post("q2", tid2)
        res = (await c.post(f"/api/v1/chat/tasks/{q1['task_id']}/cancel")).json()
        check(res["status"] == "cancelled", "cancel queued task -> cancelled immediately")
        evs = await events(q1["task_id"])
        check([e["event"] for e in evs] == ["status", "done"] and evs[0]["data"]["code"] == "CANCELLED",
              "cancelled queued task stream: status(CANCELLED) + done")
        th = (await c.get(f"/api/v1/chat/threads/{tid2}")).json()
        check([q["task_id"] for q in th["queued_tasks"]] == [q2["task_id"]], "withdrawn message left the queue")

        # ---- 3. 停止執行中的任務 → 半段回答交給整理流程 → 下一則自動開始 ----
        await wait_status(r, s["task_id"], {"processing"})
        # 等兩個 token 都發布了才停止：機器忙的時候固定等 0.2 秒不一定夠，停止會搶在「回答」之前
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            published = await r.xrange(tq.events_key(s["task_id"]))
            if sum(1 for _, f in published if f.get("event") == "token") >= 2:
                break
            await asyncio.sleep(0.05)
        t0 = time.monotonic()
        res = (await c.post(f"/api/v1/chat/tasks/{s['task_id']}/cancel")).json()
        check(res["status"] == "cancelling", "cancel processing task -> cancelling")
        evs = await events(s["task_id"])
        check(evs[-2]["event"] == "status" and evs[-2]["data"]["code"] == "CANCELLED" and evs[-1]["event"] == "done"
              and "不該出現" not in tokens(evs), f"running task stopped within {time.monotonic() - t0:.2f}s")
        check(await r.hget(tq.task_key(s["task_id"]), "status") == "cancelled", "status = cancelled (not completed)")
        check(saved_cancel and saved_cancel[-1][2] == "半段回答", f"partial answer handed to cleanup ({saved_cancel[-1:]})")
        await wait_status(r, q2["task_id"], {"completed"})
        check(True, "next queued message ran after stop")
        res = (await c.post(f"/api/v1/chat/tasks/{s['task_id']}/cancel")).json()
        check(res["status"] == "cancelled", "cancel is idempotent (already cancelled)")
        res = (await c.post(f"/api/v1/chat/tasks/{q2['task_id']}/cancel")).json()
        check(res["status"] == "completed", "cancel completed task -> no-op, returns completed")
        check((await c.post("/api/v1/chat/tasks/nope/cancel")).status_code == 404, "cancel unknown task -> 404")

        # ---- 4. 停止還在全域佇列等 worker 的任務（兩個 worker 位子都被佔滿）----
        busy1 = await post("slow-b1")
        busy2 = await post("slow-b2")
        await wait_status(r, busy1["task_id"], {"processing"})
        await wait_status(r, busy2["task_id"], {"processing"})
        p = await post("p1")
        check(p["status"] == "pending", "third thread's task waits in global queue (pending)")
        p_next = await post("p2", p["thread_id"])
        res = (await c.post(f"/api/v1/chat/tasks/{p['task_id']}/cancel")).json()
        check(res["status"] == "cancelled", "cancel pending task -> cancelled")
        check(await r.hget(tq.task_key(p_next["task_id"]), "status") == "pending",
              "its queued follow-up was promoted to pending")
        for t in (busy1, busy2):
            await c.post(f"/api/v1/chat/tasks/{t['task_id']}/cancel")
        await wait_status(r, p_next["task_id"], {"completed"})
        check(order.count("p1") == 0, "cancelled pending task never ran")

        # ---- 5. HITL：中斷暫停佇列（hold）；審核插隊；之後佇列繼續 ----
        h = await post("hitl-1")
        htid = h["thread_id"]
        hq = await post("after-hitl", htid)
        check(hq["status"] == "queued", "message sent while running is queued")
        await wait_status(r, h["task_id"], {"interrupted"})
        await asyncio.sleep(0.3)
        check(await r.hget(tq.task_key(hq["task_id"]), "status") == "queued", "queue is held while HITL pending")
        th = (await c.get(f"/api/v1/chat/threads/{htid}")).json()
        check(th["pending_interrupt"] and th["held"] and th["active_task"] is None
              and th["queued_tasks"][0]["task_id"] == hq["task_id"], "thread view: pending card + held queue")
        hq2 = await post("after-hitl-2", htid)
        check(hq2["status"] == "queued" and hq2["held"], "new message while held goes behind (held=True)")
        bad = await c.post("/api/v1/chat/tasks/resume", json={"thread_id": htid, "decisions": []})
        check(bad.status_code == 422, "resume with wrong decision count -> 422")
        order.clear()
        res = await c.post("/api/v1/chat/tasks/resume", json={"thread_id": htid, "decisions": [{"type": "approve"}]})
        check(res.status_code == 202, "resume accepted")
        rt = res.json()["task_id"]
        await wait_status(r, hq2["task_id"], {"completed"})
        check(order == ["resume:approve", "after-hitl", "after-hitl-2"], f"resume jumped the queue, then queue resumed ({order})")
        stale = await c.post("/api/v1/chat/tasks/resume", json={"thread_id": htid, "decisions": [{"type": "approve"}]})
        check(stale.status_code == 409, "resume when no pending interrupt -> 409 (stale card)")

        # ---- 6. 略過審核：沒有排隊、直接送新訊息 → 立即開始 ----
        h2 = await post("hitl-2")
        await wait_status(r, h2["task_id"], {"interrupted"})
        skip = await post("never mind", h2["thread_id"])
        check(skip["status"] == "pending" and not skip["held"], "new message with pending card (empty queue) starts now")
        await wait_status(r, skip["task_id"], {"completed"})
        th = (await c.get(f"/api/v1/chat/threads/{h2['thread_id']}")).json()
        check(th["pending_interrupt"] is None, "card gone after skipping")

        # ---- 7. resume 與執行中任務衝突 ----
        h3 = await post("hitl-3")
        await wait_status(r, h3["task_id"], {"interrupted"})
        run3 = await post("slow-3", h3["thread_id"])
        await wait_status(r, run3["task_id"], {"processing"})
        fake_interrupts[h3["thread_id"]] = {"action_requests": [{"name": "x", "args": {}}], "review_configs": []}
        res = await c.post("/api/v1/chat/tasks/resume", json={"thread_id": h3["thread_id"], "decisions": [{"type": "approve"}]})
        check(res.status_code == 409, "resume while a task is running -> 409")
        fake_interrupts.pop(h3["thread_id"], None)
        check((await c.delete(f"/api/v1/chat/threads/{h3['thread_id']}")).status_code == 409, "delete busy thread -> 409")
        await c.post(f"/api/v1/chat/tasks/{run3['task_id']}/cancel")
        await wait_status(r, run3["task_id"], {"cancelled"})

        # ---- 8. 不聽話的任務：停止逾時由回收器強制收尾，排隊的下一則照常開始 ----
        st = await post("stubborn-1")
        st_next = await post("after-stubborn", st["thread_id"])
        while st["thread_id"] not in stuck_threads:
            await asyncio.sleep(0.05)
        await c.post(f"/api/v1/chat/tasks/{st['task_id']}/cancel")
        t0 = time.monotonic()
        evs = await events(st["task_id"])
        elapsed = time.monotonic() - t0
        check(evs[-1]["event"] == "done" and await r.hget(tq.task_key(st["task_id"]), "status") == "cancelled"
              and elapsed >= worker.CANCEL_DEADLINE,
              f"stubborn task force-cancelled after deadline ({elapsed:.1f}s)")
        await wait_status(r, st_next["task_id"], {"completed"})
        check(True, "queue continued after forced cancel")
        # 卡住的 agent 恢復後：巡查發現已被強制收尾 → 再取消一次，不整理 checkpointer、不再發事件
        deadline = time.monotonic() + 10
        while st["task_id"] in worker._running and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        check(st["task_id"] not in worker._running, "stuck worker abandons force-cancelled task")
        check(not any(c[0] == st["thread_id"] for c in saved_cancel),
              "no checkpointer cleanup after lock was handed to the next task")
        late = [f for _id, f in await r.xrange(tq.events_key(st["task_id"])) if "late" in f.get("data", "")]
        check(not late, "abandoned task publishes nothing after its done event")

        # ---- 9. agent 例外 → failed + error + done，佇列照常推進 ----
        bm = await post("boom")
        bm_next = await post("after-boom", bm["thread_id"])
        check(await wait_status(r, bm["task_id"], {"failed", "completed"}) == "failed", "agent exception -> failed")
        evs = await events(bm["task_id"])
        check([e["event"] for e in evs][-2:] == ["error", "done"], "failed task ends with error + done")
        await wait_status(r, bm_next["task_id"], {"completed"})
        check(True, "queue advanced after failure")

        # ---- 10. 併發上限 + Last-Event-ID ----
        global max_running
        await asyncio.sleep(0.5)
        max_running = 0
        ids = [(await post(f"c{i}"))["task_id"] for i in range(4)]
        for t in ids:
            await wait_status(r, t, {"completed"})
        check(max_running <= 2, f"worker respects WORKER_CONCURRENCY=2 (max seen {max_running})")
        evs = await events(ids[0])
        evs2 = await events(ids[0], headers={"Last-Event-ID": evs[1]["id"]})
        check(tokens(evs2) == "c0-1c0-2", f"Last-Event-ID resumes after given id ({tokens(evs2)})")
        check((await c.get("/api/v1/chat/tasks/nope/events")).status_code == 404, "unknown task events -> 404")

        # ---- 11. 刪除對話：清掉排隊中的訊息 ----
        d = await post("hitl-del")
        dq = await post("queued-before-delete", d["thread_id"])
        await wait_status(r, d["task_id"], {"interrupted"})
        res = await c.delete(f"/api/v1/chat/threads/{d['thread_id']}")
        check(res.status_code == 200 and await r.hget(tq.task_key(dq["task_id"]), "status") == "cancelled"
              and not await r.exists(tq.thread_waiting_key(d["thread_id"]), tq.thread_hold_key(d["thread_id"])),
              "delete thread cancels queued tasks and clears queue state")

        # ---- 11b. 訪客：不建對話、ephemeral 交給 worker、同一訪客依序執行、別人看不到 ----
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://t", timeout=60) as g:
            g1 = (await g.post("/api/v1/chat/tasks", json={"message": "guest-1"})).json()
            g2 = (await g.post("/api/v1/chat/tasks", json={"message": "guest-2"})).json()
            check(g1["thread_id"] is None and g1["ephemeral"] and g2["status"] == "queued",
                  "guest: no thread created, second message queued behind the first")
            await wait_status(r, g2["task_id"], {"completed"})
            check(ephemeral_seen.get("guest-1") is True and ephemeral_seen.get("guest-2") is True
                  and ephemeral_seen.get("m1") is False, "worker runs guest messages as ephemeral, users' not")
            evs = parse_sse((await g.get(f"/api/v1/chat/tasks/{g1['task_id']}/events")).text)
            check(tokens(evs) == "guest-1-0guest-1-1guest-1-2", "guest streams own output over SSE")
            check((await c.get(f"/api/v1/chat/tasks/{g1['task_id']}/events")).status_code == 404,
                  "another identity cannot stream the guest's task")
            check((await g.get("/api/v1/chat/threads")).json() == {"threads": []}, "guest has no thread list")
            async with db.get_pool().connection() as conn:
                cur = await conn.execute("SELECT count(*) AS n FROM threads WHERE thread_id LIKE 'guest%%'")
                check((await cur.fetchone())["n"] == 0, "guest left no row in threads")

            # ---- 11c. 冪等：同一次發送重送，只有一個任務 ----
            def new_key():
                return uuid.uuid4().hex

            async def send(client, msg, key, tid=None):
                return await client.post("/api/v1/chat/tasks", json={"message": msg, "thread_id": tid},
                                         headers={"Idempotency-Key": key})

            async def thread_count():
                async with db.get_pool().connection() as conn:
                    cur = await conn.execute("SELECT count(*) AS n FROM threads")
                    return (await cur.fetchone())["n"]

            await asyncio.sleep(0.5)
            order.clear()
            threads_before = await thread_count()
            key = new_key()
            first = await send(c, "idem-1", key)
            created_threads.add(first.json()["thread_id"])
            again = await send(c, "idem-1", key)
            check(first.status_code == again.status_code == 202 and "replayed" not in first.json()
                  and again.json()["replayed"] is True, "same key: second request is flagged as a replay")
            check(again.json()["task_id"] == first.json()["task_id"]
                  and again.json()["thread_id"] == first.json()["thread_id"], "same key: same task and thread")
            check(await thread_count() == threads_before + 1, "same key on a new chat: only one thread created")
            await wait_status(r, first.json()["task_id"], {"completed"})
            check(order.count("idem-1") == 1, f"same key: agent ran once ({order})")

            late = await send(c, "idem-1", key)
            check(late.status_code == 202 and late.json()["task_id"] == first.json()["task_id"]
                  and late.json()["status"] == "pending", "retry after the task finished: original task, not a new one")
            check(tokens(await events(late.json()["task_id"])) == "idem-1-0idem-1-1idem-1-2",
                  "retry after the task finished: the reply replays over SSE")

            tid_i = first.json()["thread_id"]
            repeat = await send(c, "idem-1", new_key(), tid_i)
            check(repeat.status_code == 202 and repeat.json()["task_id"] != first.json()["task_id"],
                  "same text with a new key: a new task (user really sent it again)")
            await wait_status(r, repeat.json()["task_id"], {"completed"})
            check(order.count("idem-1") == 2, f"same text with a new key: agent ran again ({order})")

            running = await post("idem-running", tid_i)
            q_key = new_key()
            queued = await send(c, "idem-queued", q_key, tid_i)
            q_again = await send(c, "idem-queued", q_key, tid_i)
            check(queued.json()["status"] == "queued" and q_again.json()["status"] == "queued"
                  and q_again.json()["position"] == queued.json()["position"]
                  and q_again.json()["task_id"] == queued.json()["task_id"], "retry while queued: same position")
            await wait_status(r, queued.json()["task_id"], {"completed"})
            await wait_status(r, running["task_id"], {"completed"})
            check(order.count("idem-queued") == 1, "retry while queued: agent ran once")

            before_race = await thread_count()
            race_key = new_key()
            race = await asyncio.gather(*[send(c, "idem-race", race_key) for _ in range(5)])
            ok = [x.json() for x in race if x.status_code == 202]
            check(len({x["task_id"] for x in ok}) == 1 and all(x.status_code in (202, 409) for x in race),
                  f"5 requests at once: one task ({[x.status_code for x in race]})")
            created_threads.add(ok[0]["thread_id"])
            check(await thread_count() == before_race + 1, "5 requests at once: one thread")
            await wait_status(r, ok[0]["task_id"], {"completed"})
            check(order.count("idem-race") == 1, "5 requests at once: agent ran once")

            res = await send(c, "different text", race_key)
            check(res.status_code == 422, f"same key with different content -> 422 ({res.status_code})")
            res = await send(c, "x", "too-short")
            check(res.status_code == 422, f"malformed key -> 422 ({res.status_code})")

            shared_key = new_key()
            mine = await send(c, "idem-mine", shared_key)
            created_threads.add(mine.json()["thread_id"])
            theirs = await send(g, "idem-theirs", shared_key)
            check(theirs.status_code == 202 and theirs.json()["task_id"] != mine.json()["task_id"]
                  and "replayed" not in theirs.json(), "same key from another user: their own task, not mine")
            g_again = await send(g, "idem-theirs", shared_key)
            check(g_again.json()["task_id"] == theirs.json()["task_id"] and g_again.json()["thread_id"] is None
                  and g_again.json()["ephemeral"] is True, "guest retry: same task, still ephemeral")
            await wait_status(r, mine.json()["task_id"], {"completed"})
            await wait_status(r, theirs.json()["task_id"], {"completed"})

            # 建立任務失敗：key 要放掉、空對話要清掉，重送才會真的建立任務
            real_submit = tq.submit_task

            async def broken_submit(*a, **k):
                raise ConnectionError("redis went away")

            before_fail = await thread_count()
            fail_key = new_key()
            tq.submit_task = broken_submit
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                    base_url="https://t", timeout=60, cookies=c.cookies,
                ) as c2:
                    res = await send(c2, "idem-fail", fail_key)
            finally:
                tq.submit_task = real_submit
            check(res.status_code == 500 and await thread_count() == before_fail,
                  f"submit failed: no empty thread left behind ({res.status_code})")
            retry = await send(c, "idem-fail", fail_key)
            check(retry.status_code == 202 and "replayed" not in retry.json(), "submit failed: retry creates the task")
            created_threads.add(retry.json()["thread_id"])
            await wait_status(r, retry.json()["task_id"], {"completed"})
            check(order.count("idem-fail") == 1, "submit failed: agent ran once, on the retry")

            # 保留期：建立完成才延長；API 當在中間時 key 很快消失，不會一直擋住重送
            me_id = str((await c.get("/api/v1/auth/me")).json()["user"]["id"])
            ttl = await r.ttl(tq.idempotency_key(me_id, fail_key))
            check(tq.IDEMPOTENCY_CLAIM_TTL < ttl <= tq.IDEMPOTENCY_TTL <= tq.EVENTS_TTL_AFTER_FINISH,
                  f"confirmed key lives as long as the events do ({ttl}s)")
            crash_key = new_key()
            check(await tq.claim_idempotency(r, me_id, crash_key, {
                "task_id": tq.new_task_id(), "thread_id": None,
                "fingerprint": hashlib.sha256(
                    json.dumps(["idem-crash", None, None], ensure_ascii=False).encode()).hexdigest(),
            }) is None, "claim: first caller wins")
            res = await send(c, "idem-crash", crash_key)
            check(res.status_code == 409, f"claimed but no task yet -> 409, not a bogus task ({res.status_code})")
            check(0 < await r.ttl(tq.idempotency_key(me_id, crash_key)) <= tq.IDEMPOTENCY_CLAIM_TTL,
                  "claimed but never confirmed: expires quickly")
            await r.delete(tq.idempotency_key(me_id, crash_key))
            res = await send(c, "idem-crash", crash_key)
            check(res.status_code == 202 and "replayed" not in res.json(), "after the claim expires: retry creates the task")
            created_threads.add(res.json()["thread_id"])
            await wait_status(r, res.json()["task_id"], {"completed"})

    # ---- 12. 崩潰回收：某 worker 拉走任務後死掉，回收後排隊的下一則照常開始 ----
    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)
    r = tq.get_redis()
    dead = tq.new_task_id()
    await tq.submit_task(r, task_id=dead, kind="chat", thread_id="thread_dead", user_id=None, payload={"message": "x"})
    follow = tq.new_task_id()
    await tq.submit_task(r, task_id=follow, kind="chat", thread_id="thread_dead", user_id=None, payload={"message": "follow"})
    await r.xreadgroup(tq.QUEUE_GROUP, "dead-consumer", {tq.QUEUE_STREAM: ">"}, count=1)
    await tq.mark_processing(r, dead, "dead-consumer", None)
    # 須大於假任務執行時間，否則會搶走執行中的任務。1.5 秒在機器忙的時候（例如剛重建完容器）
    # 不夠，排在後面的任務還在跑就被當成孤兒回收；3 秒留足餘裕
    worker.CLAIM_IDLE_MS = 3000
    worker.RECLAIM_INTERVAL = 0.3
    worker_task = asyncio.create_task(worker.main())
    check(await wait_status(r, dead, {"failed"}, timeout=12) == "failed", "orphaned task reclaimed -> failed")
    check(await wait_status(r, follow, {"completed"}, timeout=5) == "completed", "orphan's queued follow-up still ran")
    # 狀態標成 completed 之後 worker 才 XACK，兩者之間有一小段空檔：等一下再判定
    deadline = time.monotonic() + 3
    while (pending := await r.xpending(tq.QUEUE_STREAM, tq.QUEUE_GROUP))["pending"] and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    check(pending["pending"] == 0, "no pending entries left in queue")

    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)
    # 測試在真 Postgres 的 threads 表留下的訪客對話（agent 是假的，checkpointer 沒有資料）
    async with db.get_pool().connection() as conn:
        await conn.execute("DELETE FROM threads WHERE thread_id = ANY(%s)", (list(created_threads),))
        await conn.execute("DELETE FROM users WHERE email = %s", (email,))
    await history.close_history()
    await db.close_pool()
    print("ALL PASSED")


asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
