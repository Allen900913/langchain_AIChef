"""Agent worker：從 Redis 佇列主動拉任務執行，把輸出發布到該任務的事件串流。

啟動：python -m app.worker
可水平擴充（多台 / docker compose --scale worker=N），每台用 WORKER_CONCURRENCY
限制同時處理的任務數——只在有空位時才去拉下一個任務（pull-based），佇列負責削峰填谷，
worker 不會被瞬間湧入的請求壓垮。

停止（cancel）：API 把停止意圖寫進任務（cancel_requested_at，持久化），再用 Pub/Sub 通知。
worker 收到通知（或輪詢發現旗標，補 Pub/Sub 可能漏送的情況）後取消該任務的 asyncio task，
整理 checkpointer、標記 cancelled、發布「已停止」終止事件，並推進同一對話排隊中的下一則訊息。
被回收器強制收尾的任務（停止逾時 / 判定崩潰），鎖已交給下一則：原 worker 恢復後由巡查發現、
停掉 agent，且不再整理 checkpointer——那時最新的 checkpoint 已是下一則的進度。
asyncio 的取消會在下一個 await 點生效，所以模型串流、DB/Redis 呼叫都會立刻中斷；
但同步工具跑在執行緒池裡，執行緒無法被中斷——它會在背景跑完，只是結果被丟棄
（已發生的副作用無法收回）。worker 若卡死沒有回應，回收器在 CANCEL_DEADLINE 後強制收尾。

訪客任務（payload.ephemeral）：交給沒有 checkpointer / store 的訪客 agent，不讀寫 checkpointer、
不記歷史界線、停止後也不整理對話——輸出只經由事件串流送出，跑完不留任何紀錄。

崩潰回收：任務被拉走後若 worker 掛掉，該 entry 會停在 consumer group 的 pending 清單。
回收器定期用 XAUTOCLAIM 撿回閒置過久的 entry，把任務標記為失敗並通知前端。
不自動重跑：agent 跑到一半可能已改過冰箱/記憶等長期資料，重跑會重複套用副作用。
"""

import asyncio
import json
import os
import socket
import sys
import time
from contextlib import aclosing

from dotenv import load_dotenv

load_dotenv()

# 必須在 import agent 之前：app.agents.app 會在 import 階段處理 stdout 編碼並讀取環境變數
from app.agents.app import (  # noqa: E402
    MAX_TIMEOUT,
    call_agent,
    close_agent_infra,
    get_last_message_id,
    init_agent_infra,
    resume_agent,
    save_cancelled_turn,
)
from app.common import task_queue as tq  # noqa: E402

# 每台 worker 同時處理幾個任務。agent 主要時間花在等遠端 LLM / E2B 回應（I/O bound），
# 這個上限實際保護的是記憶體與上游 API 的 RPM/TPM，而非 CPU。
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "4"))

# pending entry 閒置超過這麼久才視為 worker 已崩潰。單一任務最長約
# MAX_TIMEOUT（agent）+ 圖片辨識/輸入檢查的逾時，留足餘裕避免搶走仍在執行的任務。
CLAIM_IDLE_MS = (MAX_TIMEOUT + 180) * 1000
RECLAIM_INTERVAL = 60

# 輪詢停止旗標的間隔：Pub/Sub 是主要通知管道，輪詢只補它漏送（重連期間）的情況
CANCEL_POLL_INTERVAL = 2.0
# 要求停止後多久還沒收尾，就由回收器強制標記為已停止（worker 卡死 / 失聯時的保底）
CANCEL_DEADLINE = 60
REAP_INTERVAL = 10

CONSUMER = f"{socket.gethostname()}-{os.getpid()}"


class _Run:
    """一個執行中任務的狀態，供停止流程使用。"""

    def __init__(self, task: dict):
        self.task = task
        self.inner: asyncio.Task | None = None
        self.cancel_requested = False
        self.partial_text = ""  # 目前這段已串流出去的回答（工具呼叫開始時歸零，跟前端一致）
        self.saw_error = False
        self.terminal: tuple[str, dict] | None = None
        # 已被回收器強制收尾、thread 鎖已交給下一則：不能再寫這個 thread 的 checkpointer
        self.lost = False

    def request_cancel(self) -> None:
        if self.cancel_requested or self.inner is None or self.inner.done():
            return
        self.cancel_requested = True
        print(f"⏹️ [WORKER] 停止任務 {self.task['task_id']}")
        self.inner.cancel()

    def abandon(self) -> None:
        """任務還在跑，Redis 裡卻已是終態：被回收器強制收尾了，立刻停掉 agent。

        不管先前是否已要求停止都再取消一次——卡住的 agent 可能吞掉了第一次取消。
        """
        if self.inner is None or self.inner.done():
            return
        if not self.lost:
            print(f"🪦 [WORKER] 任務 {self.task['task_id']} 已被強制收尾，停止 agent")
        self.lost = True
        self.cancel_requested = True
        self.inner.cancel()


_running: dict[str, _Run] = {}


async def _consume(r, run: _Run, events) -> None:
    """把 agent 事件逐一發布到 Redis；遇到終止事件只記下來，交由 _run_task 收尾。

    記下終止事件後不能直接 return：done 是在 _stream_agent 的 finally 裡 yield 的，
    提早關閉串流會在那個 yield 丟 GeneratorExit，後面排程反思與評分的程式就不會執行。
    """
    task_id = run.task["task_id"]
    # aclosing：發布失敗或被取消跳出迴圈時立刻關閉 agent 串流，不讓它在背景繼續跑
    async with aclosing(events):
        async for event, data in events:
            if event in tq.TERMINAL_EVENTS:
                run.terminal = (event, data)
                continue
            if event == "error":
                run.saw_error = True
            elif event == "token":
                run.partial_text += data.get("delta", "")
            elif event == "tool_start":
                run.partial_text = ""
            await tq.publish_event(r, task_id, event, data)


async def _run_task(r, task_id: str) -> None:
    task = await tq.get_task(r, task_id)
    if not task or task.get("status") != tq.TaskStatus.PENDING:
        # 任務已過期、已被取消或已被處理過（例如回收器已標記失敗），直接確認掉
        print(f"⏭️ [WORKER] 略過任務 {task_id}（status={task.get('status')}）")
        return

    thread_id = task["thread_id"]
    payload = json.loads(task["payload"])
    ephemeral = bool(payload.get("ephemeral"))
    task["ephemeral"] = ephemeral
    # 歷史界線：開跑前 checkpointer 的最後一則訊息。此時本任務已持有 thread 鎖，沒有別人會改動它。
    # 訪客任務沒有 checkpoint，不需要界線。
    boundary_id = None if ephemeral else await get_last_message_id(thread_id)
    if not await tq.mark_processing(r, task_id, CONSUMER, boundary_id):
        print(f"⏭️ [WORKER] 任務 {task_id} 在開始前已被取消")
        return
    task["boundary_id"] = boundary_id or ""
    await tq.publish_event(r, task_id, "started", {})

    run = _Run(task)
    _running[task_id] = run
    try:
        user_id = task.get("user_id") or None
        print(f"🚚 [WORKER] 開始 {task['kind']} 任務 {task_id}（thread={thread_id}）")

        async def produce():
            if task["kind"] == "resume":
                events = resume_agent(thread_id, payload["decisions"], user_id)
            else:
                events = await call_agent(
                    payload["message"], payload.get("image_url"), thread_id, user_id, ephemeral=ephemeral
                )
            await _consume(r, run, events)

        run.inner = asyncio.create_task(produce())
        # 開跑前就已被要求停止（API 寫旗標時 worker 還沒登記這個任務）
        if await tq.is_cancel_requested(r, task_id):
            run.request_cancel()
        # asyncio.wait 不會把 inner 的取消/例外拋到這裡，方便區分「任務被停止」與「worker 自己被關閉」
        await asyncio.wait({run.inner})

        if run.inner.cancelled():
            await _finish_cancelled(r, run)
        elif run.inner.exception() is not None:
            raise run.inner.exception()
        else:
            await _finish_normally(r, run)
        print(f"🏁 [WORKER] 任務 {task_id} 結束")

    except asyncio.CancelledError:
        # worker 自己被關閉（優雅關機逾時）：連同 agent 一起停掉，任務標記失敗
        if run.inner and not run.inner.done():
            run.inner.cancel()
        if await tq.finish_task(r, task_id, tq.TaskStatus.FAILED, error="worker 關閉"):
            await tq.publish_event(r, task_id, "error", {"message": "伺服器重新啟動，這則訊息沒有處理完，請重新送出。"})
            await tq.publish_event(r, task_id, "done", {"status": "failed"}, ttl=tq.EVENTS_TTL_AFTER_FINISH)
        raise
    except Exception as exc:
        print(f"💥 [WORKER] 任務 {task_id} 失敗：{type(exc).__name__}: {exc}")
        if await tq.finish_task(r, task_id, tq.TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:500]):
            await tq.publish_event(r, task_id, "error", {"message": "服務暫時發生問題，請稍後再試。"})
            await tq.publish_event(r, task_id, "done", {"status": "failed"}, ttl=tq.EVENTS_TTL_AFTER_FINISH)
    finally:
        _running.pop(task_id, None)


async def _finish_normally(r, run: _Run) -> None:
    task_id = run.task["task_id"]
    event, data = run.terminal or ("done", {"status": "complete"})
    # 先寫終態、放鎖、推進下一個，再發布終止事件（理由見 finish_task docstring）
    if event == "interrupt":
        ok = await tq.finish_task(r, task_id, tq.TaskStatus.INTERRUPTED)
    else:
        ok = await tq.finish_task(r, task_id, tq.TaskStatus.FAILED if run.saw_error else tq.TaskStatus.COMPLETED)
    if ok:
        await tq.publish_event(r, task_id, event, data, ttl=tq.EVENTS_TTL_AFTER_FINISH)


async def _finish_cancelled(r, run: _Run) -> None:
    task = run.task
    # save_cancelled_turn 讀寫的是 thread「最新」的 checkpoint，只有仍持有 thread 鎖時那才是自己的。
    # 已被強制收尾時鎖在下一則手上，整理會把「已停止」接到別人的對話後面、替別人執行中的工具
    # 補上「結果未知」，所以跳過；懸空的工具呼叫下一輪由 PatchDanglingToolCalls 補上。
    # 檢查與寫入無法同一個原子操作，仍有毫秒級空窗（要卡超過 CANCEL_DEADLINE 又剛好這時恢復）。
    if task.get("ephemeral"):
        pass  # 訪客任務沒有 checkpoint，沒有對話要整理
    elif run.lost or not await tq.owns_thread(r, task["task_id"], task["thread_id"]):
        print(f"⚠️ [WORKER] 任務 {task['task_id']} 已不持有 thread 鎖，跳過整理對話")
    else:
        try:
            # 必須在放鎖之前：下一則排隊訊息開跑時，checkpointer 要已經是可以接續的狀態
            await save_cancelled_turn(task["thread_id"], task["kind"], task.get("boundary_id") or None, run.partial_text)
        except Exception as exc:
            # 整理失敗不影響停止本身：殘留的懸空工具呼叫下一輪會由 PatchDanglingToolCalls 補上
            print(f"⚠️ [WORKER] 停止後整理對話失敗：{type(exc).__name__}: {exc}")
    if await tq.finish_task(r, task["task_id"], tq.TaskStatus.CANCELLED):
        await tq.publish_cancelled(r, task["task_id"])


async def _watch_cancel(r) -> None:
    """接收停止通知：Pub/Sub 即時通知為主，定期巡查兜底（見 _poll_running）。"""
    pubsub = None
    last_poll = 0.0
    while True:
        try:
            if pubsub is None:
                pubsub = r.pubsub(ignore_subscribe_messages=True)
                await pubsub.subscribe(tq.CANCEL_CHANNEL)
                # 剛（重新）訂閱：補查一次訂閱前可能錯過的要求
                await _poll_running(r)
                last_poll = time.monotonic()
            msg = await pubsub.get_message(timeout=CANCEL_POLL_INTERVAL)
            if msg:
                run = _running.get(msg["data"])
                if run:
                    run.request_cancel()
            # 依時間巡查，而非「沒收到通知才巡查」：停止通知接連不斷時巡查也不會被餓死
            if time.monotonic() - last_poll >= CANCEL_POLL_INTERVAL:
                await _poll_running(r)
                last_poll = time.monotonic()
        except asyncio.CancelledError:
            if pubsub is not None:
                await pubsub.aclose()
            raise
        except Exception as exc:
            print(f"⚠️ [WORKER] 停止通知連線中斷，重新訂閱：{type(exc).__name__}: {exc}")
            if pubsub is not None:
                try:
                    await pubsub.aclose()
                except Exception:
                    pass
            pubsub = None
            await asyncio.sleep(1)


async def _poll_running(r) -> None:
    """巡查本機執行中的任務：補 Pub/Sub 漏送的停止要求，並找出已被回收器強制收尾的任務。"""
    for task_id, run in list(_running.items()):
        status, cancel_requested = await tq.poll_task(r, task_id)
        if status != tq.TaskStatus.PROCESSING:
            # 還在跑，Redis 卻已是終態（或已過期）：停止逾時 / 判定崩潰，鎖已交給下一則
            run.abandon()
        elif cancel_requested:
            run.request_cancel()


async def _handle_entry(r, entry_id: str, fields: dict, slots: asyncio.Semaphore) -> None:
    try:
        await _run_task(r, fields.get("task_id", ""))
        await tq.ack_entry(r, entry_id)
    except Exception as exc:
        # 多半是 Redis 斷線：entry 沒 ack 會留在 pending，之後由回收器處理
        print(f"⚠️ [WORKER] 處理佇列 entry {entry_id} 失敗：{type(exc).__name__}: {exc}")
    finally:
        slots.release()


async def _fail_orphan(r, entry_id: str, fields: dict) -> None:
    """回收崩潰 worker 留下的任務：標記失敗、推進排隊中的下一則、通知前端、確認掉佇列 entry。"""
    task_id = fields.get("task_id", "")
    task = await tq.get_task(r, task_id)
    if task and task.get("status") in tq.RUNNING_STATUSES:
        print(f"♻️ [WORKER] 回收孤兒任務 {task_id}（原 worker={task.get('worker', '?')}）")
        if await tq.finish_task(r, task_id, tq.TaskStatus.FAILED, error="worker 中斷，任務未完成"):
            await tq.publish_event(r, task_id, "error", {"message": "處理此任務的伺服器中斷了，請重新送出訊息。"})
            await tq.publish_event(r, task_id, "done", {"status": "failed"}, ttl=tq.EVENTS_TTL_AFTER_FINISH)
    await tq.ack_entry(r, entry_id)


async def _reclaim_loop(r) -> None:
    while True:
        await asyncio.sleep(RECLAIM_INTERVAL)
        try:
            start = "0-0"
            while True:
                start, claimed, _deleted = await r.xautoclaim(
                    tq.QUEUE_STREAM, tq.QUEUE_GROUP, CONSUMER,
                    min_idle_time=CLAIM_IDLE_MS, start_id=start, count=20,
                )
                for entry_id, fields in claimed:
                    await _fail_orphan(r, entry_id, fields or {})
                if start == "0-0":
                    break
        except Exception as exc:
            print(f"⚠️ [WORKER] 回收孤兒任務失敗：{type(exc).__name__}: {exc}")


async def _reap_overdue_cancellations(r) -> None:
    """要求停止超過 CANCEL_DEADLINE 仍未收尾的任務，強制標記為已停止。

    正常情況 worker 幾秒內就會停下；會走到這裡代表負責的 worker 卡死或失聯。
    不砍整個 worker 行程——它同時在跑別人的任務。只把這個任務收尾、放鎖、推進下一則，
    讓使用者不會一直停在「停止中」；finish_task 的 CAS 保證卡住的 worker 之後醒來也覆寫不了終態，
    _poll_running 會發現它並停掉 agent，_finish_cancelled 也不會再去整理 checkpointer。
    """
    while True:
        await asyncio.sleep(REAP_INTERVAL)
        try:
            for task_id in await tq.overdue_cancellations(r, CANCEL_DEADLINE):
                if await tq.finish_task(r, task_id, tq.TaskStatus.CANCELLED, error="停止逾時，強制結束"):
                    print(f"⏱️ [WORKER] 任務 {task_id} 停止逾時，強制標記為已停止")
                    await tq.publish_cancelled(r, task_id)
                else:
                    await r.zrem(tq.CANCELLING_ZSET, task_id)
        except Exception as exc:
            print(f"⚠️ [WORKER] 檢查停止逾時失敗：{type(exc).__name__}: {exc}")


async def main() -> None:
    await init_agent_infra()
    r = tq.get_redis()
    await tq.ensure_group(r)

    slots = asyncio.Semaphore(WORKER_CONCURRENCY)
    running: set[asyncio.Task] = set()
    background = [
        asyncio.create_task(_reclaim_loop(r)),
        asyncio.create_task(_watch_cancel(r)),
        asyncio.create_task(_reap_overdue_cancellations(r)),
    ]
    print(f"👷 [WORKER] {CONSUMER} 就緒，同時處理上限 {WORKER_CONCURRENCY} 個任務")

    try:
        while True:
            # 有空位才拉：一次只拿一個，拿到就佔一個位子
            await slots.acquire()
            try:
                resp = await r.xreadgroup(
                    tq.QUEUE_GROUP, CONSUMER, {tq.QUEUE_STREAM: ">"}, count=1, block=5000,
                )
            except Exception as exc:
                slots.release()
                print(f"⚠️ [WORKER] 讀取佇列失敗，稍後重試：{type(exc).__name__}: {exc}")
                await asyncio.sleep(2)
                continue
            if not resp:
                slots.release()
                continue
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    t = asyncio.create_task(_handle_entry(r, entry_id, fields, slots))
                    running.add(t)
                    t.add_done_callback(running.discard)
    finally:
        # 優雅關機：不再拉新任務，等手上的任務跑完（docker stop_grace_period 需大於此等待）。
        # 停止通知仍要繼續監聽，關機期間使用者按停止才有反應。
        background[0].cancel()
        background[2].cancel()
        if running:
            print(f"🛑 [WORKER] 停止接單，等待 {len(running)} 個進行中任務完成…")
            await asyncio.wait(running, timeout=MAX_TIMEOUT + 30)
        background[1].cancel()
        await asyncio.gather(*background, return_exceptions=True)
        await tq.close_redis()
        await close_agent_infra()


def _run() -> None:
    if sys.platform == "win32":
        # psycopg 的 async 模式不支援 Windows 預設的 ProactorEventLoop（同 main.py 的說明）
        asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
    else:
        import signal

        async def _main_with_sigterm() -> None:
            # docker stop 送 SIGTERM：轉成取消主迴圈，走 finally 的優雅關機
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            try:
                await main()
            except asyncio.CancelledError:
                print("🛑 [WORKER] 已關機")

        asyncio.run(_main_with_sigterm())


if __name__ == "__main__":
    try:
        _run()
    except KeyboardInterrupt:
        pass
