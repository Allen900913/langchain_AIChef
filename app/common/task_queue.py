"""非同步任務佇列 + 事件串流（皆以 Redis Streams 實作），API 伺服器與 agent worker 共用。

只依賴 redis，不 import agent——API 伺服器必須保持輕量，不能被模型/向量庫等重依賴拖垮。

資料佈局：
- chef:queue                         全域任務佇列（Stream + consumer group），entry 只帶 task_id。
                                      worker 依自身容量用 XREADGROUP 主動拉取（pull-based 削峰填谷），
                                      做完 XACK；worker 崩潰留下的 pending entry 由 XAUTOCLAIM 回收。
- chef:task:{task_id}                任務中繼資料（Hash）：status / kind / thread_id / payload / boundary_id …
- chef:task:{task_id}:events         該任務的輸出事件（Stream）。每個事件有遞增 ID，直接當 SSE 的 id:，
                                      前端斷線/重新整理後帶 Last-Event-ID 即可從斷點精確續傳。
                                      這是不用 Pub/Sub 的原因：Pub/Sub 不留存，訂閱前或斷線期間
                                      的輸出會直接遺失，而 checkpointer 只在節點結束時落地，補不回半段回答。
- chef:thread:{thread_id}:active     thread 鎖（值 = 正在執行的 task_id）。同一 thread 同時只能有一個
                                      任務在跑，否則兩個 worker 會同時寫同一份 checkpointer 互相覆蓋。
- chef:thread:{thread_id}:waiting    該 thread 的等待清單（List）。同一對話連發訊息時採 enqueue 策略：
                                      後到的訊息在這裡排隊，前一個任務結束時由收尾腳本推進下一個。
- chef:thread:{thread_id}:hold       前一個任務停在 HITL 審核時設定：等待清單暫停推進，直到使用者
                                      審核完（resume 跑完）才繼續——不讓排隊訊息替使用者默默略過審核。
- chef:idem:{user_id}:{key}          冪等紀錄（String，JSON）：同一次發送重送時，回傳原本的任務而不建立新任務。
- chef:cancel                        Pub/Sub 頻道：通知 worker 某個任務被要求停止（即時性用；
                                      Pub/Sub 不保證送達，worker 另以輪詢 cancel_requested 兜底）。
- chef:cancelling                    ZSET（task_id → 要求停止的時間）：超過寬限期仍未停下的任務，
                                      由回收器強制收尾，避免 worker 卡死時前端永遠停在「停止中」。

狀態機：
  queued ──(前一個結束)──→ pending ──(worker 拉到)──→ processing ──→ completed / interrupted / failed / cancelled
    │                         │                          │
    └──── cancel ─────────────┴──── cancel ──→ cancelled  └── cancel ──→（worker 中止後）cancelled

所有跨 key 的狀態轉換（送出/推進/取消/收尾）都用 Lua 腳本原子執行：例如「取消排隊中任務」與
「前一個任務結束時把它推進全域佇列」同時發生時，兩者只會有一個成功。
腳本內會以 task_id 動態組出 key，只適用單機 Redis（不支援 Redis Cluster）。
"""

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from enum import StrEnum

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6389/0")
# 單次 Redis 指令的讀取逾時（秒），須大於所有 BLOCK 等待時間
READ_TIMEOUT = 30

QUEUE_STREAM = "chef:queue"
QUEUE_GROUP = "chef-workers"
CANCEL_CHANNEL = "chef:cancel"
CANCELLING_ZSET = "chef:cancelling"

# 任務中繼資料保留多久（秒）。只用於「執行中/排隊中任務的接續」；
# 對話內容與待審核的 HITL 都以 checkpointer 為準，過期不影響歷史。
TASK_TTL = 24 * 3600
# 任務結束後事件串流保留多久：只為了讓結束前一刻斷線的前端還能重連補完；
# 已完成的任務重新載入時直接讀 checkpointer，不需要事件串流。
EVENTS_TTL_AFTER_FINISH = 3600
# thread 鎖的租約（只有正在執行的任務持鎖，排隊中不持鎖）。worker 崩潰時
# 由回收器提前釋放，這個 TTL 只是所有 worker 都掛掉時的最後保險。
THREAD_LOCK_TTL = 3600
# 冪等 key 保留多久（秒）：重送時要靠事件串流重播原本的回答，所以不能比事件串流活得久；
# 過期後同一個 key 就當成新訊息。
IDEMPOTENCY_TTL = EVENTS_TTL_AFTER_FINISH
# 任務建立完成前的保留期：API 在這段期間當掉時 key 會自己消失，之後的重送不會一直被擋
IDEMPOTENCY_CLAIM_TTL = 60

# 讀到這些事件代表任務的串流已結束（done = 正常結束/失敗/停止；interrupt = 停在 HITL 等待審核）
TERMINAL_EVENTS = frozenset({"done", "interrupt"})


class TaskStatus(StrEnum):
    QUEUED = "queued"            # 在 thread 等待清單，等前一個任務結束
    PENDING = "pending"          # 已持有 thread 鎖、在全域佇列等 worker
    PROCESSING = "processing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"  # 停在 HITL 等待審核
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_STATUSES = frozenset({TaskStatus.QUEUED, TaskStatus.PENDING, TaskStatus.PROCESSING})
# 「正在佔用 thread」的狀態（排隊中的不算）
RUNNING_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.PROCESSING})


class ThreadBusyError(Exception):
    """該 thread 已有正在執行的任務（只有 resume 會遇到：審核必須在沒有任務執行時送出）。"""

    def __init__(self, task_id: str | None):
        super().__init__(f"thread 已有進行中的任務：{task_id}")
        self.task_id = task_id


def task_key(task_id: str) -> str:
    return f"chef:task:{task_id}"


def events_key(task_id: str) -> str:
    return f"chef:task:{task_id}:events"


def thread_lock_key(thread_id: str) -> str:
    return f"chef:thread:{thread_id}:active"


def thread_waiting_key(thread_id: str) -> str:
    return f"chef:thread:{thread_id}:waiting"


def thread_hold_key(thread_id: str) -> str:
    return f"chef:thread:{thread_id}:hold"


_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """延遲建立 Redis client 單例（須在事件迴圈內首次呼叫）。"""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            # redis-py 8 預設 socket_timeout=5 秒，會把 XREAD/XREADGROUP 的 BLOCK 等待
            # （API 15 秒、worker 5 秒）當成讀取逾時直接拋 TimeoutError。必須大於最長 BLOCK。
            socket_timeout=READ_TIMEOUT,
        )
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _now() -> str:
    return f"{time.time():.3f}"


# ==============================================================================
# Lua 腳本：所有跨 key 的狀態轉換
# ==============================================================================

# 共用函式。advance(thread_id, finishing_id, hold)：任務結束時呼叫，
#   - 鎖已被別的任務持有 → 不動（那個任務結束時自己會推進）
#   - hold=1（停在 HITL）→ 記下 hold、放鎖，等待清單暫停
#   - 否則從等待清單取下一個仍為 queued 的任務：轉移鎖給它並推進全域佇列；清單空了就放鎖
_LUA_LIB = """
local function tkey(id) return 'chef:task:' .. id end

local function start(thread_id, id, lock_ttl, now)
  redis.call('SET', 'chef:thread:' .. thread_id .. ':active', id, 'EX', lock_ttl)
  redis.call('HSET', tkey(id), 'status', 'pending', 'updated_at', now)
  redis.call('XADD', 'chef:queue', '*', 'task_id', id)
end

local function advance(thread_id, finishing_id, hold, lock_ttl, task_ttl, now)
  local lock = 'chef:thread:' .. thread_id .. ':active'
  local cur = redis.call('GET', lock)
  if cur and cur ~= finishing_id then return false end
  if hold == '1' then
    redis.call('SET', 'chef:thread:' .. thread_id .. ':hold', finishing_id, 'EX', task_ttl)
    redis.call('DEL', lock)
    return false
  end
  local waiting = 'chef:thread:' .. thread_id .. ':waiting'
  while true do
    local nxt = redis.call('LPOP', waiting)
    if not nxt then
      redis.call('DEL', lock)
      return false
    end
    -- 過期（hash 已消失）或已被取消的就跳過
    if redis.call('HGET', tkey(nxt), 'status') == 'queued' then
      start(thread_id, nxt, lock_ttl, now)
      return nxt
    end
  end
end
"""

# 送出任務。KEYS: -；ARGV: thread_id, task_id, kind, lock_ttl, task_ttl, now
# 回傳 {'started'} / {'queued', held(0/1), position} / {'busy', 持鎖的 task_id}
_SUBMIT_LUA = _LUA_LIB + """
local thread_id, id, kind = ARGV[1], ARGV[2], ARGV[3]
local lock_ttl, task_ttl, now = ARGV[4], ARGV[5], ARGV[6]
local lock = 'chef:thread:' .. thread_id .. ':active'
local waiting = 'chef:thread:' .. thread_id .. ':waiting'
local hold = 'chef:thread:' .. thread_id .. ':hold'
local locked = redis.call('EXISTS', lock) == 1

if kind == 'resume' then
  -- 審核插隊：要先於排隊中的訊息執行，但不能跟執行中的任務並行
  if locked then return {'busy', redis.call('GET', lock)} end
  redis.call('DEL', hold)
  start(thread_id, id, lock_ttl, now)
  return {'started'}
end

if not locked and redis.call('LLEN', waiting) == 0 then
  -- 沒有任務在跑也沒人排隊：直接開始（若停在 HITL，等於略過審核，由 agent 的 middleware 收拾）
  redis.call('DEL', hold)
  start(thread_id, id, lock_ttl, now)
  return {'started'}
end

redis.call('RPUSH', waiting, id)
redis.call('EXPIRE', waiting, task_ttl)
redis.call('HSET', tkey(id), 'status', 'queued', 'updated_at', now)
local held = 0
if not locked then
  if redis.call('EXISTS', hold) == 1 then
    held = 1
  else
    -- 沒人持鎖、沒有 hold 卻有人在排隊：異常殘留（例如所有 worker 掛掉、鎖過期），自己推進
    advance(thread_id, '', '0', lock_ttl, task_ttl, now)
  end
end
return {'queued', tostring(held), tostring(redis.call('LLEN', waiting))}
"""

# worker 開始處理：pending → processing（CAS，已被取消的就不跑）。ARGV: task_id, worker, boundary_id, lock_ttl, now
_MARK_PROCESSING_LUA = """
local k = 'chef:task:' .. ARGV[1]
if redis.call('HGET', k, 'status') ~= 'pending' then return 0 end
redis.call('HSET', k, 'status', 'processing', 'worker', ARGV[2], 'boundary_id', ARGV[3],
           'started_at', ARGV[5], 'updated_at', ARGV[5])
local thread_id = redis.call('HGET', k, 'thread_id')
redis.call('EXPIRE', 'chef:thread:' .. thread_id .. ':active', ARGV[4])
return 1
"""

# 收尾：pending/processing → 終態（CAS），並推進該 thread 的等待清單。
# ARGV: task_id, status, hold, lock_ttl, task_ttl, now, [field, value, ...]
# 回傳 {寫入成功 0/1, 被推進的下一個 task_id 或 ''}
_FINISH_LUA = _LUA_LIB + """
local id = ARGV[1]
local k = tkey(id)
local st = redis.call('HGET', k, 'status')
if st ~= 'processing' and st ~= 'pending' then return {0, ''} end
redis.call('HSET', k, 'status', ARGV[2], 'finished_at', ARGV[6], 'updated_at', ARGV[6])
for i = 7, #ARGV, 2 do redis.call('HSET', k, ARGV[i], ARGV[i + 1]) end
redis.call('EXPIRE', k, ARGV[5])
redis.call('ZREM', 'chef:cancelling', id)
local nxt = advance(redis.call('HGET', k, 'thread_id'), id, ARGV[3], ARGV[4], ARGV[5], ARGV[6])
return {1, nxt or ''}
"""

# 要求停止。ARGV: task_id, now, lock_ttl, task_ttl
# 回傳 {結果, 被推進的下一個 task_id 或 ''}；結果：
#   cancelled  = 還沒開始跑（queued / pending），已直接取消
#   cancelling = 執行中，已通知 worker
#   missing / 其他終態 = 不需要動作（冪等）
_CANCEL_LUA = _LUA_LIB + """
local id, now = ARGV[1], ARGV[2]
local k = tkey(id)
local st = redis.call('HGET', k, 'status')
if not st then return {'missing', ''} end
local thread_id = redis.call('HGET', k, 'thread_id')
if st == 'queued' then
  redis.call('LREM', 'chef:thread:' .. thread_id .. ':waiting', 0, id)
  redis.call('HSET', k, 'status', 'cancelled', 'finished_at', now, 'updated_at', now)
  return {'cancelled', ''}
elseif st == 'pending' then
  -- 已持鎖、在全域佇列等 worker：直接取消並把鎖交給下一個；worker 之後拉到會因 status 不是 pending 而略過
  redis.call('HSET', k, 'status', 'cancelled', 'finished_at', now, 'updated_at', now)
  local nxt = advance(thread_id, id, '0', ARGV[3], ARGV[4], now)
  return {'cancelled', nxt or ''}
elseif st == 'processing' then
  if not redis.call('HGET', k, 'cancel_requested_at') then
    redis.call('HSET', k, 'cancel_requested_at', now)
    redis.call('ZADD', 'chef:cancelling', now, id)
  end
  redis.call('PUBLISH', 'chef:cancel', id)
  return {'cancelling', ''}
end
return {st, ''}
"""


async def _eval(r: aioredis.Redis, script: str, *args) -> list:
    return await r.eval(script, 0, *[str(a) for a in args])


# ==============================================================================
# API 端：送任務 / 取消
# ==============================================================================

def new_task_id() -> str:
    return uuid.uuid4().hex


async def submit_task(
    r: aioredis.Redis,
    *,
    task_id: str,
    kind: str,
    thread_id: str,
    user_id: str | None,
    payload: dict,
) -> dict:
    """寫入任務中繼資料並依 thread 狀態決定直接開始或排隊。

    回傳 {"status": "pending" | "queued", "held": bool, "position": int}。
    kind="resume" 且 thread 有任務在跑時拋 ThreadBusyError。

    歷史界線（boundary_id）不在這裡決定，由 worker 開始處理時從 checkpointer 讀：
    排隊中的任務要等前一個任務跑完，送出當下的界線早已過時。
    """
    now = _now()
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(task_key(task_id), mapping={
            "task_id": task_id,
            "kind": kind,
            "thread_id": thread_id,
            "user_id": user_id or "",
            "payload": json.dumps(payload, ensure_ascii=False),
            "status": "submitting",
            "created_at": now,
            "updated_at": now,
        })
        pipe.expire(task_key(task_id), TASK_TTL)
        await pipe.execute()

    res = await _eval(r, _SUBMIT_LUA, thread_id, task_id, kind, THREAD_LOCK_TTL, TASK_TTL, now)
    if res[0] == "busy":
        await r.delete(task_key(task_id))
        raise ThreadBusyError(res[1])
    if res[0] == "started":
        return {"status": TaskStatus.PENDING, "held": False, "position": 0}
    return {"status": TaskStatus.QUEUED, "held": res[1] == "1", "position": int(res[2])}


def idempotency_key(user_id: str, key: str) -> str:
    return f"chef:idem:{user_id}:{key}"


async def claim_idempotency(r: aioredis.Redis, user_id: str, key: str, record: dict) -> dict | None:
    """搶下冪等 key（SET NX）。搶到回傳 None；已被用過回傳當時的紀錄（剛好過期回傳空 dict）。"""
    rkey = idempotency_key(user_id, key)
    if await r.set(rkey, json.dumps(record), nx=True, ex=IDEMPOTENCY_CLAIM_TTL):
        return None
    raw = await r.get(rkey)
    return json.loads(raw) if raw else {}


async def confirm_idempotency(r: aioredis.Redis, user_id: str, key: str) -> None:
    """任務已建立：把保留期延長到完整的 IDEMPOTENCY_TTL。"""
    await r.expire(idempotency_key(user_id, key), IDEMPOTENCY_TTL)


async def release_idempotency(r: aioredis.Redis, user_id: str, key: str) -> None:
    """任務建立失敗：放掉 key，讓重送能真的建立任務。"""
    await r.delete(idempotency_key(user_id, key))


async def replay_submission(r: aioredis.Redis, task_id: str) -> dict | None:
    """重送時用：回傳跟 submit_task 相同格式的結果；任務還在建立中（或不存在）回傳 None。

    前端只需要知道這則是不是排隊中。其他狀態一律回 pending：前端照常開事件串流，
    還在跑的接著看，已結束的從頭重播，最後都會收到終止事件。
    """
    task = await get_task(r, task_id)
    status = task.get("status")
    if status is None or status == "submitting":
        return None
    if status != TaskStatus.QUEUED:
        return {"status": TaskStatus.PENDING, "held": False, "position": 0}
    tasks = await get_thread_tasks(r, task["thread_id"])
    queued_ids = [t["task_id"] for t in tasks["queued"]]
    position = queued_ids.index(task_id) + 1 if task_id in queued_ids else 1
    return {"status": TaskStatus.QUEUED, "held": tasks["held"], "position": position}


async def cancel_task(r: aioredis.Redis, task_id: str) -> str:
    """要求停止任務，回傳 cancelled / cancelling / missing / 既有終態（冪等）。

    還沒開始跑的任務直接取消並補發終止事件；執行中的交給 worker 中止（見 worker._watch_cancel）。
    """
    res = await _eval(r, _CANCEL_LUA, task_id, _now(), THREAD_LOCK_TTL, TASK_TTL)
    if res[0] == "cancelled":
        await publish_cancelled(r, task_id)
    return res[0]


async def publish_cancelled(r: aioredis.Redis, task_id: str) -> None:
    await publish_event(r, task_id, "status", {
        "code": "CANCELLED", "level": "info", "message": "已停止回應",
    })
    await publish_event(r, task_id, "done", {"status": "cancelled"}, ttl=EVENTS_TTL_AFTER_FINISH)


async def get_task(r: aioredis.Redis, task_id: str) -> dict:
    """回傳任務中繼資料；不存在（或已過期）時回傳空 dict。"""
    return await r.hgetall(task_key(task_id))


async def get_thread_tasks(r: aioredis.Redis, thread_id: str) -> dict:
    """回傳 thread 目前的任務概況：{"active": 執行中任務 | None, "queued": [排隊任務...], "held": bool}。"""
    async with r.pipeline(transaction=True) as pipe:
        pipe.get(thread_lock_key(thread_id))
        pipe.lrange(thread_waiting_key(thread_id), 0, -1)
        pipe.exists(thread_hold_key(thread_id))
        active_id, waiting_ids, held = await pipe.execute()

    active = await get_task(r, active_id) if active_id else {}
    if active.get("status") not in RUNNING_STATUSES:
        active = {}
    queued = []
    for tid in waiting_ids:
        t = await get_task(r, tid)
        if t.get("status") == TaskStatus.QUEUED:
            queued.append(t)
    return {"active": active or None, "queued": queued, "held": bool(held) and not active}


async def is_thread_busy(r: aioredis.Redis, thread_id: str) -> bool:
    return bool(await r.exists(thread_lock_key(thread_id)))


async def clear_thread(r: aioredis.Redis, thread_id: str) -> None:
    """刪除對話時呼叫（呼叫前須確認沒有任務在跑）：取消所有排隊中的任務並清掉 thread 的佇列狀態。"""
    for tid in await r.lrange(thread_waiting_key(thread_id), 0, -1):
        await cancel_task(r, tid)
    await r.delete(thread_waiting_key(thread_id), thread_hold_key(thread_id))


# ==============================================================================
# API 端：讀事件串流
# ==============================================================================

async def read_events(
    r: aioredis.Redis,
    task_id: str,
    last_id: str = "0-0",
    block_ms: int = 15000,
) -> AsyncIterator[tuple[str, str, str] | None]:
    """從 last_id 之後持續讀取任務事件，yield (entry_id, event, data_json)。

    同一個 XREAD 游標先補完積壓的事件、再阻塞等新事件，兩段之間沒有接縫，不漏也不重複。
    每 block_ms 沒有新事件就 yield None，讓呼叫端送 SSE 心跳維持連線（排隊中的任務也是這樣等）。
    讀到終止事件（done / interrupt）即結束。若任務中繼資料已消失、或任務已是終態
    卻遲遲沒有終止事件（worker 在收尾途中崩潰），補一個 done 後結束，避免前端永遠掛著。
    """
    key = events_key(task_id)
    while True:
        resp = await r.xread({key: last_id}, count=200)
        if not resp:
            status = await r.hget(task_key(task_id), "status")
            if status is None or status not in ACTIVE_STATUSES:
                # 任務已結束：收尾都是先寫終態才發布終止事件，留一小段寬限等它到達，
                # 否則會把真正的 interrupt 誤換成合成的 done。等不到才補 done 收尾。
                resp = await r.xread({key: last_id}, count=200, block=2000)
                if not resp:
                    yield (last_id, "done", json.dumps({"status": status or "expired"}))
                    return
            else:
                resp = await r.xread({key: last_id}, count=200, block=block_ms)
                if not resp:
                    yield None
                    continue
        for _stream, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                event = fields.get("event", "token")
                yield (entry_id, event, fields.get("data", "{}"))
                if event in TERMINAL_EVENTS:
                    return


# ==============================================================================
# Worker 端：拉任務 / 發事件 / 收尾
# ==============================================================================

async def ensure_group(r: aioredis.Redis) -> None:
    """建立 consumer group（冪等）；MKSTREAM 讓佇列尚不存在時也能建立。"""
    try:
        await r.xgroup_create(QUEUE_STREAM, QUEUE_GROUP, id="0", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def publish_event(
    r: aioredis.Redis,
    task_id: str,
    event: str,
    data: dict | str,
    ttl: int = TASK_TTL,
) -> str:
    """寫入一筆事件並（重新）設定串流 TTL。

    TTL 跟著每次寫入設定，而非在開跑/收尾時另外設：串流 key 在第一筆 XADD 前並不存在，
    事先 EXPIRE 會落空。執行中用 TASK_TTL 兜底 worker 崩潰；終止事件傳 EVENTS_TTL_AFTER_FINISH。
    """
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
    async with r.pipeline(transaction=False) as pipe:
        pipe.xadd(events_key(task_id), {"event": event, "data": payload})
        pipe.expire(events_key(task_id), ttl)
        entry_id, _ = await pipe.execute()
    return entry_id


async def mark_processing(r: aioredis.Redis, task_id: str, worker: str, boundary_id: str | None) -> bool:
    """pending → processing。回傳 False 代表任務已不是 pending（排隊期間被取消），不該執行。"""
    return bool(await _eval(
        r, _MARK_PROCESSING_LUA, task_id, worker, boundary_id or "", THREAD_LOCK_TTL, _now(),
    ))


async def finish_task(
    r: aioredis.Redis,
    task_id: str,
    status: TaskStatus,
    **extra: str,
) -> bool:
    """寫入終態、釋放 thread 鎖並推進等待清單中的下一個任務（停在 HITL 時暫停推進）。

    必須在「發布終止事件之前」呼叫：前端收到 done 後會接上下一個任務的串流，
    或立刻送出下一則訊息，此時鎖與等待清單都必須已經是新狀態。

    回傳 False 代表任務已被別人收尾（例如停止逾時被回收器強制結束），呼叫端不該再發布終止事件。
    """
    args = [task_id, status, "1" if status == TaskStatus.INTERRUPTED else "0",
            THREAD_LOCK_TTL, TASK_TTL, _now()]
    for k, v in extra.items():
        args += [k, v]
    ok, _next = await _eval(r, _FINISH_LUA, *args)
    return bool(ok)


async def is_cancel_requested(r: aioredis.Redis, task_id: str) -> bool:
    return bool(await r.hexists(task_key(task_id), "cancel_requested_at"))


async def poll_task(r: aioredis.Redis, task_id: str) -> tuple[str | None, bool]:
    """worker 巡查執行中任務用：回傳 (status, 是否已要求停止)。"""
    status, requested = await r.hmget(task_key(task_id), "status", "cancel_requested_at")
    return status, requested is not None


async def owns_thread(r: aioredis.Redis, task_id: str, thread_id: str) -> bool:
    """任務是否仍是該 thread 的執行者：status 仍為 processing，且 thread 鎖仍在它手上。

    被回收器強制收尾（停止逾時 / 判定崩潰）後兩者都不成立：鎖已交給下一則，
    原 worker 就算之後恢復，也不能再寫該 thread 的 checkpointer。
    """
    async with r.pipeline(transaction=True) as pipe:
        pipe.hget(task_key(task_id), "status")
        pipe.get(thread_lock_key(thread_id))
        status, holder = await pipe.execute()
    return status == TaskStatus.PROCESSING and holder == task_id


async def overdue_cancellations(r: aioredis.Redis, older_than: float) -> list[str]:
    """要求停止已超過 older_than 秒、仍未收尾的任務。"""
    return await r.zrangebyscore(CANCELLING_ZSET, "-inf", time.time() - older_than)


async def ack_entry(r: aioredis.Redis, entry_id: str) -> None:
    """確認處理完畢並從佇列刪除（佇列 entry 不需留存，任務資料在 task hash）。"""
    await r.xack(QUEUE_STREAM, QUEUE_GROUP, entry_id)
    await r.xdel(QUEUE_STREAM, entry_id)
