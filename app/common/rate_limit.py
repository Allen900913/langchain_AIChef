"""送訊息的速率限制：保護模型額度（這不是 DDoS 防護，請求已經到達 API 才會被擋）。

訪客沒有帳號，cookie 清掉就重來，只能看 IP；登入使用者看 user_id，換網路或換裝置都是同一份額度。

檢查與計數放在同一段 Lua 腳本裡，同時到的請求不會一起通過。
被擋下的請求不計數，否則連按幾次就會把當天的額度也耗光。
"""

from dataclasses import dataclass

_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class Limit:
    count: int
    seconds: int


# 主模型一天約 200,000 tokens，每則訊息約 4,000–12,000，全站一天只夠 16–50 則，所以訪客要設得很小
GUEST_LIMITS = (Limit(3, 60), Limit(10, _DAY))
USER_LIMITS = (Limit(5, 60), Limit(30, _DAY))

# 回傳 0 = 通過並計數；否則回傳還要等幾秒（有多個上限被超過時取最久的）
_CHECK_AND_COUNT = """
local wait = 0
for i, key in ipairs(KEYS) do
    if tonumber(redis.call('GET', key) or '0') >= tonumber(ARGV[2 * i - 1]) then
        local ttl = redis.call('TTL', key)
        wait = math.max(wait, ttl > 0 and ttl or 1)
    end
end
if wait > 0 then
    return wait
end
for i, key in ipairs(KEYS) do
    if redis.call('INCR', key) == 1 then
        redis.call('EXPIRE', key, tonumber(ARGV[2 * i]))
    end
end
return 0
"""


async def hit(r, identity: str, limits: tuple[Limit, ...]) -> int:
    """計一次。回傳 0 = 通過，否則回傳還要等幾秒。identity 例如 user:<id> 或 ip:<address>。"""
    if not limits:
        return 0
    keys = [f"chef:rate:chat:{identity}:{limit.seconds}" for limit in limits]
    args = [value for limit in limits for value in (limit.count, limit.seconds)]
    return int(await r.eval(_CHECK_AND_COUNT, len(keys), *keys, *args))
