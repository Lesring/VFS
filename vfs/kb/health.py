"""数据源可达性探测与降级决策。

为什么需要这个：
"境外优先、不可达才用国内"如果只是按固定顺序试，会被墙的域名拖死——
实测 www.google.com.hk 会卡 5 秒然后 SSL handshake 失败，每个查询都这样就没法用了。

做法：
1. 带超时的 DNS 预检（Windows 上 urllib 的 timeout 管不住 DNS 解析）
2. 探测结果按 TTL 缓存，整个进程共享
3. 检索前先问一遍"这个源通不通"，不通直接跳过
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any, Iterable

# 各数据源的代表性域名，用于判断"这一组能不能用"
SOURCE_HOSTS: dict[str, str] = {
    "google": "www.google.com",
    "google_api": "www.googleapis.com",
    "google_maps": "maps.googleapis.com",
    "wikimedia": "api.wikimedia.org",
    "bing": "www.bing.com",
    "duckduckgo": "api.duckduckgo.com",
    "adsbdb": "api.adsbdb.com",
    "opensky": "opensky-network.org",
    "openflights": "raw.githubusercontent.com",
    "tavily": "api.tavily.com",
    "brave": "api.search.brave.com",
    "serper": "google.serper.dev",
    "sogou": "www.sogou.com",
    "baidu": "www.baidu.com",
    "nominatim": "nominatim.openstreetmap.org",
    "osm_tiles": "tile.openstreetmap.org",
}

# ---- 全局探测缓存：host -> (ok, checked_at, detail) ----
_CACHE: dict[str, tuple[bool, float, str]] = {}
_TTL = 600.0  # 10 分钟，避免每次分析都重探


def _dns_ok(host: str, timeout: float = 3.0) -> tuple[bool, str]:
    """带超时的 DNS 解析。

    socket.getaddrinfo 不受 socket 超时控制，被墙域名可能阻塞几十秒，
    所以必须扔进独立线程并自己对 future 设超时。
    """
    def _resolve() -> tuple[bool, str]:
        try:
            infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
            ips = sorted({i[4][0] for i in infos})
            return True, ",".join(ips[:2])
        except Exception as e:
            return False, type(e).__name__

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已在事件循环里：不能再嵌套 run_until_complete，用线程池 + 轮询
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_resolve)
            try:
                return fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return False, "dns_timeout"

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_resolve)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False, "dns_timeout"


def check(name: str, ttl: float | None = None, force: bool = False) -> bool:
    """某个数据源是否可达。带缓存。"""
    host = SOURCE_HOSTS.get(name, name)
    now = time.time()
    ttl = _TTL if ttl is None else ttl
    cached = _CACHE.get(host)
    if cached and not force and now - cached[1] < ttl:
        return cached[0]
    ok, detail = _dns_ok(host)
    _CACHE[host] = (ok, now, detail)
    return ok


def detail(name: str) -> dict[str, Any]:
    host = SOURCE_HOSTS.get(name, name)
    c = _CACHE.get(host)
    if not c:
        return {"host": host, "known": False}
    return {"host": host, "known": True, "ok": c[0], "age_s": round(time.time() - c[1]), "detail": c[2]}


def snapshot() -> dict[str, Any]:
    """当前所有已探测源的状态，用于设置页展示。"""
    out = {}
    for name, host in SOURCE_HOSTS.items():
        c = _CACHE.get(host)
        if c:
            out[name] = {"host": host, "ok": c[0], "age_s": round(time.time() - c[1]), "detail": c[2]}
    return out


def reset() -> None:
    _CACHE.clear()


def first_available(names: Iterable[str], default: str | None = None) -> str | None:
    """按顺序返回第一个可达的源。全不通则返回 default。"""
    for n in names:
        if check(n):
            return n
    return default


# ---- 区域策略 ----

# 境外源（按优先级）
OVERSEAS = ["google", "tavily", "brave", "serper", "bing", "duckduckgo", "wikimedia"]
# 国内源
DOMESTIC = ["sogou", "baidu"]


def ordered_sources(strategy: str = "overseas_first") -> list[str]:
    """根据区域策略给出尝试顺序。

    overseas_first: 境外优先，逐个探测；不通的自动跳过，最后兜底国内。
    domestic_first: 国内优先，境外作为备选。
    overseas_only / domestic_only: 只用一个区域。
    """
    strategy = (strategy or "overseas_first").lower()

    def usable(names: list[str]) -> list[str]:
        return [n for n in names if check(n)]

    if strategy == "domestic_only":
        return DOMESTIC
    if strategy == "overseas_only":
        return OVERSEAS

    os_ok = usable(OVERSEAS)
    dom = DOMESTIC
    if strategy == "domestic_first":
        return dom + os_ok
    # overseas_first（默认）
    return os_ok + dom
