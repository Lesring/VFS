"""验证搜索降级链：境外优先 → 有 key 走 Google CSE，无 key 走 Google News → 再兜底。"""
from __future__ import annotations

import asyncio

from vfs.kb import health
from vfs.kb.http import CachedClient
from vfs.kb.search import web_search


async def main() -> int:
    client = CachedClient("data/cache", ttl_s=3600, timeout_s=15)

    cases = [
        ("无 Google key（应走 News RSS 或降级）", dict(google_key="", google_cx="")),
        ("有假 Google key（CSE 失败后降级）", dict(google_key="FAKE", google_cx="FAKE_CX")),
    ]
    for tag, kw in cases:
        print("=" * 68)
        print(tag)
        results, attempted, notes = await web_search(
            client, "China Eastern MU5101 flight route", limit=3,
            backend="auto", lang="en", strategy="overseas_first", **kw,
        )
        print("  attempted:", attempted)
        print("  notes:", notes)
        print("  n:", len(results))
        for r in results[:3]:
            print(f"    · [{r.get('engine')}] {r.get('title','')[:70]}")
            print(f"      {r.get('snippet','')[:110]}")

    print("=" * 68)
    print("国内策略（domestic_first）")
    results, attempted, notes = await web_search(
        client, "上海浦东机场 航班", limit=3, backend="auto", lang="zh",
        strategy="domestic_first", google_key="", google_cx="",
    )
    print("  attempted:", attempted, "n:", len(results))
    for r in results[:3]:
        print(f"    · [{r.get('engine')}] {r.get('title','')[:70]}")

    print("=" * 68)
    print("指定后端 bing")
    results, attempted, notes = await web_search(
        client, "Airbus A320 range", limit=2, backend="bing", lang="en",
    )
    print("  attempted:", attempted, "n:", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
