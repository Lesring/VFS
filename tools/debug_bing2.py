"""看 Bing 单个 b_algo 里 h2 / a / cite 的确切结构。"""
from __future__ import annotations

import asyncio
import re

from vfs.kb.http import CachedClient
from vfs.kb.search import _BING_ITEM, _BING_H2, _BING_A


async def main() -> int:
    client = CachedClient("data/cache", ttl_s=60, timeout_s=20)
    html = await client.get_text(
        "https://www.bing.com/search?q=Airbus+A320&setlang=en-US&count=8", force=True)
    print("len", len(html))
    blocks = _BING_ITEM.findall(html)
    print("blocks", len(blocks))
    if not blocks:
        return 0
    b = blocks[0]
    # 去掉 style/link 干扰
    b_clean = re.sub(r"<link[^>]*>", "", b)
    b_clean = re.sub(r"<style.*?</style>", "", b_clean, flags=re.S)
    print("\n--- 清洗后前 1500 字符 ---")
    print(re.sub(r"\s+", " ", b_clean)[:1500])

    h2 = _BING_H2.search(b_clean)
    print("\n--- h2 ---")
    print(re.sub(r"\s+", " ", h2.group(1))[:600] if h2 else "无 h2")
    all_a = _BING_A.findall(b_clean)
    print("\n--- 所有 a href ---")
    for a in all_a[:6]:
        print("  ", a[:120])
    cites = re.findall(r"<cite[^>]*>(.*?)</cite>", b_clean, re.S)
    print("\n--- cite ---")
    for c in cites[:4]:
        print("  ", re.sub(r"<[^>]+>", "", c)[:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
