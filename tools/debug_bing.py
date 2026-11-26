"""调试 Bing 解析：到底是请求失败、HTML 结构变了，还是被缓存了空页。"""
from __future__ import annotations

import asyncio
import re

from vfs.kb.http import CachedClient


async def main() -> int:
    client = CachedClient("data/cache", ttl_s=3600, timeout_s=20)
    for host, lang, q in [("www.bing.com", "en", "Airbus A320 range"),
                          ("cn.bing.com", "zh", "东方航空 MU5101")]:
        url = f"https://{host}/search?q={q.replace(' ', '+')}&setlang={'zh-CN' if lang=='zh' else 'en-US'}&count=8"
        html = await client.get_text(url)
        print("=" * 60)
        print(host, "len:", len(html or ""))
        if not html:
            print("  空响应")
            continue
        print("  b_algo:", html.count('class="b_algo"'))
        print("  <h2:", html.count("<h2"))
        print("  <li:", html.count("<li"))
        print("  has captcha:", "captcha" in html.lower())
        print("  has consent:", "consent" in html.lower())
        # 抽样结构
        m = re.search(r'<li class="b_algo"(.*?)</li>', html, re.S)
        if m:
            print("  首个条目片段:", re.sub(r"\s+", " ", m.group(1))[:300])
        else:
            # 看看有没有别的容器
            for pat in ['class="b_algo"', 'class="b_ans"', 'id="b_results"', '<h2>']:
                print(f"  contains {pat}:", pat in html)
            idx = html.find('<h2')
            print("  首个 h2 附近:", re.sub(r"\s+", " ", html[max(0,idx-100):idx+250]))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
