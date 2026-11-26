"""验证地图：无 key 时走 OSM，有 key 时走 Google。"""
from __future__ import annotations

import asyncio
import sys

from vfs.kb.http import CachedClient
from vfs.models import Candidate, SlotResult
from vfs.retrieval import build_map


def make_slots() -> dict[str, SlotResult]:
    return {
        "destination": SlotResult(
            slot="destination", label="目的地",
            candidates=[Candidate(value="Beijing Capital International Airport (PEK/ZBAA)",
                                  score=0.95,
                                  details={"lat": 40.0801, "lon": 116.585, "iata": "PEK", "icao": "ZBAA"})],
            chosen="Beijing Capital International Airport (PEK/ZBAA)", confidence=0.95),
    }


async def run(cfg: dict, tag: str) -> None:
    import os

    client = CachedClient("data/cache", ttl_s=3600, timeout_s=15)
    outdir = os.path.join("data", "maptest", tag.replace(" ", "_"))
    os.makedirs(outdir, exist_ok=True)
    m = await build_map(client, make_slots(), cfg, job_id="maptest", output_dir=outdir)
    print(f"\n[{tag}]")
    if not m:
        print("  无地图")
        return
    for k, v in m.items():
        print(f"  {k}: {str(v)[:110]}")


async def main() -> int:
    # 1) 无 key：应降级到 OSM
    await run({"maps_enabled": True, "maps_api_key": ""}, "无 Maps key")

    # 2) 关掉地图
    await run({"maps_enabled": False, "maps_api_key": ""}, "关闭地图")

    # 3) 假 key：应尝试 Google 但失败后仍给出链接（不能崩）
    await run({"maps_enabled": True, "maps_api_key": "INVALID_KEY_FOR_TEST"}, "无效 Maps key")

    # 4) 只有地名没有坐标
    slots = {"city": SlotResult(slot="city", label="城市",
                               candidates=[Candidate(value="Shanghai", score=0.7, details={})],
                               chosen="Shanghai", confidence=0.7)}
    client = CachedClient("data/cache", ttl_s=3600, timeout_s=20)
    m = await build_map(client, slots, {"maps_enabled": True, "maps_api_key": ""})
    print("\n[仅城市名]")
    for k, v in (m or {}).items() if m else []:
        print(f"  {k}: {str(v)[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
