"""端到端验证 Web 接口：上传 → SSE 进度 → 结果渲染。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8931"
VIDEO = Path("data/demo/airport_plane.mp4")


def main() -> int:
    if not VIDEO.exists():
        print("演示视频不存在，先运行 tools/make_demo_video.py")
        return 1

    with httpx.Client(timeout=180) as c:
        with VIDEO.open("rb") as fh:
            r = c.post(
                f"{BASE}/api/analyze",
                files={"file": (VIDEO.name, fh, "video/mp4")},
                data={"question": "这架飞机的目的地是哪里？", "online": "true"},
            )
        print("analyze:", r.status_code, r.text[:300])
        r.raise_for_status()
        job = r.json()["job_id"]

        # SSE
        stages: list[str] = []
        with c.stream("GET", f"{BASE}/api/jobs/{job}/stream", timeout=180) as s:
            for line in s.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                stages.append(ev["stage"])
                print(f"  · [{ev['stage']}] {ev['message']}")

        print("\nstages:", " → ".join(dict.fromkeys(stages)))

        res = c.get(f"{BASE}/api/jobs/{job}/result")
        print("result:", res.status_code)
        if res.status_code != 200:
            print(res.text[:500])
            return 1
        d = res.json()

        # 断言关键结构存在
        ok = True
        for key in ("answer", "frames", "evidence", "slots", "timeline", "job_id"):
            if key not in d:
                print(f"缺少字段 {key}")
                ok = False
        if not d.get("frames"):
            print("没有关键帧")
            ok = False
        if not d.get("answer"):
            print("答案为空")
            ok = False

        # 帧图片可访问
        if d.get("frames"):
            fn = Path(d["frames"][0]["path"]).name
            fr = c.get(f"{BASE}/api/frames/{job}/{fn}")
            print(f"frame image: {fr.status_code}, {len(fr.content)} bytes")
            ok = ok and fr.status_code == 200 and len(fr.content) > 100

        print("\n答案摘要：", d["answer"].splitlines()[0][:160])
        print("证据条数：", len(d["evidence"]), " 槽位数：", len(d["slots"]))
        print("\n通过" if ok else "\n失败")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
