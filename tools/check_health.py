"""检查 /api/health/sources 与 /api/settings 接口。"""
from __future__ import annotations

import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8931"


def main() -> int:
    with httpx.Client(timeout=90) as c:
        r = c.get(f"{BASE}/api/health/sources")
        print("health:", r.status_code)
        d = r.json()
        print(f"  strategy={d['strategy']} degraded={d['degraded']}")
        print(f"  ordered={d['ordered']}")
        for k, v in d["sources"].items():
            flag = "OK " if v["ok"] else "NO "
            print(f"  {flag}{k:14s} {v.get('detail', '')}")

        r2 = c.get(f"{BASE}/api/settings")
        print("\nsettings:", r2.status_code)
        s = r2.json()
        print("  status:", json.dumps(s["status"], ensure_ascii=False))
        print(f"  fields={len(s['fields'])}")
        for f in s["fields"][:6]:
            val = f.get("masked") if f["type"] == "password" else f.get("value")
            print(f"    {f['group']:6s} {f['key']:32s} type={f['type']:8s} value={val}")
        # 确认密钥不回传明文
        for f in s["fields"]:
            if f["type"] == "password" and f.get("value"):
                print("  !! 密钥字段泄露明文:", f["key"])
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
