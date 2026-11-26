"""验证配置保存 → 热生效（无需重启）。"""
from __future__ import annotations

import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8931"


def get(c, path):
    r = c.get(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()


def main() -> int:
    with httpx.Client(timeout=60) as c:
        before = get(c, "/api/config")
        print("改前:", before["vlm_provider"], before["vlm_model"], before["region_strategy"])

        # 1) 改模型（不填 key，验证 provider/model 热切换）
        r = c.post(f"{BASE}/api/settings", json={
            "vlm.provider": "openai",
            "vlm.model": "qwen-vl-max",
            "vlm.base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vlm.max_frames": 8,
            "retrieval.region_strategy": "domestic_first",
        })
        print("save:", r.status_code, r.json())
        after = get(c, "/api/config")
        print("改后:", after["vlm_provider"], after["vlm_model"], after["region_strategy"])

        assert after["vlm_provider"] == "openai", "provider 未热生效"
        assert after["vlm_model"] == "qwen-vl-max", "model 未热生效"
        assert after["region_strategy"] == "domestic_first", "region 未热生效"
        print("OK 热生效")

        # 2) 验证密钥写入且不回传明文
        c.post(f"{BASE}/api/settings", json={"vlm.api_key": "sk-test-secret-1234567890"})
        s = get(c, "/api/settings")
        keyf = next(f for f in s["fields"] if f["key"] == "vlm.api_key")
        print("key 字段:", {k: keyf[k] for k in ("value", "masked", "configured")})
        assert keyf["value"] == "", "密钥回传了明文！"
        assert keyf["configured"] is True
        print("OK 密钥打码")

        # 3) 留空不应清空已有 key
        c.post(f"{BASE}/api/settings", json={"vlm.api_key": ""})
        s2 = get(c, "/api/settings")
        k2 = next(f for f in s2["fields"] if f["key"] == "vlm.api_key")
        assert k2["configured"] is True, "留空把已有 key 清掉了！"
        print("OK 留空不清空")

        # 4) 复原
        c.post(f"{BASE}/api/settings", json={
            "vlm.provider": "mock", "vlm.model": "gpt-4o-mini",
            "retrieval.region_strategy": "overseas_first",
        })
        print("已复原")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
