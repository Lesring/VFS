#!/usr/bin/env python
"""便捷启动脚本：python run.py  （可选 --port 8000）"""
from __future__ import annotations

import argparse

from vfs.app import run
from vfs.config import CONFIG

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="启动视频取证推理系统 Web 界面")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    host = args.host or CONFIG["server"]["host"]
    port = args.port or CONFIG["server"]["port"]
    print(f"正在启动，访问 http://{host}:{port}")
    run(host, port)
