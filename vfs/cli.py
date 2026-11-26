from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONFIG
from .pipeline import AnalysisJob
from .utils import ensure_dir, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vfs",
        description="视频取证推理：输入视频 + 问题，自动抽取线索、联网核实、给出带推理链的答案。",
    )
    ap.add_argument("video", help="视频或图片路径")
    ap.add_argument("-q", "--question", default="", help="要回答的问题，如「这架飞机的目的地是哪里？」")
    ap.add_argument("--hints", default="", help="补充提示，如「拍摄于 2024 年 5 月，上海」")
    ap.add_argument("--offline", action="store_true", help="不联网，只用离线知识")
    ap.add_argument("--frames", type=int, default=None, help="最多抽取的关键帧数")
    ap.add_argument("--provider", default=None, help="强制视觉模型后端：mock/openai/ollama/anthropic")
    ap.add_argument("--json", dest="json_out", default=None, help="把完整结果写到该 JSON 文件")
    ap.add_argument("--trace", action="store_true", help="打印完整推理链与证据表")
    args = ap.parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"找不到文件：{video}", file=sys.stderr)
        return 2

    cfg = dict(CONFIG)
    if args.provider:
        cfg["vlm"] = {**cfg["vlm"], "provider": args.provider}

    job = AnalysisJob(
        video, question=args.question, hints=args.hints, config=cfg,
        online=not args.offline, max_frames=args.frames,
        progress=lambda s, m, d: print(f"  · [{s}] {m}", file=sys.stderr),
    )
    result = job.run()

    print()
    print(result.answer)
    print()
    print(f"— 耗时：{' / '.join(f'{t.stage} {t.ms}ms' for t in result.timeline)}")

    if args.trace:
        print()
        print("=" * 70)
        print("推理链")
        print("=" * 70)
        for i, t in enumerate(result.reasoning_trace, 1):
            print(f"{i:3d}. {t}")
        print()
        print("=" * 70)
        print("证据表")
        print("=" * 70)
        print(f"{'ID':<7}{'类型':<12}{'槽位':<16}{'置信':<7}内容")
        for e in result.evidence:
            print(f"{e.id:<7}{e.kind:<12}{e.slot:<16}{e.confidence:<7.2f}{e.claim[:90]}")

    out_json = args.json_out or str(Path(cfg["paths"]["outputs_dir"]) / result.job_id / "result.json")
    write_json(out_json, result.model_dump(mode="json"))
    print()
    print(f"完整结果：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
