"""离线/无模型时的兜底 VLM。

两种模式：
1. 若视频同目录存在 `<视频名>.observations.json` 旁挂文件，直接读取（用于回归测试与演示，
   让你在没有 API key 的情况下也能验证整条推理链）。
2. 否则用本地 CV 结果生成"低置信度、明确标注来自本地启发式"的观察，
   绝不伪装成模型输出，也绝不编造具体型号/航班号。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import FrameInfo
from ..utils import clip
from .base import BaseVLM, Observation, VLMBatch, observations_from_payload


class MockVLM(BaseVLM):
    name = "mock"
    model = "none"

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.model = "heuristic-fallback"

    def _sidecar(self, frames: list[FrameInfo]) -> dict[str, Any] | None:
        if not frames:
            return None
        p = Path(frames[0].path)
        # 帧在 outputs/<job>/frames 下，原始视频名不在路径里，所以向上找 job 目录里的 sidecar
        for base in [p.parent, p.parent.parent, p.parent.parent.parent]:
            for cand in list(base.glob("*.observations.json")) + list(base.glob("observations.json")):
                try:
                    return json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return None

    def analyze_frames(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        batch = VLMBatch(provider=self.name, model=self.model)

        sidecar = self._sidecar(frames)
        if sidecar:
            payloads = sidecar.get("frames") if isinstance(sidecar.get("frames"), list) else [sidecar]
            for i, payload in enumerate(payloads):
                if not isinstance(payload, dict):
                    continue
                frame = frames[min(i, len(frames) - 1)] if frames else None
                batch.observations.extend(observations_from_payload(payload, frame))
                if frame:
                    batch.per_frame[frame.id] = payload
            if isinstance(sidecar.get("summary"), str):
                batch.scene_summary = sidecar["summary"]
            if isinstance(sidecar.get("synthesis"), dict):
                batch.synthesis["merged"] = sidecar["synthesis"]
                for ob in observations_from_payload(sidecar["synthesis"], None):
                    ob.raw["from"] = "sidecar_synthesis"
                    batch.observations.append(ob)
            batch.ok = bool(batch.observations)
            batch.warnings.append("视觉模型为 mock，证据来自旁挂的 observations.json（测试/演示模式）")
            return batch

        # 无模型、无旁挂：只用可解释的本地统计，且明确标注
        batch.ok = False
        batch.warnings.append(
            "未配置视觉模型（vlm.provider=mock），无法做机型/涂装/文字的语义判读。"
            "系统仅使用本地图像统计与 OCR 结果。要获得高质量结论，请配置 VLM_API_KEY。"
        )
        if frames:
            batch.scene_summary = (
                f"共 {len(frames)} 个关键帧，时长约 {frames[-1].timestamp_s:.1f}s；"
                "未启用语义视觉模型，因此不给出机型、航司、地点等需要识别的结论。"
            )
        return batch
