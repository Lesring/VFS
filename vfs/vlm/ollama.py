from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from ..frames import frame_bytes
from ..models import FrameInfo
from ..utils import clip, squash
from .base import BaseVLM, VLMBatch, extract_json, observations_from_payload
from .prompts import frame_prompt, synthesis_prompt


class OllamaVLM(BaseVLM):
    """本地 Ollama：ollama pull qwen2.5vl:7b / llama3.2-vision / minicpm-v"""

    name = "ollama"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg or {}
        self.model = str(self.cfg.get("model") or "qwen2.5vl:7b")
        self.base_url = str(self.cfg.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        self.max_side = int(self.cfg.get("max_image_side") or 1024)
        self.quality = int(self.cfg.get("jpeg_quality") or 85)
        self.temperature = float(self.cfg.get("temperature") or 0.1)
        self.timeout = float(self.cfg.get("timeout_s") or 180)
        self.max_frames = int(self.cfg.get("max_frames") or 8)
        self.synthesize = bool(self.cfg.get("synthesize", True))

    async def _chat(self, prompt: str, images: list[bytes] | None = None, fmt_json: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 1200},
        }
        if images:
            payload["messages"][0]["images"] = [base64.b64encode(b).decode("ascii") for b in images]
        if fmt_json:
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {clip(r.text, 240)}")
            data = r.json()
        msg = (data.get("message") or {})
        return str(msg.get("content") or "")

    async def analyze_frames_async(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        batch = VLMBatch(provider=self.name, model=self.model)
        use = frames[: self.max_frames]
        notes: list[str] = []
        for i, f in enumerate(use):
            try:
                img = frame_bytes(f.path, self.max_side, self.quality)
                text = await self._chat(frame_prompt(question, hints, i, len(use), f.timestamp_s), [img])
                payload = extract_json(text)
                if payload:
                    batch.observations.extend(observations_from_payload(payload, f))
                    batch.per_frame[f.id] = payload
                    notes.append(f"- {f.timestamp_s:.1f}s: {squash(str(payload.get('scene') or ''))}")
            except Exception as e:
                batch.warnings.append(f"帧 {f.id} 分析失败：{clip(str(e), 140)}")
        if self.synthesize and notes:
            try:
                text = await self._chat(synthesis_prompt(question, hints, "\n".join(notes)))
                payload = extract_json(text)
                if payload:
                    batch.synthesis["merged"] = payload
                    for ob in observations_from_payload(payload, None):
                        ob.raw["from"] = "synthesis"
                        batch.observations.append(ob)
            except Exception as e:
                batch.warnings.append(f"跨帧综合失败：{clip(str(e), 140)}")
        batch.ok = bool(batch.observations)
        return batch

    def analyze_frames(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        return asyncio.run(self.analyze_frames_async(frames, question, hints))
