from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..frames import frame_bytes
from ..models import FrameInfo
from ..utils import clip, squash
from .base import BaseVLM, VLMBatch, extract_json, observations_from_payload
from .prompts import frame_prompt, synthesis_prompt


class AnthropicVLM(BaseVLM):
    name = "anthropic"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg or {}
        self.model = str(self.cfg.get("model") or "claude-sonnet-4-5")
        self.base_url = str(self.cfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
        self.api_key = str(self.cfg.get("api_key") or "")
        self.version = str(self.cfg.get("anthropic_version") or "2023-06-01")
        self.max_side = int(self.cfg.get("max_image_side") or 1280)
        self.quality = int(self.cfg.get("jpeg_quality") or 85)
        self.timeout = float(self.cfg.get("timeout_s") or 120)
        self.max_frames = int(self.cfg.get("max_frames") or 10)
        self.synthesize = bool(self.cfg.get("synthesize", True))

    async def _chat(self, content: list[dict[str, Any]], max_tokens: int = 1400) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": content}],
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/v1/messages", json=payload, headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"Anthropic HTTP {r.status_code}: {clip(r.text, 260)}")
            data = r.json()
        blocks = data.get("content") or []
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")

    async def analyze_frames_async(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        batch = VLMBatch(provider=self.name, model=self.model)
        use = frames[: self.max_frames]
        notes: list[str] = []
        for i, f in enumerate(use):
            try:
                img_b64 = __import__("base64").b64encode(frame_bytes(f.path, self.max_side, self.quality)).decode()
                content = [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": frame_prompt(question, hints, i, len(use), f.timestamp_s)},
                ]
                text = await self._chat(content)
                payload = extract_json(text)
                if payload:
                    batch.observations.extend(observations_from_payload(payload, f))
                    batch.per_frame[f.id] = payload
                    notes.append(f"- {f.timestamp_s:.1f}s: {squash(str(payload.get('scene') or ''))}")
            except Exception as e:
                batch.warnings.append(f"帧 {f.id} 分析失败：{clip(str(e), 140)}")
        if self.synthesize and notes:
            try:
                text = await self._chat([{"type": "text", "text": synthesis_prompt(question, hints, "\n".join(notes))}])
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
