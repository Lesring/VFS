from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..frames import frame_to_data_url, make_contact_sheet
from ..models import FrameInfo
from ..utils import clip, squash
from .base import BaseVLM, Observation, VLMBatch, extract_json, observations_from_payload
from .prompts import contact_sheet_prompt, frame_prompt, synthesis_prompt

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatVLM(BaseVLM):
    """任何兼容 /v1/chat/completions 的多模态服务：OpenAI、DashScope、智谱、Moonshot、
    硅基流动、vLLM、LM Studio、DeepSeek(部分模型) 等。"""

    name = "openai_compat"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg or {}
        self.model = str(self.cfg.get("model") or "gpt-4o-mini")
        self.base_url = str(self.cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = str(self.cfg.get("api_key") or "")
        self.max_side = int(self.cfg.get("max_image_side") or 1280)
        self.quality = int(self.cfg.get("jpeg_quality") or 85)
        self.temperature = float(self.cfg.get("temperature") or 0.1)
        self.timeout = float(self.cfg.get("timeout_s") or 120)
        self.max_frames = int(self.cfg.get("max_frames") or 12)
        self.synthesize = bool(self.cfg.get("synthesize", True))
        self.concurrency = int(self.cfg.get("concurrency") or 4)

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def _chat(self, messages: list[dict[str, Any]], max_tokens: int = 1400) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        last_err = ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(3):
                try:
                    r = await client.post(self._endpoint(), json=payload, headers=self._headers())
                    if r.status_code in RETRY_STATUS and attempt < 2:
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
                    if r.status_code >= 400:
                        last_err = f"HTTP {r.status_code}: {clip(r.text, 300)}"
                        break
                    data = r.json()
                    return _read_text(data)
                except Exception as e:  # 网络抖动
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < 2:
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
        raise RuntimeError(last_err or "视觉模型调用失败")

    async def _one_frame(self, frame: FrameInfo, question: str, hints: str, idx: int, total: int) -> tuple[FrameInfo, dict | None, str]:
        prompt = frame_prompt(question, hints, idx, total, frame.timestamp_s)
        data_url = frame_to_data_url(frame.path, self.max_side, self.quality)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }
        ]
        text = await self._chat(messages)
        return frame, extract_json(text), text

    # ------------------------------------------------------------------ #
    async def analyze_frames_async(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        batch = VLMBatch(provider=self.name, model=self.model)
        use = frames[: self.max_frames]
        if not use:
            batch.ok = False
            batch.warnings.append("没有可用关键帧")
            return batch

        sem = asyncio.Semaphore(self.concurrency)

        async def worker(i: int, f: FrameInfo):
            async with sem:
                try:
                    return await self._one_frame(f, question, hints, i, len(use))
                except Exception as e:
                    batch.warnings.append(f"帧 {f.id} 分析失败：{clip(str(e), 160)}")
                    return f, None, ""

        results = await asyncio.gather(*(worker(i, f) for i, f in enumerate(use)))
        notes: list[str] = []
        for frame, payload, raw_text in results:
            if payload:
                batch.observations.extend(observations_from_payload(payload, frame))
                batch.per_frame[frame.id] = payload
                scene = squash(str(payload.get("scene") or ""))
                notes.append(f"- {frame.timestamp_s:.1f}s: {scene or '（无摘要）'}")
                batch.scene_summary = batch.scene_summary or scene
            elif raw_text:
                notes.append(f"- {frame.timestamp_s:.1f}s: 模型输出无法解析为 JSON：{clip(raw_text, 120)}")

        # 总览图：捕捉跨帧变化
        try:
            sheet_target = Path(tempfile.gettempdir()) / f"_vfs_sheet_{uuid.uuid4().hex[:8]}.jpg"
            sheet_path = make_contact_sheet([f.path for f in use], sheet_target)
            if sheet_path:
                prompt = contact_sheet_prompt(question, hints, [f.timestamp_s for f in use])
                data_url = frame_to_data_url(sheet_path, self.max_side * 2, self.quality)
                text = await self._chat(
                    [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ]}],
                    max_tokens=1600,
                )
                payload = extract_json(text)
                if payload:
                    batch.per_frame["__sheet__"] = payload
                    batch.synthesis["sheet"] = payload
                    # 总览观察置信度略低，作为补充
                    for ob in observations_from_payload(payload, None):
                        ob.confidence *= 0.92
                        ob.raw["from"] = "contact_sheet"
                        batch.observations.append(ob)
        except Exception as e:
            batch.warnings.append(f"总览图分析失败：{clip(str(e), 160)}")

        # 文本综合
        if self.synthesize and notes:
            try:
                text = await self._chat(
                    [{"role": "user", "content": synthesis_prompt(question, hints, "\n".join(notes))}],
                    max_tokens=1800,
                )
                payload = extract_json(text)
                if payload:
                    batch.synthesis["merged"] = payload
                    for ob in observations_from_payload(payload, None):
                        ob.raw["from"] = "synthesis"
                        ob.confidence = min(1.0, ob.confidence * 1.05)
                        batch.observations.append(ob)
            except Exception as e:
                batch.warnings.append(f"跨帧综合失败：{clip(str(e), 160)}")

        batch.ok = bool(batch.observations)
        if not batch.ok:
            batch.warnings.append("视觉模型未返回任何可解析观察")
        return batch

    def analyze_frames(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        return asyncio.run(self.analyze_frames_async(frames, question, hints))


def _read_text(data: dict[str, Any]) -> str:
    """兼容 chat/completions 的多种返回结构。"""
    try:
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                buf = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                        buf.append(part.get("text") or "")
                    elif isinstance(part, str):
                        buf.append(part)
                return "".join(buf)
            # 部分服务把结果放在 reasoning_content
            if msg.get("reasoning_content"):
                return str(msg["reasoning_content"])
    except Exception:
        pass
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    if isinstance(data.get("response"), str):
        return data["response"]
    return json.dumps(data, ensure_ascii=False)[:4000]
