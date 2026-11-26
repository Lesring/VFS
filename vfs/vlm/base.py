from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..models import FrameInfo


@dataclass
class Observation:
    slot: str
    value: str
    confidence: float = 0.5
    note: str = ""
    frame_id: str | None = None
    timestamp_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMBatch:
    """一次视觉分析的产物。"""
    provider: str = ""
    model: str = ""
    observations: list[Observation] = field(default_factory=list)
    scene_summary: str = ""
    per_frame: dict[str, dict[str, Any]] = field(default_factory=dict)
    synthesis: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True


VALID_SLOTS = {
    "aircraft_type", "airline", "flight_number", "registration", "airport",
    "origin", "destination", "landmark", "city", "country", "coordinate",
    "date", "time_of_day", "weather", "text_sign", "object", "person_role",
    "brand", "activity", "viewpoint_vehicle", "photographer_ride",
}

_SLOT_ALIASES = {
    "aircraft": "aircraft_type", "aircraft_model": "aircraft_type", "plane_type": "aircraft_type",
    "plane": "aircraft_type", "机型": "aircraft_type",
    "carrier": "airline", "airline_name": "airline", "航空公司": "airline", "航司": "airline",
    "flight": "flight_number", "flight_no": "flight_number", "航班": "flight_number",
    "tail_number": "registration", "reg": "registration", "tail": "registration", "注册号": "registration",
    "airport_name": "airport", "机场": "airport",
    "from": "origin", "departure": "origin", "出发": "origin", "出发地": "origin",
    "to": "destination", "arrival": "destination", "目的地": "destination", "到达": "destination",
    "building": "landmark", "地标": "landmark",
    "城市": "city", "国家": "country", "坐标": "coordinate",
    "日期": "date", "时间": "time_of_day", "time": "time_of_day",
    "天气": "weather", "sign": "text_sign", "text": "text_sign", "文字": "text_sign",
    "物体": "object", "人物": "person_role", "品牌": "brand", "活动": "activity",
}


def normalize_slot(slot: str) -> str | None:
    s = (slot or "").strip().lower().replace("-", "_").replace(" ", "_")
    s = _SLOT_ALIASES.get(s, s)
    if s in VALID_SLOTS:
        return s
    if s in _SLOT_ALIASES:
        return _SLOT_ALIASES[s]
    return None


def _coerce_conf(v: Any, default: float = 0.5) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f > 1.0:
        f = f / 100.0
    return max(0.0, min(1.0, f))


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个合法的 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json 围栏
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # 括号配对扫描
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def observations_from_payload(payload: dict[str, Any], frame: FrameInfo | None = None) -> list[Observation]:
    """把模型返回的 JSON 转成 Observation 列表。"""
    out: list[Observation] = []
    items: list[tuple[str, Any]] = []

    # 形式 A: {"observations":[{"slot":..,"value":..}]}
    for key in ("observations", "findings", "facts", "entities"):
        arr = payload.get(key)
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                slot = normalize_slot(str(it.get("slot") or it.get("type") or it.get("key") or ""))
                value = it.get("value") or it.get("text") or it.get("name")
                if slot and value not in (None, ""):
                    out.append(
                        Observation(
                            slot=slot,
                            value=str(value).strip(),
                            confidence=_coerce_conf(it.get("confidence", it.get("confidence_score"))),
                            note=str(it.get("note") or it.get("reason") or ""),
                            frame_id=frame.id if frame else None,
                            timestamp_s=frame.timestamp_s if frame else None,
                            raw=it,
                        )
                    )

    # 形式 B: {"aircraft_type": {"value": "A320", "confidence": 0.8}, ...}
    for key, val in payload.items():
        slot = normalize_slot(key)
        if not slot:
            continue
        if isinstance(val, dict):
            v = val.get("value") or val.get("text") or val.get("name")
            if v in (None, ""):
                continue
            items.append((slot, (
                str(v).strip(),
                _coerce_conf(val.get("confidence", val.get("confidence_score"))),
                str(val.get("note") or val.get("reason") or ""),
                val,
            )))
        elif isinstance(val, list):
            for v in val[:6]:
                if isinstance(v, str) and v.strip():
                    items.append((slot, (v.strip(), 0.45, "", {})))
                elif isinstance(v, dict):
                    vv = v.get("value") or v.get("text") or v.get("name")
                    if vv:
                        items.append((slot, (
                            str(vv).strip(),
                            _coerce_conf(v.get("confidence")),
                            str(v.get("note") or ""),
                            v,
                        )))
        elif isinstance(val, (str, int, float)) and str(val).strip():
            items.append((slot, (str(val).strip(), 0.5, "", {})))

    for slot, (value, conf, note, raw) in items:
        out.append(
            Observation(
                slot=slot, value=value, confidence=conf, note=note,
                frame_id=frame.id if frame else None,
                timestamp_s=frame.timestamp_s if frame else None,
                raw=raw if isinstance(raw, dict) else {},
            )
        )
    return out


class BaseVLM(ABC):
    name = "base"
    model = ""

    @abstractmethod
    def analyze_frames(self, frames: list[FrameInfo], question: str, hints: str = "") -> VLMBatch:
        ...

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model}


def build_vlm(cfg: dict[str, Any], force: str | None = None) -> BaseVLM:
    provider = (force or cfg.get("provider") or "mock").lower()
    if provider in ("openai", "openai_compat", "compat", "vllm", "lmstudio", "qwen", "dashscope", "deepseek", "moonshot", "zhipu"):
        from .openai_compat import OpenAICompatVLM
        return OpenAICompatVLM(cfg)
    if provider in ("ollama", "local"):
        from .ollama import OllamaVLM
        return OllamaVLM(cfg)
    if provider in ("anthropic", "claude"):
        from .anthropic_vlm import AnthropicVLM
        return AnthropicVLM(cfg)
    from .mock import MockVLM
    return MockVLM(cfg)


_CACHE: dict[str, BaseVLM] = {}


def get_vlm(cfg: dict[str, Any], force: str | None = None) -> BaseVLM:
    key = force or cfg.get("provider", "mock")
    if key not in _CACHE:
        _CACHE[key] = build_vlm(cfg, force)
    return _CACHE[key]
