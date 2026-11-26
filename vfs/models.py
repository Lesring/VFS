from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 证据层
# --------------------------------------------------------------------------- #

EvidenceKind = Literal[
    "vlm",          # 视觉模型直接读帧得出的观察
    "ocr",          # OCR 文本
    "metadata",     # 视频容器元数据
    "heuristic",    # 本地 CV 规则（亮度/运动/场景切换）
    "knowledge",    # 离线知识库（机场表、航司表）
    "web",          # 联网检索结果
    "user",         # 用户提供的要求/提示
]


class Evidence(BaseModel):
    id: str = ""
    kind: EvidenceKind = "vlm"
    source: str = ""                      # 具体来源：模型名、URL、文件名
    claim: str = ""                       # 一句话陈述
    slot: str = ""                        # 归属实体槽位，如 aircraft_type / airport
    value: str = ""                       # 抽取出的值
    confidence: float = 0.5               # 0~1
    timestamp_s: float | None = None      # 对应视频时间点
    frame_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    verified: bool | None = None          # 是否被外部知识交叉确认
    url: str | None = None


class FrameInfo(BaseModel):
    id: str
    index: int
    timestamp_s: float
    path: str
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    brightness: float = 0.0
    scene_change: float = 0.0
    reason: str = ""                      # 为什么被选中
    ocr_text: str = ""


# --------------------------------------------------------------------------- #
# 槽位 / 实体
# --------------------------------------------------------------------------- #

# 我们关心的实体槽位。domain 用于路由知识库。
SLOT_REGISTRY: dict[str, dict[str, str]] = {
    "aircraft_type": {"label": "机型", "domain": "aircraft"},
    "airline": {"label": "航空公司", "domain": "airline"},
    "flight_number": {"label": "航班号", "domain": "flight"},
    "registration": {"label": "注册号/机身编号", "domain": "aircraft"},
    "airport": {"label": "机场", "domain": "airport"},
    "origin": {"label": "出发地", "domain": "airport"},
    "destination": {"label": "目的地", "domain": "airport"},
    "landmark": {"label": "地标/建筑", "domain": "geo"},
    "city": {"label": "城市", "domain": "geo"},
    "country": {"label": "国家", "domain": "geo"},
    "coordinate": {"label": "坐标", "domain": "geo"},
    "date": {"label": "日期", "domain": "time"},
    "time_of_day": {"label": "时刻", "domain": "time"},
    "weather": {"label": "天气", "domain": "misc"},
    "text_sign": {"label": "画面文字/招牌", "domain": "misc"},
    "object": {"label": "显著物体", "domain": "misc"},
    "person_role": {"label": "人物/制服", "domain": "misc"},
    "brand": {"label": "品牌/标识", "domain": "misc"},
    "activity": {"label": "正在发生的事", "domain": "misc"},
    "viewpoint_vehicle": {"label": "拍摄载具", "domain": "geo"},
    "photographer_ride": {"label": "拍摄者乘坐班次", "domain": "transit"},
}


class Candidate(BaseModel):
    value: str
    score: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SlotResult(BaseModel):
    slot: str
    label: str
    candidates: list[Candidate] = Field(default_factory=list)
    chosen: str | None = None
    confidence: float = 0.0
    confidence_label: str = "无法判定"
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 知识 / 检索
# --------------------------------------------------------------------------- #


class KnowledgeHit(BaseModel):
    title: str
    url: str | None = None
    snippet: str = ""
    source: str = ""
    score: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 任务
# --------------------------------------------------------------------------- #


class AnalyzeRequest(BaseModel):
    question: str = ""
    # 可选：用户直接给出的提示，如"这是 2024 年 5 月拍的"
    hints: str = ""
    # 是否强制联网
    online: bool = True
    # 覆盖抽帧数量
    max_frames: int | None = None


class VideoMeta(BaseModel):
    filename: str = ""
    path: str = ""
    size_bytes: int = 0
    duration_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    container_meta: dict[str, Any] = Field(default_factory=dict)


class StageTiming(BaseModel):
    stage: str
    ms: int = 0
    note: str = ""


class AnalysisResult(BaseModel):
    job_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    question: str = ""
    hints: str = ""
    video: VideoMeta = Field(default_factory=VideoMeta)
    frames: list[FrameInfo] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    slots: dict[str, SlotResult] = Field(default_factory=dict)
    knowledge: list[KnowledgeHit] = Field(default_factory=list)
    searches: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    answer_confidence: float = 0.0
    answer_confidence_label: str = "无法判定"
    # 地图可视化：{label, lat, lon, provider, static_url, embed_url, link, address, note}
    map: dict[str, Any] | None = None
    timeline: list[StageTiming] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    # 系统识别出的"用户到底在问什么"
    intent: dict[str, Any] = Field(default_factory=dict)
