"""问题理解：把用户的自然语言要求映射成"目标槽位"。

不做大模型调用，纯规则 + 同义词，保证离线可用、可解释、零成本。
"""

from __future__ import annotations

import re

from .models import SLOT_REGISTRY

# 目标槽位的关键词（中文 + 英文）
GOAL_PATTERNS: list[tuple[str, list[str]]] = [
    ("destination", ["目的地", "飞往", "飞向", "去哪儿", "去哪", "到达", "终点", "降落在", "降落机场",
                     "destination", "flying to", "arrive", "where is it going", "where is the plane going"]),
    ("origin", ["出发地", "从哪儿飞", "从哪起飞", "起飞地", "始发", "起飞机场", "origin", "departed from",
                "took off from", "where did it take off"]),
    ("flight_number", ["航班号", "航班", "班次", "哪一班", "飞机班次", "flight number", "flight no", "which flight"]),
    ("photographer_ride", ["拍摄者所乘", "拍摄者乘坐", "乘坐班次", "所乘班次", "火车班次", "列车班次",
                           "photographer ride", "which train", "train number"]),
    ("viewpoint_vehicle", ["乘坐什么", "在什么车上", "拍摄载具", "viewpoint", "from a train"]),
    ("airline", ["航空公司", "哪家航司", "航司", "承运人", "哪个航空", "airline", "carrier"]),
    ("aircraft_type", ["机型", "什么飞机", "型号", "飞机型号", "aircraft type", "model", "what plane",
                       "what aircraft"]),
    ("registration", ["注册号", "机身号", "尾号", "机尾号", "registration", "tail number", "reg"]),
    ("airport", ["哪个机场", "机场", "在哪个机场", "机场名", "airport", "which airport"]),
    ("city", ["城市", "哪个城市", "什么地方", "地点", "city", "where", "location", "place"]),
    ("country", ["国家", "哪国", "country"]),
    ("date", ["日期", "什么时候", "哪一天", "时间", "年份", "date", "when", "what day", "time"]),
    ("time_of_day", ["几点", "时段", "白天", "夜晚", "时刻", "time of day"]),
    ("landmark", ["地标", "建筑", "这是什么地方", "landmark", "building"]),
    ("weather", ["天气", "weather"]),
    ("activity", ["在做什么", "发生什么", "干什么", "activity", "what is happening", "what's going on"]),
    ("person_role", ["谁", "人物", "穿着", "制服", "person", "who", "crew"]),
    ("coordinate", ["坐标", "经纬度", "coordinate", "latitude", "longitude", "gps"]),
    ("text_sign", ["写了什么", "文字", "招牌", "屏幕上", "text", "sign", "written"]),
]

# 问题模板 → 推理任务类型
TASK_HINTS = {
    "flight_route": ["目的地", "出发地", "航班", "飞往", "航线", "destination", "origin", "route"],
    "identify_aircraft": ["机型", "什么飞机", "型号", "aircraft", "plane model"],
    "identify_place": ["哪里", "哪个机场", "什么城市", "什么国家", "地点", "拍摄地点", "where", "location"],
    "identify_time": ["什么时候", "日期", "时间", "date", "when"],
    "osint_geo": ["拍摄地点", "拍摄者", "地理定位", "geolocat", "planes", "trains"],
}


def parse_intent(question: str) -> dict:
    q = (question or "").strip()
    ql = q.lower()
    goals: list[str] = []
    matched_phrases: dict[str, list[str]] = {}

    for slot, pats in GOAL_PATTERNS:
        for p in pats:
            if p in ql:
                if slot not in goals:
                    goals.append(slot)
                matched_phrases.setdefault(slot, []).append(p)

    tasks = []
    for task, kws in TASK_HINTS.items():
        if any(k in ql for k in kws):
            tasks.append(task)

    # "这架飞机要去哪里" 这类没有显式关键词的
    if not goals and q:
        if re.search(r"(去哪|到哪里|目的地|去处)", q):
            goals.append("destination")
        elif re.search(r"(是什么|识别|这是)", q):
            goals.extend(["aircraft_type", "landmark", "city"])

    primary = goals[0] if goals else "activity"
    return {
        "question": q,
        "goals": goals or ["activity"],
        "primary_goal": primary,
        "primary_label": SLOT_REGISTRY.get(primary, {}).get("label", primary),
        "tasks": tasks or ["general"],
        "matched": matched_phrases,
        "is_empty": not q,
    }


def goal_candidates(intent: dict) -> list[str]:
    """当主目标推不出来时，按这个顺序退让到次优可回答的槽位。"""
    order = list(intent.get("goals") or [])
    for s in ("flight_number", "photographer_ride", "airport", "landmark", "city",
              "aircraft_type", "airline", "country", "viewpoint_vehicle",
              "date", "time_of_day", "weather", "activity", "object"):
        if s not in order:
            order.append(s)
    return order
