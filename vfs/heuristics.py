"""不依赖任何模型的本地 CV 启发式。作为廉价先验，同时在没有视觉模型时提供最低限度的分析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import Evidence, FrameInfo


def _sky_ratio(img: np.ndarray) -> float:
    """上方 40% 区域里"蓝天/灰天"像素占比。"""
    h = img.shape[0]
    top = img[: max(1, int(h * 0.4))]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = ((hue > 90) & (hue < 135) & (sat > 35) & (val > 90)).mean()
    bright_gray = ((sat < 28) & (val > 140)).mean()
    return float(max(blue, bright_gray))


def _green_ratio(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((hue > 35) & (hue < 85) & (sat > 40) & (val > 40)).mean())


def _metal_ratio(img: np.ndarray) -> float:
    """低饱和中高明度——机身涂装、停机坪常见。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    return float(((sat < 30) & (val > 90) & (val < 235)).mean())


def _horizon_tilt(img: np.ndarray) -> float | None:
    """用最长直线估计地平线倾角（度）。粗略，仅作参考。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90,
                            minLineLength=int(img.shape[1] * 0.35), maxLineGap=12)
    if lines is None:
        return None
    lines = np.asarray(lines).reshape(-1, 4)
    best = None
    for x1, y1, x2, y2 in lines:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= 25:
            ln = np.hypot(x2 - x1, y2 - y1)
            if best is None or ln > best[0]:
                best = (ln, ang)
    return round(float(best[1]), 1) if best else None


def analyze_frames(frames: list[FrameInfo]) -> tuple[list[Evidence], dict[str, Any]]:
    ev: list[Evidence] = []
    summary: dict[str, Any] = {}

    if not frames:
        return ev, summary

    per_frame: list[dict[str, Any]] = []
    for f in frames:
        img = cv2.imread(f.path)
        if img is None:
            continue
        # 统一缩到小图算得快
        small = cv2.resize(img, (320, int(320 * img.shape[0] / max(1, img.shape[1]))), interpolation=cv2.INTER_AREA)
        info = {
            "frame_id": f.id,
            "ts": f.timestamp_s,
            "sky": round(_sky_ratio(small), 3),
            "green": round(_green_ratio(small), 3),
            "metal": round(_metal_ratio(small), 3),
            "tilt": _horizon_tilt(small),
            "brightness": round(float(np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))), 1),
        }
        per_frame.append(info)

    summary["per_frame"] = per_frame
    n = len(per_frame)
    avg = lambda k: float(np.mean([p[k] for p in per_frame if p.get(k) is not None])) if per_frame else 0.0
    sky, green, metal, bright = avg("sky"), avg("green"), avg("metal"), avg("brightness")

    otsu = cv2.threshold(
        np.array([[int(b) for b in [bright]]], dtype=np.uint8), 0, 255, cv2.THRESH_OTSU
    )[0]
    summary.update({"sky_ratio": round(sky, 3), "green_ratio": round(green, 3),
                    "metal_ratio": round(metal, 3), "mean_brightness": round(bright, 1)})

    # 室内/室外
    if sky > 0.18 and metal > 0.12:
        ev.append(Evidence(kind="heuristic", source="local_cv", slot="activity",
                           claim="室外场景，天空可见且大面积金属/混凝土结构", value="outdoor",
                           confidence=0.5, raw={"sky": sky, "metal": metal}))
    elif sky < 0.05 and bright < 110:
        ev.append(Evidence(kind="heuristic", source="local_cv", slot="activity",
                           claim="室内或夜间场景（几乎无天空区域、整体偏暗）", value="indoor_or_night",
                           confidence=0.35, raw={"sky": sky, "brightness": bright}))

    # 时段
    if bright >= 150:
        tod, c = "daytime", 0.55
    elif bright >= 105:
        tod, c = "day_or_dusk", 0.4
    elif bright >= 65:
        tod, c = "dusk_or_dawn", 0.4
    else:
        tod, c = "night", 0.5
    ev.append(Evidence(kind="heuristic", source="local_cv", slot="time_of_day",
                       claim=f"画面平均亮度 {bright:.0f}/255，推断为 {tod}", value=tod,
                       confidence=c, raw={"brightness": bright, "otsu": float(otsu)}))

    # 天气
    contrast = float(np.mean([np.std(cv2.imread(p.path, cv2.IMREAD_GRAYSCALE)) for p in frames if Path(p.path).exists()]))
    summary["contrast"] = round(contrast, 1)
    if sky > 0.25 and contrast > 55:
        ev.append(Evidence(kind="heuristic", source="local_cv", slot="weather",
                           claim="天空占比大且对比度高，倾向晴朗", value="clear",
                           confidence=0.4, raw={"sky": sky, "contrast": contrast}))
    elif contrast < 38:
        ev.append(Evidence(kind="heuristic", source="local_cv", slot="weather",
                           claim="画面对比度低，可能有雾霾/阴天/雨", value="hazy_or_overcast",
                           confidence=0.4, raw={"contrast": contrast}))

    # 绿植 → 城市绿地/郊区
    if green > 0.2:
        ev.append(Evidence(kind="heuristic", source="local_cv", slot="landmark",
                           claim="画面上半部分有大面积植被，室外绿化明显", value="vegetation",
                           confidence=0.3, raw={"green": green}))

    # 地平线倾角 → 拍摄姿态
    tilts = [p["tilt"] for p in per_frame if p.get("tilt") is not None]
    if tilts:
        summary["horizon_tilt_deg"] = round(float(np.mean(tilts)), 1)
        ev.append(Evidence(kind="heuristic", source="local_cv", slot="coordinate",
                           claim=f"检测到地平线倾角约 {summary['horizon_tilt_deg']}°，可辅助判断相机姿态而非位置",
                           value=f"tilt={summary['horizon_tilt_deg']}", confidence=0.2))

    # 场景变化 → 镜头/主体运动
    changes = [f.scene_change for f in frames]
    if changes:
        summary["max_scene_change"] = round(max(changes), 3)
        summary["frame_reasons"] = sorted({f.reason for f in frames})

    # 证据 ID 由 FusionEngine 统一分配，这里不自行生成，避免前缀冲突。
    return ev, summary
