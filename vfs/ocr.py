"""轻量 OCR。优先用系统 tesseract；找不到可执行文件就静默跳过，绝不阻断主流程。

模型自己读图通常比传统 OCR 强，所以这里主要是在没有视觉模型时兜底，
以及补一些模型容易忽略的小字（登机桥编号、车牌、屏幕）。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .models import Evidence, FrameInfo
from .utils import clip, squash

_TESS_READY: bool | None = None


def tesseract_available() -> bool:
    global _TESS_READY
    if _TESS_READY is not None:
        return _TESS_READY
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESS_READY = True
    except Exception:
        _TESS_READY = False
    return _TESS_READY


def _prep(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)


def _keep(text: str) -> bool:
    """过滤 OCR 噪声：太短、纯符号、字符种类过乱。"""
    t = squash(text)
    if len(t) < 3:
        return False
    alnum = sum(ch.isalnum() for ch in t)
    if alnum < 2 or alnum / max(1, len(t)) < 0.45:
        return False
    if len(set(t)) <= 2 and len(t) > 4:
        return False
    return True


def run_ocr(frames: list[FrameInfo], cfg: dict[str, Any]) -> tuple[list[Evidence], dict[str, Any]]:
    if not cfg.get("enabled", True):
        return [], {"skipped": "disabled"}
    if not tesseract_available():
        return [], {"skipped": "tesseract_not_installed"}

    import pytesseract

    langs = str(cfg.get("langs") or "eng")
    min_conf = float(cfg.get("min_confidence") or 45)
    ev: list[Evidence] = []
    stats = {"engine": "tesseract", "frames_scanned": 0, "hits": 0}

    for f in frames:
        img = cv2.imread(f.path)
        if img is None:
            continue
        stats["frames_scanned"] += 1
        try:
            data = pytesseract.image_to_data(_prep(img), lang=langs, output_type=pytesseract.Output.DICT)
        except Exception:
            # 语言包缺失时退回 eng
            try:
                data = pytesseract.image_to_data(_prep(img), lang="eng", output_type=pytesseract.Output.DICT)
            except Exception:
                return ev, {**stats, "error": "ocr_failed"}

        words: list[str] = []
        confs: list[float] = []
        for i, txt in enumerate(data.get("text", [])):
            try:
                c = float(data["conf"][i])
            except (TypeError, ValueError):
                c = -1
            if c < min_conf or not str(txt).strip():
                continue
            words.append(str(txt).strip())
            confs.append(c / 100.0)

        text = squash(" ".join(words))
        f.ocr_text = text
        if not _keep(text):
            continue
        stats["hits"] += 1
        conf = float(np.mean(confs)) if confs else 0.5
        ev.append(Evidence(
            kind="ocr", source="tesseract", slot="text_sign",
            claim=f"OCR 识别到文字：{clip(text, 160)}", value=clip(text, 300),
            confidence=round(min(0.85, 0.35 + conf * 0.5), 3),
            timestamp_s=f.timestamp_s, frame_id=f.id, raw={"words": words[:60]},
        ))

    # 汇总去重
    all_text = squash(" | ".join(e.value for e in ev))
    if all_text:
        stats["aggregate"] = clip(all_text, 800)
    return ev, stats
