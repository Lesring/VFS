from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# 文本工具
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def squash(text: str) -> str:
    """压缩空白，去掉零宽字符。"""
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return _WS.sub(" ", text).strip()


def norm_key(text: str) -> str:
    """归一化成用于去重/比较的 key。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).upper()
    return _NON_ALNUM.sub("", text)


def clip(text: str, n: int) -> str:
    text = squash(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def uniq(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = norm_key(it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def token_set(text: str) -> set[str]:
    return {t for t in re.split(r"[\s,;/、，。()（）|]+", (text or "").upper()) if t}


def similarity(a: str, b: str) -> float:
    """粗糙的相似度：token 交集 + 字符 Jaccard，用于候选合并。"""
    a, b = norm_key(a), norm_key(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.86
    ga = {a[i : i + 2] for i in range(len(a) - 1)}
    gb = {b[i : i + 2] for i in range(len(b) - 1)}
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


# --------------------------------------------------------------------------- #
# 概率 / 打分
# --------------------------------------------------------------------------- #


def noisy_or(pairs: Iterable[tuple[float, float]]) -> float:
    """noisy-OR 合成：pairs 为 (weight, confidence)。"""
    miss = 1.0
    for w, c in pairs:
        p = max(0.0, min(1.0, w * c))
        miss *= 1.0 - p
    return 1.0 - miss


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def fmt_pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def confidence_label(p: float, low: float = 0.35, high: float = 0.7) -> str:
    if p >= high:
        return "高"
    if p >= low:
        return "中"
    if p > 0.05:
        return "低"
    return "无法判定"


# --------------------------------------------------------------------------- #
# 文件工具
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def parse_dms(text: str) -> float | None:
    """解析 "39°54'27\"N" 形式坐标。"""
    if not text:
        return None
    t = text.strip().upper()
    neg = bool(re.search(r"[SW]", t))
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    if not nums:
        return None
    vals = [float(x) for x in nums[:3]]
    deg = vals[0] + (vals[1] / 60 if len(vals) > 1 else 0) + (vals[2] / 3600 if len(vals) > 2 else 0)
    return -deg if neg else deg
