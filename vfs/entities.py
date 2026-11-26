"""把证据里的自由文本归一化成可比对的值。"""

from __future__ import annotations

import re
from typing import Any

from .utils import norm_key, squash

# --------------------------------------------------------------------------- #
# 机型
# --------------------------------------------------------------------------- #

_AIRBUS = re.compile(r"\bA\s?-?\s*(3\d{2})\s*-?\s*(neo|ceo|xlr|n)\b|\bA\s?-?\s*(3\d{2})\b", re.I)
_BOEING = re.compile(
    r"\b(?:Boeing\s*|B\s?-?\s*)?(7\d{2})\s*-?\s*(MAX\s*\d{1,2}|MAX|(\d{2,3})(?:ER|LR|ULR)?)\b"
    r"|\b(?:Boeing\s*|B\s?-?\s*)?(7\d{2})\b",
    re.I,
)

TYPE_ALIASES = {
    "a320": "A320", "a321": "A321", "a319": "A319", "a318": "A318",
    "a330": "A330", "a350": "A350", "a380": "A380", "a220": "A220",
    "b737": "B737", "b738": "B737-800", "b739": "B737-900", "b38m": "B737 MAX 8",
    "b747": "B747", "b757": "B757", "b767": "B767", "b777": "B777", "b787": "B787",
    "e190": "E190", "e195": "E195", "crj900": "CRJ900", "atr72": "ATR72",
}

BODY_HINTS = {
    "wide": ["宽体", "双通道", "wide-body", "widebody", "双过道"],
    "narrow": ["窄体", "单通道", "narrow-body", "narrowbody", "单过道"],
}


def normalize_aircraft_type(text: str) -> dict[str, Any]:
    """把各种说法归一：返回 {key, family, body, manufacturer, display}。key 为空表示无法归一。"""
    raw = squash(text or "")
    t = raw.upper()
    body = ""
    for b, words in BODY_HINTS.items():
        if any(w.upper() in t or w in raw for w in words):
            body = b
            break

    key = ""
    display = raw

    # ICAO 机型代码
    compact = re.sub(r"[^A-Z0-9]", "", t)
    if compact[:3] in ("A20", "A21", "A19", "A33", "A35", "A38"):
        key = compact[:3] + compact[3:4] if len(compact) > 3 else compact
    if compact in TYPE_ALIASES:
        key = TYPE_ALIASES[compact]

    if not key:
        m = _AIRBUS.search(t)
        if m:
            num = m.group(1) or m.group(3) or ""
            variant = (m.group(2) or "").lower()
            key = f"A{num}"
            if variant in ("neo", "n"):
                key += "neo"
            display = _airbus_display(key)
    if not key:
        m = _BOEING.search(t)
        if m:
            num = m.group(1) or m.group(4) or ""
            variant = (m.group(2) or "").upper().strip()
            sub = m.group(3) or ""
            key = f"B{num}"
            if variant.startswith("MAX"):
                # "737 MAX 8" → B737 MAX 8；只说 "MAX" 时保持 B737 MAX
                mx = re.search(r"MAX\s*(\d{1,2})", variant)
                key = f"B{num} MAX" + (f" {mx.group(1)}" if mx else "")
            elif sub:
                key = _boeing_subtype(num, sub)
            display = _boeing_display(key)

    # family 用于跨源比对：去掉 -800 / MAX 8 的"8"这类子型号编号。
    # 但世代标识必须保留（A320neo / B737 MAX），因为它们代表不同的机型与航程。
    family = key
    if not family.endswith("neo"):
        family = re.sub(r"\s+MAX\s*\d{1,2}$", " MAX", family)   # 737 MAX 8 → 737 MAX
        family = re.sub(r"-\d{2,3}$", "", family)                # 737-800 → 737

    manufacturer = ""
    if key.startswith("A") and not key.startswith("ATR"):
        manufacturer = "Airbus"
    elif key.startswith("B"):
        manufacturer = "Boeing"
    elif key.startswith("E"):
        manufacturer = "Embraer"
    elif key.startswith("CRJ"):
        manufacturer = "Bombardier"
    elif key.startswith("ATR"):
        manufacturer = "ATR"

    if not body and key:
        if family in {"A330", "A350", "A380", "B747", "B767", "B777", "B787"}:
            body = "wide"
        elif family in {"A318", "A319", "A320", "A321", "A220", "B737", "B757", "E190", "E195", "CRJ900", "ATR72"}:
            body = "narrow"

    return {"key": key, "family": family, "body": body, "manufacturer": manufacturer,
            "display": display if key else raw, "raw": raw}


def _airbus_display(key: str) -> str:
    m = re.match(r"A(\d{3})(neo|ceo)?", key)
    if not m:
        return key
    return f"Airbus A{m.group(1)}{(' ' + m.group(2).upper()) if m.group(2) else ''}"


def _boeing_model_range(key: str) -> tuple[int, int] | None:
    """返回 (最小航程km, 最大航程km)，用于航线可行性校验。"""
    num = key[1:4]
    table = {
        "737": (3000, 5800), "747": (11000, 14800), "757": (4500, 7200),
        "767": (5500, 11000), "777": (9000, 17400), "787": (11000, 15700),
    }
    if key.startswith("B737 MAX"):
        return (5500, 6500)
    return table.get(num)


def _boeing_subtype(num: str, sub: str) -> str:
    """把 737+800 / 777+300 / 787+9 拼成规范子型号。"""
    if len(sub) == 2:
        # 738 风格的两位码：737 → -800，777 → -300
        return f"B{num}-{sub}0"
    # 787 的 -8/-9/-10 是系列编号而不是 x00 形式
    if num == "787" and len(sub) <= 2:
        return f"B{num}-{int(sub)}"
    return f"B{num}-{sub}"


def _boeing_display(key: str) -> str:
    m = re.match(r"B(\d{3})\s*(MAX\s*\d{1,2}|MAX)?(?:\s*-(\d{1,3}))?", key)
    if not m:
        return key
    num, mx, suf = m.group(1), m.group(2), m.group(3)
    if mx:
        return f"Boeing {num} {mx.replace('MAX  ', 'MAX ')}"
    if suf:
        return f"Boeing {num}-{suf}"
    return f"Boeing {num}"


# --------------------------------------------------------------------------- #
# 航班号 / 注册号 / 机场码
# --------------------------------------------------------------------------- #

FLIGHT_RE = re.compile(r"\b([A-Z][A-Z0-9])\s?-?\s?(\d{1,4})(?![0-9])\b")


def normalize_flight(text: str) -> dict[str, str] | None:
    t = squash(text or "").upper()
    if not t:
        return None
    # 3 字母 ICAO
    m3 = re.search(r"\b([A-Z]{3})(\d{1,4})\b", t)
    # 2 字母 IATA
    m2 = re.search(r"\b([A-Z]{2})\s?-?\s?(\d{1,4})\b", t)
    if m2:
        num = m2.group(2)
        return {"iata": f"{m2.group(1)}{num}", "icao": f"{m3.group(1)}{num}" if m3 else "",
                "prefix_iata": m2.group(1), "number": num, "raw": t}
    if m3:
        return {"iata": "", "icao": f"{m3.group(1)}{m3.group(2)}", "prefix_iata": "",
                "prefix_icao": m3.group(1), "number": m3.group(2), "raw": t}
    return None


# 顺序很重要：带横杠的长格式必须排在短格式前面，否则 JA01AJ 会被 N\d 之类的分支抢先匹配
REG_RE = re.compile(
    r"("
    r"9V-[A-Z]{3}"                # 新加坡
    r"|VT-[A-Z]{3}"               # 印度
    r"|A6-[A-Z]{3}"               # 阿联酋
    r"|B-\d{4}|B-[A-Z0-9]{4}"     # 中国
    r"|[A-Z]-[A-Z]{4}"            # 德国等 D-AIBX
    r"|JA\d{2,4}[A-Z]{0,2}"       # 日本（JA8088 / JA01AJ）
    r"|HL\d{4}"                   # 韩国
    r"|N\d{1,5}[A-Z]{0,2}"        # 美国
    r"|B[0-9A-Z]{4}"              # 中国（无横杠 B1672）
    r")"
)


def normalize_registration(text: str) -> str | None:
    t = squash(text or "").upper()
    if not t:
        return None
    t = re.sub(r"\s+", "", t)
    m = REG_RE.search(t)
    if not m:
        return None
    reg = m.group(1)
    # B1672 → B-1672；但别把 9V/4X 这类国家前缀也改掉
    if re.fullmatch(r"B[0-9A-Z]{4}", reg):
        reg = "B-" + reg[1:]
    return reg


AIRPORT_CODE = re.compile(r"\b([A-Z]{3,4})\b")


def airport_codes(text: str) -> list[str]:
    t = squash(text or "").upper()
    codes = []
    # 括号里的码
    for m in re.finditer(r"\(([A-Z]{3,4})(?:/([A-Z]{4}))?\)", t):
        for g in m.groups():
            if g:
                codes.append(g)
    for m in AIRPORT_CODE.finditer(t):
        c = m.group(1)
        if len(c) == 3 or len(c) == 4:
            codes.append(c)
    return list(dict.fromkeys(codes))


# --------------------------------------------------------------------------- #
# 日期 / 时间
# --------------------------------------------------------------------------- #

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

DATE_PATTERNS = [
    re.compile(r"(20\d{2})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})"),
    re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月"),
    re.compile(r"(20\d{2})\s*[-/]\s*(\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(20\d{2})"),
]


def normalize_date(text: str) -> str | None:
    t = squash(text or "")
    if not t:
        return None
    for i, pat in enumerate(DATE_PATTERNS):
        m = pat.search(t)
        if not m:
            continue
        groups = m.groups()
        try:
            if i == 3:
                mo, d, y = int(groups[0]), int(groups[1]), int(groups[2])
            else:
                y = int(groups[0])
                mo = int(groups[1])
                d = int(groups[2]) if len(groups) > 2 and groups[2] else 0
            if 1 <= mo <= 12:
                return f"{y:04d}-{mo:02d}-{d:02d}" if d else f"{y:04d}-{mo:02d}"
        except (TypeError, ValueError):
            continue
    low = t.lower()
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", low):
            ym = re.search(r"(20\d{2})", t)
            if ym:
                return f"{ym.group(1)}-{num:02d}"
    ym = re.search(r"\b(19[89]\d|20[0-4]\d)\b", t)
    if ym:
        return ym.group(1)
    return None


TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:：]([0-5]\d)(?:[:：]([0-5]\d))?\b")


def normalize_time(text: str) -> str | None:
    t = squash(text or "")
    if not t:
        return None
    m = TIME_RE.search(t)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    low = t.lower()
    mapping = [
        (["night", "夜晚", "夜间", "深夜", "晚上"], "night"),
        (["sunrise", "日出", "清晨", "早晨"], "sunrise"),
        (["sunset", "dusk", "黄昏", "日落", "傍晚"], "sunset"),
        (["noon", "中午", "正午"], "noon"),
        (["afternoon", "下午"], "afternoon"),
        (["morning", "上午", "早上"], "morning"),
        (["day", "白天", "日间"], "daytime"),
    ]
    for words, val in mapping:
        if any(w in low for w in words):
            return val
    return None


# --------------------------------------------------------------------------- #
# 坐标
# --------------------------------------------------------------------------- #

COORD_PAIR = re.compile(
    r"([-+]?\d{1,3}\.\d{2,6})\s*[,\s]\s*([-+]?\d{1,3}\.\d{2,6})"
)
COORD_DMS = re.compile(
    r"(\d{1,3})[°d]\s*(\d{1,2})?['′]?\s*(\d{1,2}(?:\.\d+)?)?[\"″]?\s*([NS])"
    r"[\s,]*"
    r"(\d{1,3})[°d]\s*(\d{1,2})?['′]?\s*(\d{1,2}(?:\.\d+)?)?[\"″]?\s*([EW])",
    re.I,
)


def normalize_coordinate(text: str) -> tuple[float, float] | None:
    t = squash(text or "")
    if not t:
        return None
    m = COORD_DMS.search(t)
    if m:
        lat = _dms(m.group(1), m.group(2), m.group(3), m.group(4))
        lon = _dms(m.group(5), m.group(6), m.group(7), m.group(8))
        if lat is not None and lon is not None:
            return lat, lon
    # 简易格式 "33.9S 151.2E"
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*([NS])[\s,]+(\d{1,3}(?:\.\d+)?)\s*([EW])", t, re.I)
    if m:
        lat = float(m.group(1)) * (-1 if m.group(2).upper() == "S" else 1)
        lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
        return lat, lon
    m = COORD_PAIR.search(t)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
        if abs(a) <= 90 and abs(b) <= 180:
            return a, b
        if abs(b) <= 90 and abs(a) <= 180:
            return b, a
    return None


def _dms(d: str, m: str | None, s: str | None, hemi: str) -> float | None:
    try:
        val = float(d) + (float(m) / 60 if m else 0) + (float(s) / 3600 if s else 0)
    except (TypeError, ValueError):
        return None
    if hemi.upper() in ("S", "W"):
        val = -val
    return val


# --------------------------------------------------------------------------- #
# 通用
# --------------------------------------------------------------------------- #


def canonical(slot: str, value: str) -> dict[str, Any]:
    """按 slot 归一，返回 {key, display, extra}。key 用于去重比较。"""
    v = squash(value or "")
    if slot == "aircraft_type":
        n = normalize_aircraft_type(v)
        return {"key": n["family"] or n["key"] or norm_key(v), "display": n["display"],
                "extra": n}
    if slot in ("flight_number",):
        n = normalize_flight(v)
        if n:
            return {"key": n["iata"] or n["icao"], "display": n["iata"] or n["icao"], "extra": n}
    if slot == "registration":
        reg = normalize_registration(v)
        if reg:
            return {"key": norm_key(reg), "display": reg, "extra": {"registration": reg}}
    if slot in ("airport", "origin", "destination"):
        codes = airport_codes(v)
        return {"key": norm_key(v), "display": v, "extra": {"codes": codes}}
    if slot == "date":
        d = normalize_date(v)
        if d:
            return {"key": d, "display": d, "extra": {"date": d}}
    if slot == "time_of_day":
        t = normalize_time(v)
        if t:
            return {"key": t, "display": t, "extra": {"time": t}}
    if slot == "coordinate":
        c = normalize_coordinate(v)
        if c:
            return {"key": f"{c[0]:.2f},{c[1]:.2f}", "display": f"{c[0]:.4f}, {c[1]:.4f}",
                    "extra": {"lat": c[0], "lon": c[1]}}
    if slot in ("airline", "city", "country", "landmark", "object", "brand", "text_sign",
                "weather", "activity", "person_role"):
        # 去掉常见润色词
        cleaned = re.sub(r"^(a|an|the|likely|possibly|probably|可能|似乎是|看起来是)\s+", "", v, flags=re.I)
        return {"key": norm_key(cleaned), "display": cleaned, "extra": {"raw": v}}
    return {"key": norm_key(v), "display": v, "extra": {}}
