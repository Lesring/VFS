"""航班 / 航空器知识层。

数据源：
- adsbdb（免 key）：按呼号查航线（起降机场，含机场经纬度）；按注册号查机型与机主。
- OpenFlights airlines.dat（免 key）：IATA/ICAO 航司码 ↔ 名称。
- OpenSky（免 key，有速率限制）：按区域查实时航迹，用于反推"画面里的飞机在飞哪条航路"。
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from ..utils import clip, norm_key, similarity, squash
from .http import CachedClient

ADSBDB = "https://api.adsbdb.com/v0"
AIRLINES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"

_CALLSIGN_RE = re.compile(r"\b([A-Z]{2,3}\d{1,4}[A-Z]?)\b")


# --------------------------------------------------------------------------- #
# 航司表
# --------------------------------------------------------------------------- #


class AirlineDB:
    def __init__(self, rows: list[list[str]]):
        self.rows: list[dict[str, str]] = []
        self.by_code: dict[str, dict[str, str]] = {}
        self.by_name_key: dict[str, dict[str, str]] = {}
        for r in rows:
            if len(r) < 5:
                continue
            rec = {
                "name": _clean(r[1]),
                "alias": _clean(r[2]),
                "iata": _clean(r[3]).upper(),
                "icao": _clean(r[4]).upper(),
                "callsign": _clean(r[5]) if len(r) > 5 else "",
                "country": _clean(r[6]) if len(r) > 6 else "",
                "active": _clean(r[7]) if len(r) > 7 else "",
            }
            if not rec["name"]:
                continue
            self.rows.append(rec)
            for c in (rec["iata"], rec["icao"]):
                if c and len(c) <= 4:
                    self.by_code.setdefault(c, rec)
            for nm in (rec["name"], rec["alias"]):
                k = norm_key(nm)
                if k:
                    self.by_name_key.setdefault(k, rec)

    def __len__(self) -> int:
        return len(self.rows)

    def lookup(self, text: str) -> list[dict[str, str]]:
        if not text:
            return []
        t = text.strip()
        up = t.upper().replace(" ", "")
        hits: list[dict[str, str]] = []
        if up in self.by_code:
            hits.append(self.by_code[up])
        m = _CALLSIGN_RE.search(t.upper())
        if m:
            prefix = m.group(1)
            for L in (3, 2):
                sub = prefix[:L]
                if sub.isalpha() and sub in self.by_code:
                    rec = self.by_code[sub]
                    if rec not in hits:
                        hits.append(rec)
                    break
        key = norm_key(t)
        if key in self.by_name_key and self.by_name_key[key] not in hits:
            hits.append(self.by_name_key[key])
        if hits:
            return hits[:5]
        # 包含 / 模糊
        for k, rec in self.by_name_key.items():
            if len(k) >= 5 and (key in k or k in key):
                hits.append(rec)
        if hits:
            return _dedup(hits)[:5]
        scored = [(similarity(key, k), rec) for k, rec in self.by_name_key.items()
                  if abs(len(k) - len(key)) <= max(6, len(key) // 2)]
        scored.sort(key=lambda x: -x[0])
        return _dedup([rec for s, rec in scored if s >= 0.74])[:5]

    def by_callsign_prefix(self, cs: str) -> dict[str, str] | None:
        up = (cs or "").upper()
        for L in (3, 2):
            rec = self.by_code.get(up[:L])
            if rec:
                return rec
        return None


def _dedup(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen, out = set(), []
    for r in rows:
        k = (r.get("icao") or r.get("iata") or r.get("name", "")).upper()
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _clean(v: str) -> str:
    return "" if v in ("\\N", "NULL", "-") else squash(v)


AIRLINE_CACHE = "openflights_airlines.dat"


async def load_airlines(client: CachedClient, cache_dir: str | Path) -> AirlineDB:
    cache_path = Path(cache_dir) / AIRLINE_CACHE
    if not cache_path.exists():
        await client.download_binary(AIRLINES_URL, cache_path, ttl_s=30 * 86400)
    rows: list[list[str]] = []
    if cache_path.exists():
        try:
            reader = csv.reader(io.StringIO(cache_path.read_text(encoding="utf-8", errors="ignore")))
            rows = [r for r in reader if r]
        except Exception:
            rows = []
    return AirlineDB(rows)


# --------------------------------------------------------------------------- #
# adsbdb
# --------------------------------------------------------------------------- #


async def route_by_callsign(client: CachedClient, callsign: str) -> dict[str, Any] | None:
    """查呼号对应的起降机场。返回 {callsign, airline, origin, destination} 或 None。"""
    cs = re.sub(r"\s+", "", (callsign or "").upper())
    if not cs:
        return None
    data = await client.get_json(f"{ADSBDB}/callsign/{cs}")
    if not isinstance(data, dict):
        return None
    fr = ((data.get("response") or {}).get("flightroute")) or {}
    if not fr:
        return None
    origin = fr.get("origin") or {}
    dest = fr.get("destination") or {}
    if not origin or not dest:
        return None
    return {
        "callsign": fr.get("callsign") or cs,
        "callsign_iata": fr.get("callsign_iata") or "",
        "callsign_icao": fr.get("callsign_icao") or cs,
        "airline": fr.get("airline") or {},
        "origin": origin,
        "destination": dest,
        "source": "adsbdb",
    }


async def aircraft_by_registration(client: CachedClient, reg: str) -> dict[str, Any] | None:
    r = re.sub(r"[^A-Za-z0-9-]", "", (reg or "").upper())
    if not r:
        return None
    data = await client.get_json(f"{ADSBDB}/aircraft/{r}")
    if not isinstance(data, dict):
        return None
    ac = ((data.get("response") or {}).get("aircraft")) or {}
    if not ac:
        return None
    return {
        "registration": ac.get("registration") or r,
        "type": ac.get("type") or "",
        "icao_type": ac.get("icao_type") or "",
        "manufacturer": ac.get("manufacturer") or "",
        "owner": ac.get("registered_owner") or "",
        "owner_country": ac.get("registered_owner_country_name") or "",
        "owner_flag": ac.get("registered_owner_operator_flag_code") or "",
        "mode_s": ac.get("mode_s") or "",
        "source": "adsbdb",
    }


# --------------------------------------------------------------------------- #
# OpenSky：区域实时航迹
# --------------------------------------------------------------------------- #


async def state_vectors(client: CachedClient, lat: float, lon: float, radius_deg: float = 1.2) -> list[dict[str, Any]]:
    params = {
        "lamin": round(max(-90, lat - radius_deg), 4),
        "lomin": round(max(-180, lon - radius_deg), 4),
        "lamax": round(min(90, lat + radius_deg), 4),
        "lomax": round(min(180, lon + radius_deg), 4),
    }
    data = await client.get_json("https://opensky-network.org/api/states/all", params=params)
    if not isinstance(data, dict):
        return []
    out = []
    for s in data.get("states") or []:
        if not s or len(s) < 11:
            continue
        out.append({
            "icao24": s[0], "callsign": (s[1] or "").strip(),
            "country": s[2], "lon": s[5], "lat": s[6],
            "baro_alt": s[7], "on_ground": s[8], "velocity": s[9],
            "true_track": s[10], "vertical_rate": s[11] if len(s) > 11 else None,
        })
    return out


# --------------------------------------------------------------------------- #
# 机型归一化：把视觉模型的各种说法收敛到统一族
# --------------------------------------------------------------------------- #

FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("A320neo", re.compile(r"\bA320\s*NEO\b|\bA320N\b", re.I)),
    ("A321neo", re.compile(r"\bA321\s*NEO\b|\bA321N\b", re.I)),
    ("A319neo", re.compile(r"\bA319\s*NEO\b", re.I)),
    ("A320", re.compile(r"\bA-?320\b|\bA320\b", re.I)),
    ("A321", re.compile(r"\bA-?321\b|\bA321\b", re.I)),
    ("A319", re.compile(r"\bA-?319\b|\bA319\b", re.I)),
    ("A318", re.compile(r"\bA-?318\b|\bA318\b", re.I)),
    ("A330", re.compile(r"\bA-?330\b", re.I)),
    ("A350", re.compile(r"\bA-?350\b", re.I)),
    ("A380", re.compile(r"\bA-?380\b", re.I)),
    ("B737 MAX", re.compile(r"\b737\s*-?\s*(MAX|8|9|10)\b(?!.*\b8\d\d\b)", re.I)),
    ("B737", re.compile(r"\b737(?:-\d{2,3})?\b|\bB-?737\b", re.I)),
    ("B747", re.compile(r"\b747(?:-\d{2,3})?\b", re.I)),
    ("B757", re.compile(r"\b757\b", re.I)),
    ("B767", re.compile(r"\b767\b", re.I)),
    ("B777", re.compile(r"\b777(?:-\d{3})?\b", re.I)),
    ("B787", re.compile(r"\b787(?:-\d)?\b", re.I)),
    ("E190", re.compile(r"\bE-?1(?:90|95)\b|\bERJ-?190\b", re.I)),
    ("CRJ", re.compile(r"\bCRJ-?\d{3}\b", re.I)),
    ("ATR72", re.compile(r"\bATR\s*-?\s*72\b", re.I)),
    ("A220", re.compile(r"\bA-?220\b|\bCS300\b|\bBCS3\b", re.I)),
]

WIDEBODY = {"A330", "A350", "A380", "B747", "B767", "B777", "B787"}
NARROWBODY = {"A318", "A319", "A320", "A321", "A220", "B737", "B737 MAX", "B757", "E190", "CRJ", "ATR72"}


def normalize_aircraft(text: str) -> dict[str, Any]:
    """返回 {family, manufacturer, body, span}。"""
    t = squash(text or "")
    family = ""
    for name, pat in FAMILY_PATTERNS:
        if pat.search(t):
            family = name
            break
    manufacturer = ""
    if family:
        manufacturer = "Airbus" if family.startswith("A") and not family.startswith("ATR") else (
            "Boeing" if family.startswith("B") else (
                "Embraer" if family.startswith("E") else (
                    "Bombardier" if family.startswith("CRJ") else (
                        "ATR" if family.startswith("ATR") else ""))))
    body = "wide" if family in WIDEBODY else ("narrow" if family in NARROWBODY else "")
    if "wide" in t.lower() or "双通道" in t or "宽体" in t:
        body = body or "wide"
    if "narrow" in t.lower() or "单通道" in t or "窄体" in t:
        body = body or "narrow"
    return {"family": family, "manufacturer": manufacturer, "body": body, "raw": t}


# --------------------------------------------------------------------------- #
# 航班号归一化
# --------------------------------------------------------------------------- #

_FLIGHT_RE = re.compile(r"\b([A-Z]{2})\s*-?\s*(\d{1,4}[A-Z]?)\b")


def normalize_flight_number(text: str, airline_db: AirlineDB | None = None) -> dict[str, Any] | None:
    t = squash(text or "").upper()
    if not t:
        return None
    # 去掉 ICAO 前缀的情况：CCA981 -> CA981
    m = re.search(r"\b([A-Z]{3})\s*-?\s*(\d{1,4})\b", t)
    if m and airline_db:
        rec = airline_db.by_code.get(m.group(1))
        if rec and rec.get("iata"):
            return {"iata": f"{rec['iata']}{m.group(2)}", "icao": f"{m.group(1)}{m.group(2)}",
                    "airline_icao": m.group(1), "airline_iata": rec["iata"], "raw": t}
    m = _FLIGHT_RE.search(t)
    if m:
        iata = f"{m.group(1)}{m.group(2)}"
        icao = ""
        ai_iata = m.group(1)
        if airline_db:
            for rec in airline_db.rows:
                if rec.get("iata") == ai_iata and rec.get("icao"):
                    icao = f"{rec['icao']}{m.group(2)}"
                    break
        return {"iata": iata, "icao": icao, "airline_iata": ai_iata, "airline_icao": icao[:3] if icao else "", "raw": t}
    return None


_REG_RE = re.compile(r"\b([A-Z]{1,2}-?[A-Z0-9]{3,5})\b")


def normalize_registration(text: str) -> str | None:
    t = squash(text or "").upper()
    if not t:
        return None
    # 常见格式：B-1234 / N12345 / D-AIBX / JA01AJ / 9V-SKA
    m = re.search(r"\b(B-\d{4}|B-[A-Z0-9]{4}|N\d{3,5}[A-Z]{0,2}|[A-Z]-[A-Z]{4}|[A-Z]{2}-?[A-Z]{3,4}|JA\d{3}[A-Z])\b", t)
    if m:
        return m.group(1).replace("--", "-")
    return None
