"""离线机场库。基于 OpenFlights airports.dat（ODbL），首次运行自动下载并缓存。

能力：IATA/ICAO 码、名称、城市、国家互查；按名称模糊匹配；haversine 最近机场。
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from ..utils import norm_key, similarity
from .http import CachedClient

AIRPORTS_URLS = [
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports-extended.dat",
]


class Airport:
    __slots__ = ("id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "type", "source")

    def __init__(self, row: list[str]):
        def g(i: int) -> str:
            v = row[i] if i < len(row) else ""
            return "" if v in ("\\N", "NULL", "-") else v

        self.id = g(0)
        self.name = g(1)
        self.city = g(2)
        self.country = g(3)
        self.iata = g(4).upper()
        self.icao = g(5).upper()
        self.lat = _f(g(6))
        self.lon = _f(g(7))
        self.alt = _f(g(8))
        # OpenFlights: ... Altitude, Timezone, DST, TzDatabase, Type, Source
        self.tz = g(11) or g(9)          # 优先 IANA 时区名，其次数值偏移
        self.dst = g(10)
        typ = g(12)
        # 兼容旧缓存/缺列：若 type 位看起来像时区名，则视为普通机场
        if not typ or "/" in typ:
            typ = "airport"
        self.type = typ
        self.source = g(13)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "city": self.city, "country": self.country,
            "iata": self.iata, "icao": self.icao,
            "lat": self.lat, "lon": self.lon, "tz": self.tz, "type": self.type,
        }

    def label(self) -> str:
        bits = [self.name]
        if self.city and self.city not in self.name:
            bits.append(self.city)
        code = "/".join(c for c in (self.iata, self.icao) if c)
        return f"{', '.join(bits)} ({code})" if code else ", ".join(bits)


def _f(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


class AirportDB:
    def __init__(self, rows: list[list[str]], source: str = ""):
        self.source = source
        self.airports: list[Airport] = [Airport(r) for r in rows if len(r) > 5]
        self.by_code: dict[str, Airport] = {}
        self.by_name_key: dict[str, list[Airport]] = {}
        for a in self.airports:
            for code in (a.iata, a.icao):
                if code:
                    self.by_code.setdefault(code, a)
            for field in (a.name, a.city):
                k = norm_key(field)
                if k:
                    self.by_name_key.setdefault(k, []).append(a)

    def __len__(self) -> int:
        return len(self.airports)

    def lookup(self, text: str) -> list[Airport]:
        """在自由文本里找机场：先精确码，再包含匹配，再模糊。"""
        if not text:
            return []
        t = text.strip()
        up = t.upper()
        hits: list[Airport] = []
        # 直接代码
        if up in self.by_code:
            hits.append(self.by_code[up])
        # 文本中的 3/4 位大写码
        for tok in up.replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ").split():
            if tok in self.by_code and self.by_code[tok] not in hits:
                hits.append(self.by_code[tok])
        if hits:
            return hits[:5]

        key = norm_key(t)
        exact = self.by_name_key.get(key)
        if exact:
            return _uniq_airports(exact)[:5]

        # 包含匹配
        for k, lst in self.by_name_key.items():
            if len(k) >= 5 and (key in k or k in key):
                hits.extend(lst)
        if hits:
            return _uniq_airports(hits)[:5]

        # 模糊
        scored = []
        for k, lst in self.by_name_key.items():
            if abs(len(k) - len(key)) > max(6, len(key) // 2):
                continue
            s = similarity(key, k)
            if s >= 0.72:
                scored.append((s, lst[0]))
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored[:5]]

    def nearest(self, lat: float, lon: float, limit: int = 5, max_km: float = 300.0) -> list[tuple[Airport, float]]:
        out = []
        for a in self.airports:
            if a.lat is None or a.lon is None:
                continue
            d = haversine_km(lat, lon, a.lat, a.lon)
            if d <= max_km:
                out.append((a, d))
        out.sort(key=lambda x: x[1])
        return out[:limit]


def _uniq_airports(lst: list[Airport]) -> list[Airport]:
    seen, out = set(), []
    for a in lst:
        k = a.iata or a.icao or norm_key(a.name)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


_OPENFLIGHTS_CACHE_NAME = "openflights_airports.dat"


async def load_airports(client: CachedClient, cache_dir: str | Path) -> AirportDB:
    cache_path = Path(cache_dir) / _OPENFLIGHTS_CACHE_NAME
    if not cache_path.exists():
        for url in AIRPORTS_URLS:
            got = await client.download_binary(url, cache_path, ttl_s=30 * 86400)
            if got and got.exists() and got.stat().st_size > 100_000:
                break
    rows: list[list[str]] = []
    if cache_path.exists():
        try:
            text = cache_path.read_text(encoding="utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(text))
            for row in reader:
                if row and len(row) >= 6:
                    rows.append(row)
        except Exception:
            rows = []
    return AirportDB(rows, source=str(cache_path))
