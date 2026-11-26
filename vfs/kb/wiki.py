"""Wikipedia / Wikidata 查询（走 api.wikimedia.org，本机可达）。"""

from __future__ import annotations

import html as htmllib
import re
from typing import Any
from urllib.parse import quote

from ..utils import clip, squash
from .http import CachedClient

CORE = "https://api.wikimedia.org/core/v1/wikipedia"
WD_API = "https://www.wikidata.org/w/api.php"

_TAG = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    return squash(htmllib.unescape(_TAG.sub(" ", s or "")))


async def search(client: CachedClient, query: str, limit: int = 3, lang: str = "en") -> list[dict[str, Any]]:
    data = await client.get_json(
        f"{CORE}/{lang}/search/page",
        params={"q": query, "limit": limit},
    )
    if not isinstance(data, dict):
        return []
    out = []
    for p in (data.get("pages") or [])[:limit]:
        out.append({
            "title": p.get("title") or p.get("key") or "",
            "key": p.get("key") or "",
            "description": _clean(p.get("description") or ""),
            "snippet": clip(_clean(p.get("excerpt") or ""), 400),
            "url": f"https://{lang}.wikipedia.org/wiki/{quote(p.get('key') or '')}",
        })
    return out


async def page_summary(client: CachedClient, title: str, lang: str = "en") -> dict[str, Any] | None:
    key = quote(title.replace(" ", "_"), safe="")
    data = await client.get_json(f"{CORE}/{lang}/page/{key}")
    if not isinstance(data, dict):
        return None
    extract = data.get("extract") or ""
    return {
        "title": data.get("title") or title,
        "description": _clean(data.get("description") or ""),
        "extract": squash(extract)[:1800],
        "url": (data.get("content_urls") or {}).get("desktop", {}).get("page")
               or f"https://{lang}.wikipedia.org/wiki/{key}",
        "lang": lang,
    }


async def wikidata_search(client: CachedClient, query: str, limit: int = 3) -> list[dict[str, Any]]:
    data = await client.get_json(
        WD_API,
        params={
            "action": "wbsearchentities", "search": query,
            "language": "en", "uselang": "en", "format": "json", "limit": limit,
        },
    )
    if not isinstance(data, dict):
        return []
    out = []
    for s in (data.get("search") or [])[:limit]:
        out.append({
            "id": s.get("id"),
            "label": s.get("label") or "",
            "description": s.get("description") or "",
            "url": f"https://www.wikidata.org/wiki/{s.get('id')}",
        })
    return out


async def ask(client: CachedClient, query: str, limit: int = 3, langs: tuple[str, ...] = ("en", "zh")) -> list[dict[str, Any]]:
    """给一个自由文本问题，返回一组可用于佐证的条目摘要。"""
    hits: list[dict[str, Any]] = []
    for lang in langs:
        try:
            pages = await search(client, query, limit=2, lang=lang)
        except Exception:
            pages = []
        for p in pages:
            if not p.get("key"):
                continue
            summ = await page_summary(client, p["key"], lang=lang)
            if summ:
                hits.append({**summ, "snippet": p.get("snippet", ""), "source": f"wikipedia:{lang}"})
            if len(hits) >= limit:
                return hits
    return hits
