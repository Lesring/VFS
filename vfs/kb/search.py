"""网页搜索后端。本机可达的是 Bing / 百度 / 搜狗；也支持 Tavily / Brave / Serper（有 key 时用）。

设计要点：
- 不依赖 BeautifulSoup，纯正则 + 容错，抓不到就返回空列表而不是抛异常。
- 所有请求都过 CachedClient，便于离线复现。
"""

from __future__ import annotations

import base64
import html as htmllib
import os
import re
from typing import Any
from urllib.parse import quote_plus, unquote

from ..utils import clip, squash
from .http import CachedClient

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)


def _strip_tags(fragment: str) -> str:
    fragment = _SCRIPT.sub(" ", fragment or "")
    text = _TAG.sub(" ", fragment)
    text = htmllib.unescape(text)
    return squash(text)


def _ddg_redirect(url: str) -> str:
    """解出 bing/百度/搜狗的跳转链接。"""
    m = re.search(r"[?&]url=([^&]+)", url)
    if m:
        return unquote(m.group(1))
    m = re.search(r"[?&]u=([^&]+)", url)
    if m and "baidu.com" in url:
        return unquote(m.group(1))
    if url.startswith("//"):
        return "https:" + url
    return url


def _decode_bing_redirect(href: str) -> str:
    """解 Bing 的 /ck/a 跳转。

    Bing 把真实地址放在 u=a1<base64url> 里（前缀 a1 表示已编码）。
    不解码的话所有结果都会被当成站内跳转丢掉——这是最容易踩的坑。
    """
    if not href or "/ck/a" not in href:
        return href
    href = htmllib.unescape(href)
    m = re.search(r"[?&]u=([^&]+)", href)
    if not m:
        return ""
    token = unquote(m.group(1))
    if not token.startswith("a1"):
        return ""
    payload = token[2:]
    # base64url → base64，补齐 padding
    payload = payload.replace("-", "+").replace("_", "/")
    payload += "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload)
        url = raw.decode("utf-8", "ignore").strip()
    except Exception:
        return ""
    if url.startswith("http"):
        return url
    return ""


class SearchResult(dict):
    pass


def _mk(title: str, url: str, snippet: str, engine: str) -> dict[str, Any]:
    return {
        "title": clip(_strip_tags(title), 160),
        "url": _ddg_redirect(url.strip()),
        "snippet": clip(_strip_tags(snippet), 420),
        "engine": engine,
    }


# 检索结果必须与查询有关，否则是噪声。
# 实测：无 cookie 爬 Bing 时，它会返回一批与查询毫不相关的结果（查 Airbus 返 ChatGPT），
# 这类结果如果进了证据池，会严重误导结论。宁可不要，也不能污染。
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was",
    "flight", "flights", "airport", "aircraft", "what", "which", "where", "how",
    "的", "是", "在", "和", "与", "航班", "飞机", "机场",
}


def _query_terms(query: str) -> list[str]:
    """取出查询里的"有效词"，用于判断结果相关性。"""
    raw = re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", (query or "").lower())
    terms = []
    for t in raw:
        if not t or t in _STOPWORDS:
            continue
        # 中文没有空格分词，用 2-gram 兜底
        if re.search(r"[\u4e00-\u9fff]", t) and len(t) > 2:
            terms.append(t)
            terms.extend(t[i:i + 2] for i in range(len(t) - 1))
        else:
            terms.append(t)
    return list(dict.fromkeys(terms))


def is_relevant(result: dict[str, Any], query: str, min_hits: int = 1) -> bool:
    """标题+摘要里至少要命中查询的一个有效词。

    这能挡住"搜索引擎返回泛化结果"的情况。命中过少的直接丢弃。
    """
    terms = _query_terms(query)
    if not terms:
        return True
    blob = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}".lower()
    hits = sum(1 for t in terms if t in blob)
    return hits >= min_hits


def filter_relevant(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    return [r for r in results if is_relevant(r, query)]


# --------------------------------------------------------------------------- #
# Bing
# --------------------------------------------------------------------------- #

_BING_ITEM = re.compile(r'<li class="b_algo"(.*?)</li>', re.S)
_BING_H2 = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
_BING_A = re.compile(r'<a[^>]+href="([^"]+)"', re.S)
_BING_SNIP = re.compile(r'<p[^>]*>(.*?)</p>', re.S)


async def bing_search(client: CachedClient, query: str, limit: int = 5, lang: str = "zh") -> list[dict[str, Any]]:
    """Bing 检索。

    实测（2026-11）：cn.bing.com 会返回一个不含结果的落地页（约 14KB、零个 b_algo），
    所以中文也走 www.bing.com + setlang，结果结构与英文一致。
    """
    market = "zh-CN" if lang.startswith("zh") else "en-US"
    for host in ("www.bing.com", "cn.bing.com"):
        url = (f"https://{host}/search?q={quote_plus(query)}"
               f"&setlang={market}&count={limit + 4}")
        html = await client.get_text(url)
        if not html or 'class="b_algo"' not in html:
            continue
        results = _parse_bing(html, limit)
        if results:
            return results
    return []


def _parse_bing(html: str, limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block in _BING_ITEM.findall(html):
        h2m = _BING_H2.search(block)
        if not h2m:
            continue
        title = _strip_tags(h2m.group(1))
        if not title:
            continue

        # 优先取 <h2> 内的链接；Bing 有时把 cite/url 放在 h2 之后
        href = ""
        am = _BING_A.search(h2m.group(1))
        if am:
            href = am.group(1)
        if not href:
            am2 = _BING_A.search(block)
            if am2:
                href = am2.group(1)
        if not href:
            # 有的条目只给 <cite> 显示域名
            cm = re.search(r"<cite[^>]*>(.*?)</cite>", block, re.S)
            if cm:
                href = _strip_tags(cm.group(1))
        href = href.strip()

        # Bing 所有外链都包在 /ck/a 里，必须解出真实地址
        if "/ck/a" in href:
            href = _decode_bing_redirect(href)
        if not href:
            # 解不出来就退回 cite 里的展示域名
            cm = re.search(r"<cite[^>]*>(.*?)</cite>", block, re.S)
            href = _strip_tags(cm.group(1)) if cm else ""
        if not href or href.startswith("/"):
            continue
        if not href.startswith("http"):
            href = "https://" + href

        snipm = _BING_SNIP.findall(block)
        snippet = " ".join(_strip_tags(s) for s in snipm[:2])
        results.append(_mk(title, href, snippet, "bing"))
        if len(results) >= limit:
            break
    return results


# --------------------------------------------------------------------------- #
# 搜狗
# --------------------------------------------------------------------------- #

_SOGOU_ITEM = re.compile(r'<div class="vrwrap"|<div class="rb"', re.I)
_SOGOU_H3 = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)


async def sogou_search(client: CachedClient, query: str, limit: int = 5) -> list[dict[str, Any]]:
    url = f"https://www.sogou.com/web?query={quote_plus(query)}"
    html = await client.get_text(url)
    if not html:
        return []
    results: list[dict[str, Any]] = []
    for block in re.split(r'<div class="vrwrap"', html)[1:]:
        block = block[:4000]
        h3 = _SOGOU_H3.search(block)
        if not h3:
            continue
        am = re.search(r'<a[^>]+href="([^"]+)"', h3.group(1))
        title = _strip_tags(h3.group(1))
        if not title:
            continue
        href = am.group(1) if am else ""
        body = _strip_tags(re.sub(r"<h3.*?</h3>", "", block, flags=re.S))[:420]
        results.append(_mk(title, href, body, "sogou"))
        if len(results) >= limit:
            break
    return results


# --------------------------------------------------------------------------- #
# 百度
# --------------------------------------------------------------------------- #

async def baidu_search(client: CachedClient, query: str, limit: int = 5) -> list[dict[str, Any]]:
    url = f"https://www.baidu.com/s?wd={quote_plus(query)}&rn={limit + 4}"
    html = await client.get_text(url)
    if not html:
        return []
    results: list[dict[str, Any]] = []
    for block in re.split(r'<div[^>]+class="[^"]*result[^"]*"', html)[1:]:
        block = block[:6000]
        h3 = _SOGOU_H3.search(block)
        if not h3:
            continue
        title = _strip_tags(h3.group(1))
        if not title or len(title) < 3:
            continue
        href = ""
        am = re.search(r'<a[^>]+href="(http[^"]+)"', block)
        if am:
            href = am.group(1)
        body = _strip_tags(re.sub(r"<h3.*?</h3>", "", block[:2500], flags=re.S))
        results.append(_mk(title, href, body, "baidu"))
        if len(results) >= limit:
            break
    return results


# --------------------------------------------------------------------------- #
# 需要 key 的后端
# --------------------------------------------------------------------------- #


async def tavily_search(client: CachedClient, query: str, limit: int = 5) -> list[dict[str, Any]]:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return []
    data = await client.post_json(
        "https://api.tavily.com/search",
        {"api_key": key, "query": query, "max_results": limit,
         "include_answer": True, "search_depth": "basic"},
        cache_key=f"tavily::{query}::{limit}",
    )
    if not data:
        return []
    out = []
    for r in (data.get("results") or [])[:limit]:
        out.append(_mk(r.get("title", ""), r.get("url", ""), r.get("content", ""), "tavily"))
    return out


async def brave_search(client: CachedClient, query: str, limit: int = 5) -> list[dict[str, Any]]:
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return []
    data = await client.get_json(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": limit},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
    )
    if not data:
        return []
    out = []
    for r in ((data.get("web") or {}).get("results") or [])[:limit]:
        out.append(_mk(r.get("title", ""), r.get("url", ""), r.get("description", ""), "brave"))
    return out


async def serper_search(client: CachedClient, query: str, limit: int = 5) -> list[dict[str, Any]]:
    key = os.environ.get("SERPER_API_KEY", "")
    if not key:
        return []
    data = await client.post_json(
        "https://google.serper.dev/search",
        {"q": query, "num": limit},
        headers={"X-API-KEY": key},
        cache_key=f"serper::{query}::{limit}",
    )
    if not data:
        return []
    out = []
    for r in (data.get("organic") or [])[:limit]:
        out.append(_mk(r.get("title", ""), r.get("link", ""), r.get("snippet", ""), "serper"))
    return out


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #


async def web_search(
    client: CachedClient,
    query: str,
    limit: int = 5,
    backend: str = "auto",
    lang: str = "zh",
    strategy: str = "overseas_first",
    google_key: str = "",
    google_cx: str = "",
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """按区域策略与到达性探测选择后端。

    返回 (结果, 实际尝试过的后端, 各后端的失败原因)。
    这是"境外优先、不可达才降级国内"的落地点：
    探测不通的源直接跳过，不会被墙的域名卡住几十秒。
    """
    from . import health

    backend = (backend or "auto").lower()
    if backend == "off":
        return [], [], {}

    notes: dict[str, str] = {}

    if backend != "auto":
        # 用户显式指定后端：尊重它，但仍做一次可达性检查给出提示
        tries = [backend]
        if not health.check(_health_name(backend)):
            notes[backend] = f"{health.SOURCE_HOSTS.get(_health_name(backend), backend)} 不可达"
    else:
        # 自动模式：按区域策略排序 + 可行性过滤
        ordered = health.ordered_sources(strategy)
        tries = [_backend_of(src) for src in ordered]
        tries = [t for t in dict.fromkeys(tries) if t]
        for t in tries:
            if not health.check(_health_name(t)):
                notes[t] = f"{health.SOURCE_HOSTS.get(_health_name(t), t)} 不可达"

    attempted: list[str] = []
    for name in tries:
        # 每轮都复查（可能刚被探测为不可达）
        if not health.check(_health_name(name)):
            continue
        attempted.append(name)
        try:
            if name == "google":
                if google_key and google_cx:
                    res = await _google_search(client, query, limit, google_key, google_cx)
                else:
                    # 没配 Programmable Search 时，退到免 key 的 Google News
                    res = await _google_news(client, query, limit, lang)
                    if not res:
                        notes[name] = "未配置 Google API key/cx，且 News RSS 无结果"
            elif name == "google_news":
                res = await _google_news(client, query, limit, lang)
            elif name == "tavily":
                res = await tavily_search(client, query, limit)
            elif name == "brave":
                res = await brave_search(client, query, limit)
            elif name == "serper":
                res = await serper_search(client, query, limit)
            elif name == "bing":
                res = await bing_search(client, query, limit, lang)
            elif name == "sogou":
                res = await sogou_search(client, query, limit)
            elif name == "baidu":
                res = await baidu_search(client, query, limit)
            elif name == "duckduckgo":
                res = await duckduckgo_search(client, query, limit)
            else:
                res = []
        except Exception as e:
            notes[name] = f"{type(e).__name__}: {str(e)[:80]}"
            res = []
        if res:
            # 相关性过滤：宁可返回空，也不能把不相关的结果当佐证
            kept = filter_relevant(res, query)
            if len(kept) < len(res):
                notes[name] = f"返回 {len(res)} 条，其中 {len(res) - len(kept)} 条与查询无关已剔除"
            if kept:
                return kept, attempted, notes
            notes[name] = notes.get(name) or "结果均与查询无关"
    return [], attempted, notes


def _backend_of(source: str) -> str:
    """region 源名 → 具体后端名。google 一组里，有 key 用 API，没 key 用 News。"""
    mapping = {
        "google": "google",
        "google_api": "google",
        "wikimedia": "wikimedia",
        "duckduckgo": "duckduckgo",
    }
    return mapping.get(source, source)


def _health_name(backend: str) -> str:
    mapping = {
        "google": "google",
        "google_news": "google",
        "bing": "bing",
        "sogou": "sogou",
        "baidu": "baidu",
        "tavily": "tavily",
        "brave": "brave",
        "serper": "serper",
        "duckduckgo": "duckduckgo",
        "wikimedia": "wikimedia",
    }
    return mapping.get(backend, backend)


async def _google_search(client, query, limit, key, cx):
    from . import google as g
    return await g.search(client, query, limit, api_key=key, cx=cx)


async def _google_news(client, query, limit, lang):
    from . import google as g
    return await g.news(client, query, limit, lang=lang)


async def duckduckgo_search(client: CachedClient, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """DuckDuckGo Instant Answer API（免 key，但覆盖面窄，作最末兜底）。"""
    data = await client.get_json(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
    )
    if not isinstance(data, dict):
        return []
    out = []
    if data.get("AbstractText"):
        out.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL") or "",
            "snippet": clip(data["AbstractText"], 400),
            "engine": "duckduckgo",
        })
    for topic in (data.get("RelatedTopics") or [])[:limit]:
        if not isinstance(topic, dict) or not topic.get("Text"):
            continue
        out.append({
            "title": clip(topic.get("Text", ""), 120),
            "url": topic.get("FirstURL") or "",
            "snippet": clip(topic.get("Text", ""), 400),
            "engine": "duckduckgo",
        })
        if len(out) >= limit:
            break
    return out[:limit]
