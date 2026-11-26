from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from ..utils import ensure_dir, sha1_text, write_json


class CachedClient:
    """带磁盘缓存的 HTTP 客户端。所有外网访问都从这里走，方便离线复现与调试。"""

    def __init__(self, cache_dir: str | Path, ttl_s: int = 86400, timeout_s: float = 12.0,
                 user_agent: str = "VideoForensics/1.0"):
        self.cache_dir = ensure_dir(cache_dir)
        self.ttl = ttl_s
        self.timeout = timeout_s
        self.ua = user_agent
        self.stats: dict[str, int] = {"hit": 0, "miss": 0, "network": 0, "error": 0}

    def _path(self, key: str, suffix: str) -> Path:
        return self.cache_dir / f"{sha1_text(key)}{suffix}"

    def _read_cache(self, key: str, suffix: str) -> Any | None:
        p = self._path(key, suffix)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            if suffix == ".json":
                return json.loads(p.read_text(encoding="utf-8"))
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def _write_cache(self, key: str, suffix: str, data: Any) -> None:
        try:
            if suffix == ".json":
                write_json(self._path(key, suffix), data)
            else:
                self._path(key, suffix).write_text(str(data), encoding="utf-8")
        except Exception:
            pass

    async def get_text(self, url: str, headers: dict[str, str] | None = None, force: bool = False) -> str | None:
        key = f"text::{url}"
        if not force:
            cached = self._read_cache(key, ".txt")
            if cached is not None:
                self.stats["hit"] += 1
                return cached
        self.stats["miss"] += 1
        h = {"User-Agent": self.ua, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
        if headers:
            h.update(headers)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as c:
                r = await c.get(url, headers=h)
                self.stats["network"] += 1
                if r.status_code >= 400:
                    self.stats["error"] += 1
                    return None
                text = r.text
                self._write_cache(key, ".txt", text)
                return text
        except Exception:
            self.stats["error"] += 1
            return None

    async def get_json(self, url: str, params: dict[str, Any] | None = None,
                       headers: dict[str, str] | None = None, force: bool = False) -> Any | None:
        full = url + ("?" + "&".join(f"{k}={v}" for k, v in (params or {}).items()) if params else "")
        key = f"json::{full}"
        if not force:
            cached = self._read_cache(key, ".json")
            if cached is not None:
                self.stats["hit"] += 1
                return cached
        self.stats["miss"] += 1
        h = {"User-Agent": self.ua}
        if headers:
            h.update(headers)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as c:
                r = await c.get(url, params=params, headers=h)
                self.stats["network"] += 1
                if r.status_code >= 400:
                    self.stats["error"] += 1
                    return None
                data = r.json()
                self._write_cache(key, ".json", data)
                return data
        except Exception:
            self.stats["error"] += 1
            return None

    async def post_json(self, url: str, payload: dict[str, Any],
                        headers: dict[str, str] | None = None, cache_key: str | None = None) -> Any | None:
        key = f"post::{cache_key or url + json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
        cached = self._read_cache(key, ".json")
        if cached is not None:
            self.stats["hit"] += 1
            return cached
        self.stats["miss"] += 1
        h = {"User-Agent": self.ua, "Content-Type": "application/json"}
        if headers:
            h.update(headers)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as c:
                r = await c.post(url, json=payload, headers=h)
                self.stats["network"] += 1
                if r.status_code >= 400:
                    self.stats["error"] += 1
                    return None
                data = r.json()
                self._write_cache(key, ".json", data)
                return data
        except Exception:
            self.stats["error"] += 1
            return None

    async def download_binary(self, url: str, dest: str | Path, ttl_s: int | None = None) -> Path | None:
        dest = Path(dest)
        if dest.exists() and (ttl_s is None or time.time() - dest.stat().st_mtime < ttl_s):
            self.stats["hit"] += 1
            return dest
        ensure_dir(dest.parent)
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(url, headers={"User-Agent": self.ua})
                if r.status_code >= 400:
                    self.stats["error"] += 1
                    return None
                dest.write_bytes(r.content)
                self.stats["network"] += 1
                return dest
        except Exception:
            self.stats["error"] += 1
            return dest if dest.exists() else None
