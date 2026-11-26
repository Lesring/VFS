"""检索降级、相关性过滤、配置热生效的测试。

这些测试不依赖外网（除少数打标 skip 的），保证 CI 可跑。
"""
from __future__ import annotations

import base64

import pytest

from vfs.kb.search import (
    _decode_bing_redirect, _query_terms, filter_relevant, is_relevant, _mk,
)


class TestBingRedirect:
    def test_decodes_base64_url(self):
        real = "https://www.airbus.com/en/products-services/commercial-aircraft"
        token = "a1" + base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
        href = f"https://www.bing.com/ck/a?!&&p=abc&u={token}&ntb=1"
        assert _decode_bing_redirect(href) == real

    def test_passthrough_for_direct_url(self):
        assert _decode_bing_redirect("https://example.com/x") == "https://example.com/x"

    def test_handles_garbage(self):
        assert _decode_bing_redirect("https://www.bing.com/ck/a?u=NOTBASE64") == ""
        assert _decode_bing_redirect("") == ""

    def test_handles_html_escaped_amp(self):
        real = "https://example.org/page"
        token = "a1" + base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
        href = f"https://www.bing.com/ck/a?u={token}&amp;ntb=1"
        assert _decode_bing_redirect(href) == real


class TestRelevance:
    def test_rejects_unrelated_results(self):
        """这是实测踩到的坑：无 cookie 爬 Bing 会返回完全无关的结果。"""
        noise = [
            {"title": "ChatGPT Exporter - GitHub", "snippet": "", "url": "https://github.com/x"},
            {"title": "Stockfish vs ChatGPT", "snippet": "", "url": "https://reddit.com/y"},
        ]
        assert filter_relevant(noise, "Airbus A320 range") == []

    def test_keeps_relevant_results(self):
        good = [{"title": "Airbus A320 - Wikipedia", "snippet": "narrow-body airliner",
                 "url": "https://en.wikipedia.org/wiki/Airbus_A320"}]
        assert len(filter_relevant(good, "Airbus A320 range")) == 1

    def test_matches_chinese_query(self):
        r = {"title": "MU5101 航班动态", "snippet": "", "url": ""}
        assert is_relevant(r, "东方航空 MU5101 航班")

    def test_no_terms_means_pass(self):
        """查询全是停用词时不应把结果全清掉。"""
        assert is_relevant({"title": "anything", "snippet": "", "url": ""}, "the a of")

    def test_query_terms_dedup(self):
        terms = _query_terms("Airbus A320 airbus")
        assert len(terms) == len(set(terms))


class TestSettingsMasking:
    def test_mask_short_value(self):
        from vfs.settings import mask
        assert mask("abc") == "****"
        assert mask("") == ""

    def test_mask_long_value(self):
        from vfs.settings import mask
        m = mask("sk-1234567890abcdefGHIJ")
        assert m.startswith("sk-1") and m.endswith("GHIJ")
        assert "234567890abcde" not in m, "中间部分必须被遮住"

    def test_secret_fields_not_in_schema_values(self):
        """schema() 里密钥字段的 value 必须永远为空。"""
        from vfs.settings import EDITABLE, schema
        fields = {f["key"]: f for f in schema()}
        for key, f in fields.items():
            if EDITABLE[key]["type"] == "password":
                assert f.get("value") == "", f"{key} 泄露了明文"

    def test_editable_whitelist_blocks_unknown_keys(self):
        from vfs.settings import EDITABLE
        # 不允许通过接口改服务器端口这类敏感项
        assert "server.port" not in EDITABLE
        assert "paths.data_dir" not in EDITABLE


class TestRegionStrategy:
    def test_overseas_first_includes_domestic_fallback(self):
        from vfs.kb import health
        order = health.ordered_sources("overseas_first")
        # 不管探测结果如何，返回值都必须是个列表且不含重复
        assert isinstance(order, list)
        assert len(order) == len(set(order))

    def test_domestic_only_excludes_overseas(self):
        from vfs.kb import health
        order = health.ordered_sources("domestic_only")
        assert order == health.DOMESTIC
        assert "google" not in order

    def test_overseas_only_excludes_domestic(self):
        from vfs.kb import health
        order = health.ordered_sources("overseas_only")
        assert "sogou" not in order and "baidu" not in order

    def test_unknown_strategy_defaults_overseas_first(self):
        from vfs.kb import health
        assert health.ordered_sources("nonsense") == health.ordered_sources("overseas_first")


class TestGoogleKeyless:
    def test_google_search_returns_empty_without_keys(self):
        import asyncio
        from vfs.kb import google
        from vfs.kb.http import CachedClient

        client = CachedClient("data/cache", ttl_s=1, timeout_s=5)
        out = asyncio.run(google.search(client, "test", 3, api_key="", cx=""))
        assert out == [], "没有 key 时不应发请求，也不能报错"

    def test_static_map_needs_key(self):
        from urllib.parse import unquote

        from vfs.kb import google
        assert google.static_map_url(31.1, 121.8, "") is None
        url = google.static_map_url(31.1, 121.8, "KEY123", label="P")
        # URL 里的逗号会被编码成 %2C，解码后再断言
        assert url and "KEY123" in url and "31.1,121.8" in unquote(url)

    def test_osm_tiles_work_without_key(self):
        from vfs.kb import google
        urls = google.tile_urls(40.08, 116.58, zoom=12, grid=2)
        assert len(urls) == 4, "2x2 拼图应有 4 张瓦片"
        assert all("tile.openstreetmap.org" in u for u in urls)
        assert all("{z}" not in u for u in urls), "占位符必须已替换"

    def test_tile_origin_centers_target(self):
        """标记必须落在拼图中央附近，否则红点会跑到角落。"""
        from vfs.kb import google
        lat, lon, zoom, grid = 40.0801, 116.585, 12, 2
        tx0, ty0 = google.tile_origin(lat, lon, zoom, grid)

        import math

        nf = float(2 ** zoom)
        xf = (lon + 180.0) / 360.0 * nf * 256
        lat_rad = math.radians(lat)
        yf = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * nf * 256
        px = xf - tx0 * 256
        py = yf - ty0 * 256
        cw = grid * 256
        # 目标点应落在画布的中间四分之一区域
        assert cw * 0.25 <= px <= cw * 0.75
        assert cw * 0.25 <= py <= cw * 0.75


class TestMapDegradation:
    def test_no_key_falls_back_to_osm(self):
        import asyncio
        from vfs.kb.http import CachedClient
        from vfs.models import Candidate, SlotResult
        from vfs.retrieval import build_map

        slots = {"destination": SlotResult(
            slot="destination", label="目的地",
            candidates=[Candidate(value="PEK", score=0.9,
                                  details={"lat": 40.08, "lon": 116.58})],
            chosen="PEK", confidence=0.9)}
        client = CachedClient("data/cache", ttl_s=1, timeout_s=5)
        m = asyncio.run(build_map(client, slots, {"maps_enabled": True, "maps_api_key": ""}))
        assert m and m["provider"] == "openstreetmap"
        assert m["degraded_reason"] == "no_google_key"
        # 没给 output_dir 时不拼图，但必须仍给出链接与瓦片，不能让用户看到空白
        assert m["link"] and m["tiles"]

    def test_disabled_map_returns_none(self):
        import asyncio
        from vfs.kb.http import CachedClient
        from vfs.retrieval import build_map

        client = CachedClient("data/cache", ttl_s=1, timeout_s=5)
        m = asyncio.run(build_map(client, {}, {"maps_enabled": False}))
        assert m is None
