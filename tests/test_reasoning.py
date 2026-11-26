"""融合与推理测试。重点验证交叉校验能真正发现矛盾，而不是只做加权平均。"""
from __future__ import annotations

from vfs.models import Evidence
from vfs.reason import FusionEngine


def make_engine(evs: list[Evidence]) -> FusionEngine:
    e = FusionEngine()
    e.add_evidence(evs)
    e.build_slots()
    return e


class FakeAirport:
    def __init__(self, name, iata, icao, city, country, lat=None, lon=None):
        self.name, self.iata, self.icao = name, iata, icao
        self.city, self.country, self.lat, self.lon = city, country, lat, lon

    def label(self):
        code = "/".join(c for c in (self.iata, self.icao) if c)
        return f"{self.name} ({code})" if code else self.name

    def to_dict(self):
        return {"name": self.name, "iata": self.iata, "icao": self.icao,
                "city": self.city, "country": self.country, "lat": self.lat, "lon": self.lon}


class FakeAirportDB:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def lookup(self, text):
        up = (text or "").upper()
        out = []
        for a in self.items:
            if a.iata.upper() in up or a.name.upper()[:12] in up:
                out.append(a)
        return out

    def nearest(self, lat, lon, limit=3, max_km=300):
        return []


class TestFusionBasics:
    def test_cross_source_reinforces(self):
        """同一结论被两个独立来源确认，得分应高于单一来源。"""
        one = make_engine([Evidence(kind="vlm", source="m1", slot="aircraft_type",
                                    value="A320", confidence=0.7, claim="窄体")])
        two = make_engine([
            Evidence(kind="vlm", source="m1", slot="aircraft_type", value="A320", confidence=0.7, claim="窄体"),
            Evidence(kind="knowledge", source="db", slot="aircraft_type", value="A320", confidence=0.7, claim="登记"),
        ])
        assert two.confidence_of("aircraft_type") > one.confidence_of("aircraft_type")

    def test_different_values_stay_separate(self):
        e = make_engine([
            Evidence(kind="vlm", slot="aircraft_type", value="A320", confidence=0.6, claim="a"),
            Evidence(kind="vlm", slot="aircraft_type", value="B737", confidence=0.5, claim="b"),
        ])
        assert len(e.slots["aircraft_type"].candidates) == 2

    def test_empty_evidence_is_graceful(self):
        e = make_engine([Evidence(kind="vlm", slot="", value="", confidence=0.5, claim="")])
        assert e.slots == {}

    def test_weak_evidence_labelled_low(self):
        e = make_engine([Evidence(kind="heuristic", slot="weather", value="clear", confidence=0.4, claim="亮")])
        assert e.slots["weather"].confidence_label in ("低", "中")


class TestCrossValidation:
    def _route_engine(self):
        """构造一个"航班号 → PEK"的航线知识，和"画面看到 PVG"的视觉观察。"""
        e = FusionEngine()
        e.add_evidence([
            Evidence(kind="vlm", source="m", slot="airport",
                     value="Shanghai Pudong International Airport (PVG)", confidence=0.6,
                     claim="角标显示 PVG"),
            Evidence(kind="vlm", source="m", slot="flight_number", value="MU5101", confidence=0.9,
                     claim="信息屏"),
        ])
        e.build_slots()
        e.infer_from_flight({
            "callsign": "MU5101",
            "airline": {"name": "China Eastern Airlines", "iata": "MU", "icao": "CES"},
            "origin": {"name": "Shanghai Hongqiao International Airport", "iata_code": "SHA",
                       "icao_code": "ZSSS", "municipality": "Shanghai", "country_name": "China",
                       "latitude": 31.1979, "longitude": 121.336},
            "destination": {"name": "Beijing Capital International Airport", "iata_code": "PEK",
                            "icao_code": "ZBAA", "municipality": "Beijing", "country_name": "China",
                            "latitude": 40.0801, "longitude": 116.585},
            "source": "adsbdb",
        }, None)
        return e

    def test_detects_airport_mismatch(self):
        e = self._route_engine()
        e.apply_constraints(airports=None)
        e.finalize()
        notes = " ".join(e.slots["airport"].notes)
        assert "不匹配" in notes, "画面机场与航线机场不一致时应给出提示"

    def test_penalty_persists_across_rebuild(self):
        """冲突扣分必须能扛住后续的 build_slots() 重建。"""
        e = self._route_engine()
        e.apply_constraints(airports=None)
        before = e.slots["destination"].confidence
        e.build_slots()  # 模拟再次重建
        after = e.slots["destination"].confidence
        assert after <= before + 1e-9
        assert any("冲突" in r for r in e.slots["destination"].candidates[0].reasons)

    def test_match_strengthens(self):
        """画面机场与航线一致时，应被标记为互相印证而不是冲突。"""
        e = self._route_engine()
        # 把画面里的机场改成 PEK
        e2 = FusionEngine()
        e2.add_evidence([
            Evidence(kind="vlm", source="m", slot="airport",
                     value="Beijing Capital International Airport (PEK)", confidence=0.6,
                     claim="指示牌 PEK"),
        ])
        e2.build_slots()
        e2.infer_from_flight({
            "callsign": "MU5101",
            "airline": {"name": "China Eastern Airlines"},
            "origin": {"name": "Shanghai Hongqiao", "iata_code": "SHA", "icao_code": "ZSSS",
                       "latitude": 31.2, "longitude": 121.3},
            "destination": {"name": "Beijing Capital International Airport", "iata_code": "PEK",
                            "icao_code": "ZBAA", "latitude": 40.08, "longitude": 116.58},
            "source": "adsbdb",
        }, None)
        e2.apply_constraints(airports=None)
        e2.finalize()
        assert any("一致" in n or "印证" in n for n in e2.slots["airport"].notes)

    def test_registration_owner_conflict(self):
        e = FusionEngine()
        e.add_evidence([
            Evidence(kind="vlm", slot="airline", value="China Eastern Airlines", confidence=0.8,
                     claim="涂装"),
            Evidence(kind="vlm", slot="registration", value="B-1672", confidence=0.6, claim="机身"),
        ])
        e.build_slots()
        e.infer_from_registration({"registration": "B-1672", "type": "A320-214",
                                   "owner": "Spring Airlines"})
        e.apply_constraints(airports=None)
        e.finalize()
        notes = " ".join(e.slots["airline"].notes + e.slots["registration"].notes)
        assert "Spring Airlines" in notes


class TestRangeConstraint:
    def test_widebody_needed_for_long_route(self):
        """窄体机 + 跨洋航线 → 应报出航程张力。"""
        e = FusionEngine()
        e.add_evidence([
            Evidence(kind="vlm", slot="aircraft_type", value="A320", confidence=0.8, claim="窄体"),
        ])
        e.build_slots()
        e._inject("origin", "Shanghai Pudong (PVG)", 0.95, "test", "起点", verified=True,
                  extra={"iata": "PVG", "lat": 31.14, "lon": 121.80})
        e._inject("destination", "New York JFK (JFK)", 0.95, "test", "终点", verified=True,
                  extra={"iata": "JFK", "lat": 40.64, "lon": -73.78})
        e.build_slots()
        e.apply_constraints(airports=None)
        notes = " ".join(e.slots["aircraft_type"].notes)
        assert "航程" in notes


class TestRenumber:
    def test_prefixes_reflect_kind(self):
        e = make_engine([
            Evidence(kind="vlm", slot="city", value="Shanghai", confidence=0.5, claim="a"),
            Evidence(kind="ocr", slot="text_sign", value="PVG", confidence=0.5, claim="b"),
            Evidence(kind="knowledge", slot="country", value="China", confidence=0.5, claim="c"),
        ])
        e.finalize()
        ids = [ev.id for ev in e.evidence_list()]
        assert any(i.startswith("V") for i in ids)
        assert any(i.startswith("O") for i in ids)
        assert any(i.startswith("K") for i in ids)
