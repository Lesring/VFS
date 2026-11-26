"""实体归一化测试。归一化是跨源比对的基石，错一个字母就会让交叉验证失效。"""
from __future__ import annotations

import pytest

from vfs.entities import (
    canonical, normalize_aircraft_type, normalize_coordinate, normalize_date,
    normalize_flight, normalize_registration, normalize_time,
)


class TestAircraft:
    @pytest.mark.parametrize("text,family", [
        # 子型号编号（-800 / MAX 8 的 8）归并到家族，便于跨源比对
        ("Airbus A320", "A320"),
        ("A320", "A320"),
        ("A350-900", "A350"),
        ("Boeing 737-800", "B737"),
        ("Boeing 777-300ER", "B777"),
        ("737 max 8", "B737 MAX"),
        # 世代（neo / MAX）是不同机型，必须保留
        ("Airbus A320neo", "A320neo"),
        ("a321neo", "A321neo"),
        ("单通道窄体客机", ""),
    ])
    def test_family(self, text, family):
        assert normalize_aircraft_type(text)["family"] == family

    def test_subtype_kept_in_key(self):
        assert normalize_aircraft_type("Boeing 737-800")["key"] == "B737-800"
        assert normalize_aircraft_type("A350-900")["key"] == "A350"

    def test_body_inferred(self):
        assert normalize_aircraft_type("Airbus A320")["body"] == "narrow"
        assert normalize_aircraft_type("Boeing 777")["body"] == "wide"

    def test_body_from_text(self):
        # 型号判断不出机身类别时，用文字描述兜底
        assert normalize_aircraft_type("宽体双通道客机")["body"] == "wide"

    def test_manufacturer(self):
        assert normalize_aircraft_type("A320")["manufacturer"] == "Airbus"
        assert normalize_aircraft_type("737-800")["manufacturer"] == "Boeing"


class TestFlightNumber:
    @pytest.mark.parametrize("text,iata", [
        ("MU5101", "MU5101"),
        ("mu 5101", "MU5101"),
        ("MU-5101", "MU5101"),
        ("CA981", "CA981"),
    ])
    def test_iata(self, text, iata):
        r = normalize_flight(text)
        assert r and r["iata"] == iata

    def test_rejects_garbage(self):
        assert normalize_flight("A320") is None


class TestRegistration:
    @pytest.mark.parametrize("text,expected", [
        ("B-1672", "B-1672"),
        ("B1672", "B-1672"),
        ("N12345", "N12345"),
        ("D-AIBX", "D-AIBX"),
        ("JA01AJ", "JA01AJ"),
        ("9V-SKA", "9V-SKA"),
    ])
    def test_registration(self, text, expected):
        assert normalize_registration(text) == expected


class TestDate:
    @pytest.mark.parametrize("text,expected", [
        ("2024-05-18", "2024-05-18"),
        ("2024年5月18日", "2024-05-18"),
        ("2024/5/18", "2024-05-18"),
        ("拍摄于 2023 年 11 月", "2023-11"),
        ("May 2024", "2024-05"),
        ("no date here", None),
    ])
    def test_date(self, text, expected):
        assert normalize_date(text) == expected


class TestTime:
    @pytest.mark.parametrize("text,expected", [
        ("14:30", "14:30"),
        ("9:05", "09:05"),
        ("拍摄于夜晚", "night"),
        ("黄昏时分", "sunset"),
        ("morning light", "morning"),
    ])
    def test_time(self, text, expected):
        assert normalize_time(text) == expected


class TestCoordinate:
    def test_decimal_pair(self):
        assert normalize_coordinate("31.1434, 121.8050") == (31.1434, 121.805)

    def test_swapped_order(self):
        lat, lon = normalize_coordinate("121.8050, 31.1434")
        assert abs(lat - 31.1434) < 1e-6 and abs(lon - 121.805) < 1e-6

    def test_dms(self):
        lat, lon = normalize_coordinate("39°54'27\"N 116°23'17\"E")
        assert abs(lat - 39.9075) < 0.01
        assert abs(lon - 116.388) < 0.01

    def test_southern_western(self):
        lat, lon = normalize_coordinate("33.9S 151.2E")
        assert lat < 0 and lon > 0


class TestCanonical:
    def test_keys_merge_across_sources(self):
        a = canonical("aircraft_type", "Airbus A320")
        b = canonical("aircraft_type", "A320-214")
        assert a["key"] == b["key"]

    def test_airline_key_stable(self):
        a = canonical("airline", "China Eastern Airlines")
        b = canonical("airline", "china eastern airlines")
        assert a["key"] == b["key"]

    def test_unknown_slot_passthrough(self):
        c = canonical("mystery_slot", "Some Value")
        assert c["display"] == "Some Value"
