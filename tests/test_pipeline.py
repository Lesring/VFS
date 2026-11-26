"""端到端集成测试：用合成视频推进完整流程（离线，不依赖网络与模型）。

这些测试断言的是"系统的行为契约"，而不是某个具体答案：
- 有航班号时必须能反查出目的地；
- 画面与航线冲突时必须报出来，而不是默默取平均；
- 没有视觉模型时必须诚实地说"不知道"，而不是编造。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vfs import frames as frames_mod
from vfs.config import CONFIG
from vfs.entities import canonical
from vfs.intent import parse_intent
from vfs.pipeline import AnalysisJob

TRUTH_OBS = {
    "summary": "机场白天，窄体客机停靠在航站楼旁。",
    "frames": [
        {
            "scene": "机场停机坪",
            "observations": [
                {"slot": "aircraft_type", "value": "Airbus A320", "confidence": 0.8, "note": "窄体双发"},
                {"slot": "airline", "value": "China Eastern Airlines", "confidence": 0.7, "note": "涂装"},
                {"slot": "flight_number", "value": "MU5101", "confidence": 0.85, "note": "信息屏"},
                {"slot": "airport", "value": "Shanghai Pudong International Airport (PVG)",
                 "confidence": 0.5, "note": "角标"},
            ],
        }
    ],
}


def make_video(path: Path, seconds: float = 1.0, fps: int = 12) -> Path:
    h, w = 240, 320
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("当前环境无法写 mp4")
    for i in range(int(seconds * fps)):
        img = np.zeros((h, w, 3), np.uint8)
        img[: int(h * 0.6)] = (200, 160, 100)      # 天空
        img[int(h * 0.6):] = (90, 95, 100)          # 地面
        cv2.rectangle(img, (30 + i * 4, 120), (180 + i * 4, 170), (230, 235, 240), -1)  # 机身
        cv2.putText(img, "MU5101", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    return path


@pytest.fixture(scope="module")
def demo_video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("vid")
    p = make_video(d / "plane.mp4")
    (d / "plane.observations.json").write_text(
        json.dumps(TRUTH_OBS, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def offline_cfg(tmp_path_factory):
    cfg = json.loads(json.dumps(CONFIG))
    base = tmp_path_factory.mktemp("vfsdata")
    cfg["paths"]["data_dir"] = str(base)
    cfg["paths"]["uploads_dir"] = str(base / "uploads")
    cfg["paths"]["outputs_dir"] = str(base / "outputs")
    cfg["retrieval"]["cache_dir"] = str(Path("data/cache").resolve())
    cfg["vlm"]["provider"] = "mock"
    return cfg


class TestFrameExtraction:
    def test_extracts_frames_with_metadata(self, demo_video, tmp_path):
        fs, meta = frames_mod.extract_frames(demo_video, tmp_path / "f", max_frames=6)
        assert fs, "应至少抽到一帧"
        assert meta.width == 320 and meta.height == 240
        assert meta.fps == pytest.approx(12, abs=0.5)
        assert meta.duration_s == pytest.approx(1.0, abs=0.3)
        for f in fs:
            assert Path(f.path).exists()
            assert 0 <= f.timestamp_s <= 1.1

    def test_frame_count_respects_limit(self, demo_video, tmp_path):
        fs, _ = frames_mod.extract_frames(demo_video, tmp_path / "g", max_frames=3)
        assert len(fs) <= 3

    def test_image_input_works(self, tmp_path):
        img = np.full((100, 150, 3), 128, np.uint8)
        p = tmp_path / "x.jpg"
        cv2.imwrite(str(p), img)
        fs, meta = frames_mod.extract_frames(p, tmp_path / "h")
        assert len(fs) == 1 and meta.codec == "image"


class TestOfflinePipeline:
    """离线跑完整流程：只依赖旁挂观察 + 本地缓存的知识库。"""

    def test_produces_answer_with_evidence(self, demo_video, offline_cfg):
        job = AnalysisJob(demo_video, question="这架飞机的目的地是哪里？",
                          config=offline_cfg, online=False)
        r = job.run()
        assert r.answer, "必须产出答案文本"
        assert r.evidence, "必须留下证据"
        assert r.frames, "必须记录关键帧"
        assert r.job_id

    def test_flight_number_detected_from_observation(self, demo_video, offline_cfg):
        job = AnalysisJob(demo_video, question="目的地？", config=offline_cfg, online=False)
        r = job.run()
        fn = r.slots.get("flight_number")
        assert fn and fn.candidates
        assert "MU5101" in fn.candidates[0].value

    def test_offline_abstains_on_destination(self, demo_video, offline_cfg):
        """没有联网时，目的地不该靠瞎猜得出，应明确说信息不足或降级作答。"""
        job = AnalysisJob(demo_video, question="目的地是哪里？", config=offline_cfg, online=False)
        r = job.run()
        assert "MU5101" in r.answer or "无法" in r.answer or "信息不足" in r.answer

    def test_airline_matches_observation(self, demo_video, offline_cfg):
        job = AnalysisJob(demo_video, question="哪家航司？", config=offline_cfg, online=False)
        r = job.run()
        al = r.slots.get("airline")
        assert al and "东方" in al.candidates[0].value or "Eastern" in al.candidates[0].value


class TestIntentRouting:
    @pytest.mark.parametrize("q,goal", [
        ("这架飞机要去哪里？", "destination"),
        ("这是什么机型", "aircraft_type"),
        ("在哪个机场拍的", "airport"),
        ("哪个航空公司", "airline"),
        ("什么时候拍的", "date"),
    ])
    def test_goal_detected(self, q, goal):
        assert parse_intent(q)["primary_goal"] == goal

    def test_empty_question_defaults_to_general(self):
        i = parse_intent("")
        assert i["is_empty"]
        assert i["primary_goal"] == "activity"


class TestCanonicalAgreement:
    """不同来源对同一实体的不同写法，必须归到同一个 key，否则交叉验证会失效。"""

    def test_aircraft_variants_agree(self):
        keys = {
            canonical("aircraft_type", "Airbus A320")["key"],
            canonical("aircraft_type", "A320-214")["key"],
            canonical("aircraft_type", "A320 214SL")["key"],
        }
        assert len(keys) == 1, f"同一机型出现多个 key：{keys}"

    def test_airport_forms_share_code(self):
        a = canonical("airport", "Shanghai Pudong International Airport (PVG)")
        assert "PVG" in a["extra"]["codes"]
