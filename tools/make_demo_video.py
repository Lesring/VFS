"""生成一段合成"机场+飞机"演示视频，用于端到端自测。

画面元素：跑道、航站楼、一架带航司涂装的窄体客机、登机桥、航班信息屏（写有航班号与目的地）、
地面车辆、随时间缓慢移动的镜头。生成的视频同目录会附带 observations.json（旁挂真值），
供 mock VLM 在没有 API key 时跑通完整推理链。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
FPS = 24
DURATION = 6.0

# 演示用的"真值"（仅用于测试，不参与推理逻辑）
TRUTH = {
    "flight_number": "MU5101",
    "airline": "中国东方航空",
    "aircraft_type": "Airbus A320",
    "airport": "Shanghai Pudong International Airport (PVG/ZSPD)",
    "city": "Shanghai",
    "country": "China",
    "date": "2024-05-18",
    "time_of_day": "14:30",
    "weather": "clear",
}


def sky(img, t):
    for y in range(0, int(H * 0.62)):
        k = y / (H * 0.62)
        c = (int(190 - 40 * k), int(150 - 30 * k), int(90 + 10 * k))
        img[y, :] = c


def ground(img, t):
    y0 = int(H * 0.62)
    cv2.rectangle(img, (0, y0), (W, H), (86, 92, 96), -1)
    # 跑道
    cv2.rectangle(img, (0, y0 + 40), (W, H), (72, 74, 78), -1)
    off = int((t * 260) % 160)
    for x in range(-160, W + 160, 160):
        cv2.rectangle(img, (x + off, y0 + 120), (x + off + 80, y0 + 132), (230, 230, 235), -1)
    for x in range(-160, W + 160, 160):
        cv2.rectangle(img, (x + off, H - 70), (x + off + 80, H - 58), (230, 230, 235), -1)


def terminal(img, t):
    y0 = int(H * 0.62)
    cv2.rectangle(img, (0, y0 - 150), (int(W * 0.30), y0 + 30), (58, 62, 70), -1)
    for i in range(6):
        cv2.rectangle(img, (18 + i * 62, y0 - 132), (66 + i * 62, y0 - 40), (140, 196, 226), -1)
    cv2.putText(img, "T2", (int(W * 0.03), y0 - 162), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 245), 3)
    cv2.putText(img, "GATE 218", (int(W * 0.09), y0 - 162), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 240), 2)


def board(img, t):
    """航班信息屏：写有航班号与目的地，是本题的决定性线索。"""
    x, y, w, h = W - 420, 40, 380, 210
    cv2.rectangle(img, (x, y), (x + w, y + h), (16, 18, 24), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 130), 2)
    cv2.putText(img, "FLIGHT   DEST", (x + 18, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 200, 255), 2)
    rows = [("MU5101", "PEK"), ("MU5102", "CAN"), ("CA981", "JFK"), ("CZ347", "AMS")]
    for i, (fl, dst) in enumerate(rows):
        yy = y + 76 + i * 34
        col = (255, 255, 255) if i == 0 else (150, 160, 175)
        cv2.putText(img, fl, (x + 18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(img, dst, (x + 210, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(img, "MU5101 DEP 14:30", (x + 18, y + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 180), 1)


def aircraft(img, t):
    """侧视窄体客机：单通道机身、翼下双发、上翘翼梢，中国东方航空涂装。"""
    cx = int(W * 0.52 + 26 * math.sin(t * 0.7))
    cy = int(H * 0.50)
    body_l, body_h = 420, 52
    x0, x1 = cx - body_l // 2, cx + body_l // 2
    # 机身
    cv2.ellipse(img, ((x0 + x1) // 2, cy), (body_l // 2, body_h // 2), 0, 0, 360, (235, 238, 242), -1)
    # 机头
    cv2.ellipse(img, (x1 - 20, cy), (56, body_h // 2), 0, -70, 90, (235, 238, 242), -1)
    # 尾翼
    tail = np.array([[x0 + 60, cy - 12], [x0 + 10, cy - 120], [x0 + 88, cy - 114], [x0 + 116, cy - 12]], np.int32)
    cv2.fillPoly(img, [tail], (40, 62, 150))
    # 垂尾标志（白色圆点，示意航司 logo）
    cv2.circle(img, (x0 + 66, cy - 78), 16, (245, 245, 248), -1)
    # 机翼
    wing = np.array([[cx - 40, cy + 8], [cx + 130, cy + 8], [cx + 186, cy + 62], [cx + 20, cy + 30]], np.int32)
    cv2.fillPoly(img, [wing], (205, 210, 218))
    # 发动机（翼下两台）
    for dx in (-92, 46):
        cv2.ellipse(img, (cx + dx, cy + 30), (44, 22), 0, 0, 360, (170, 175, 185), -1)
        cv2.ellipse(img, (cx + dx + 30, cy + 30), (13, 20), 0, 0, 360, (110, 115, 125), -1)
    # 舷窗
    for i in range(18):
        wx = x0 + 132 + i * 15
        if wx < x1 - 34:
            cv2.circle(img, (wx, cy - 12), 3, (70, 78, 96), -1)
    # 红色腰线（东航涂装特征）
    cv2.line(img, (x0 + 26, cy + 6), (x1 - 8, cy + 6), (40, 40, 170), 5)
    # 机身文字
    cv2.putText(img, "CHINA EASTERN", (cx - 96, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 150), 2)
    cv2.putText(img, "B-1672", (x0 + 100, cy - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (95, 100, 115), 1)
    # 起落架
    for dx in (-60, 120):
        cv2.line(img, (cx + dx, cy + 24), (cx + dx, cy + 86), (60, 62, 70), 7)
        cv2.circle(img, (cx + dx, cy + 90), 12, (32, 32, 36), -1)


def vehicles(img, t):
    y = int(H * 0.80)
    x = int((t * 120) % (W + 200)) - 100
    cv2.rectangle(img, (x, y), (x + 90, y + 38), (200, 180, 40), -1)
    cv2.putText(img, "FOLLOW ME", (x + 4, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1)
    x2 = int(W * 0.2 + 40 * math.sin(t))
    cv2.rectangle(img, (x2, y + 60), (x2 + 150, y + 96), (170, 175, 180), -1)


def hud(img, t):
    cv2.putText(img, "2024-05-18 14:30", (18, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
    cv2.putText(img, "PVG  ZSPD", (18, H - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2)


def render(t: float) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    sky(img, t)
    ground(img, t)
    terminal(img, t)
    aircraft(img, t)
    vehicles(img, t)
    board(img, t)
    hud(img, t)
    return img


def write_frame(rpath: Path, episode: dict) -> None:
    frames = []
    n = int(FPS * DURATION)
    for i in range(n):
        t = i / FPS
        img = render(t)
        p = rpath / f"f{i:04d}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        frames.append(img)
    _write_video(frames, rpath.parent / (rpath.name + ".mp4"))


def _write_video(frames: list[np.ndarray], out: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, FPS, (W, H))
    if not vw.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = out.with_suffix(".avi")
        vw = cv2.VideoWriter(str(out), fourcc, FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    print("wrote", out)


def observations() -> dict:
    """旁挂真值：只给"单帧能看见的"观察，不给答案。"""
    return {
        "summary": "机场停机坪白天，一架窄体客机停靠在航站楼旁，右侧有航班信息屏。",
        "frames": [
            {
                "scene": "机场停机坪，白天，天空晴朗，一架喷气式客机侧面。",
                "observations": [
                    {"slot": "aircraft_type", "value": "Airbus A320 family, narrow-body, twin-engine under wing",
                     "confidence": 0.78, "note": "单通道窄体、翼下双发、机头上翘外形符合 A320 家族"},
                    {"slot": "airline", "value": "China Eastern Airlines", "confidence": 0.72,
                     "note": "蓝色垂尾+红色腰线+机身 CHINA EASTERN 字样，符合东航涂装"},
                    {"slot": "airport", "value": "Shanghai Pudong International Airport (PVG)", "confidence": 0.45,
                     "note": "画面角标显示 PVG ZSPD"},
                    {"slot": "city", "value": "Shanghai", "confidence": 0.5, "note": "PVG 位于上海"},
                    {"slot": "country", "value": "China", "confidence": 0.6, "note": "中国"},
                    {"slot": "date", "value": "2024-05-18", "confidence": 0.7, "note": "画面时间戳 2024-05-18"},
                    {"slot": "time_of_day", "value": "14:30", "confidence": 0.7, "note": "画面时间戳 14:30"},
                    {"slot": "weather", "value": "clear", "confidence": 0.6, "note": "能见度高，天空湛蓝"},
                    {"slot": "activity", "value": "parked at gate", "confidence": 0.65, "note": "停在登机口"},
                    {"slot": "object", "value": "boarding bridge, GATE 218", "confidence": 0.5, "note": "登机桥编号"},
                ],
                "readable_text": ["MU5101", "PEK", "MU5102", "CAN", "CA981", "JFK", "CZ347", "AMS",
                                  "MU5101 DEP 14:30", "PVG ZSPD", "2024-05-18 14:30", "GATE 218"],
            },
            {
                "scene": "同一架飞机，镜头轻微平移，可见航班信息屏与地面车辆。",
                "observations": [
                    {"slot": "flight_number", "value": "MU5101", "confidence": 0.88,
                     "note": "航班信息屏首行高亮 MU5101"},
                    {"slot": "aircraft_type", "value": "Airbus A320", "confidence": 0.8, "note": "机身比例与翼梢特征一致"},
                    {"slot": "airline", "value": "中国东方航空", "confidence": 0.85,
                     "note": "都向中国东方航空，MU 为其 IATA 码"},
                    {"slot": "text_sign", "value": "MU5101 DEP 14:30", "confidence": 0.8,
                     "note": "屏幕文字，含航班号与起飞时刻"},
                ],
                "readable_text": ["MU5101", "DEP 14:30", "GATE 218"],
            },
        ],
        "synthesis": {
            "reasoning": "两帧一致指向东航 MU5101，A320 窄体，除信息屏外无矛盾线索。",
            "observations": [
                {"slot": "flight_number", "value": "MU5101", "confidence": 0.9, "note": "两帧信息屏一致"},
                {"slot": "registration", "value": "B-1672", "confidence": 0.45, "note": "机身小字，可能为注册号"},
            ],
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tmp = root / "data" / "demo"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "frames").mkdir(exist_ok=True)
    (tmp / "observations.json").write_text(
        json.dumps(observations(), ensure_ascii=False, indent=2), encoding="utf-8")

    out = tmp / "airport_plane.mp4"
    n = int(FPS * DURATION)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out), fourcc, FPS, (W, H))
    if not vw.isOpened():
        out = tmp / "airport_plane.avi"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"XVID"), FPS, (W, H))
    for i in range(n):
        vw.write(render(i / FPS))
    vw.release()
    print("wrote", out, out.stat().st_size, "bytes")
    png = tmp / "frame_preview.jpg"
    cv2.imwrite(str(png), render(2.0), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("wrote", png)
    print("truth:", json.dumps(TRUTH, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
