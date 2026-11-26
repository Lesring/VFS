from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8000, "max_upload_bytes": 512 * 1024 * 1024},
    "vlm": {
        "provider": "mock",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "max_image_side": 1280,
        "jpeg_quality": 85,
        "temperature": 0.1,
        "timeout_s": 120,
        "max_frames": 12,
        "synthesize": True,
    },
    "frames": {
        "max_frames": 24,
        "min_gap_s": 0.35,
        "scene_threshold": 0.32,
        "uniform_ratio": 0.35,
        "blur_reject": 28.0,
    },
    "retrieval": {
        "enabled": True,
        "timeout_s": 12,
        "cache_dir": "data/cache",
        "cache_ttl_s": 86400,
        "search_backend": "auto",
        # 区域策略：
        #   overseas_first（默认）= 境外优先，逐个探测，不通的自动跳过，最后兜底国内
        #   domestic_first      = 国内优先，境外作为备选
        #   overseas_only / domestic_only = 只用某一个区域
        "region_strategy": "overseas_first",
        "max_search_results": 5,
        "user_agent": "VideoForensics/1.0 (+research)",
        # 可达性探测缓存秒数（避免每次分析都重探，也避免被墙域名卡死）
        "health_ttl_s": 600,
        "dns_timeout_s": 3.0,
    },
    "google": {
        # Programmable Search Engine：真谷歌搜索。需要 key + cx
        # key: https://console.cloud.google.com/apis/credentials
        # cx:  https://programmablesearchengine.google.com/ 创建后得到
        "api_key": "",
        "cx": "",
        # 地图：地理编码与静态图。同一把 key 需另开 Geocoding/Maps Static API
        "maps_api_key": "",
        "maps_enabled": True,
        "map_zoom": 13,
        # 没配 key 时用免 key 的替代（Google News RSS / OSM）
        "allow_keyless": True,
    },
    "ocr": {"enabled": True, "langs": "chi_sim+eng", "min_confidence": 45},
    "reasoning": {"low_confidence": 0.35, "max_candidates": 5, "allow_abstain": True},
    "paths": {
        "data_dir": "data",
        "uploads_dir": "data/uploads",
        "outputs_dir": "data/outputs",
    },
}

# 环境变量 → 配置路径
_ENV_MAP: dict[str, str] = {
    "VFS_VLM_PROVIDER": "vlm.provider",
    "VFS_VLM_BASE_URL": "vlm.base_url",
    "VFS_VLM_API_KEY": "vlm.api_key",
    "VFS_VLM_MODEL": "vlm.model",
    "VFS_VLM_SYNTHESIZE": "vlm.synthesize",
    "VFS_VLM_MAX_FRAMES": "vlm.max_frames",
    "VFS_OCR_ENABLED": "ocr.enabled",
    "VFS_RETRIEVAL_ENABLED": "retrieval.enabled",
    "VFS_SEARCH_BACKEND": "retrieval.search_backend",
    "VFS_REGION_STRATEGY": "retrieval.region_strategy",
    "VFS_MAX_FRAMES": "frames.max_frames",
    "VFS_PORT": "server.port",
    "VFS_DATA_DIR": "paths.data_dir",
    "VFS_GOOGLE_API_KEY": "google.api_key",
    "VFS_GOOGLE_CX": "google.cx",
    "VFS_GOOGLE_MAPS_KEY": "google.maps_api_key",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in _TRUE
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            return current
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            return current
    return raw


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _get_path(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = _merge({}, DEFAULTS)
    candidates = []
    if path:
        candidates.append(Path(path))
    if os.environ.get("VFS_CONFIG"):
        candidates.append(Path(os.environ["VFS_CONFIG"]))
    candidates += [ROOT / "config.yaml", ROOT / "config.example.yaml"]

    for cand in candidates:
        if cand and cand.exists() and yaml is not None:
            try:
                loaded = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
                cfg = _merge(cfg, loaded)
            except Exception:
                pass
            break

    for env_key, dotted in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None or not raw.strip():
            continue
        current = _get_path(cfg, dotted)
        _set_path(cfg, dotted, _coerce(raw, current))

    # provider 自动推断：显式设了 key 但没改 provider，则启用 openai
    if cfg["vlm"]["provider"] == "mock" and cfg["vlm"].get("api_key"):
        cfg["vlm"]["provider"] = "openai"
    if cfg["vlm"]["provider"] == "openai" and not cfg["vlm"].get("api_key"):
        # 回退到常见环境变量
        _fill_vlm_key(cfg)

    # 路径解析为绝对路径
    paths = cfg["paths"]
    for key in ("data_dir", "uploads_dir", "outputs_dir", "cache_dir"):
        holder = paths if key in paths else cfg["retrieval"]
        if key in holder and not Path(holder[key]).is_absolute():
            holder[key] = str(ROOT / holder[key])
    return cfg


_VLM_KEY_ENVS = ("VFS_VLM_API_KEY", "ARK_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")


def _fill_vlm_key(cfg: dict[str, Any]) -> None:
    """页面里的 key 为空时，从环境变量补。ARK_API_KEY 是火山方舟的约定。"""
    if cfg.get("vlm", {}).get("provider") not in ("openai", "openai_compat"):
        return
    if str(cfg["vlm"].get("api_key") or "").strip():
        return
    for env in _VLM_KEY_ENVS:
        raw = os.environ.get(env)
        if raw and raw.strip():
            cfg["vlm"]["api_key"] = raw.strip()
            return


CONFIG: dict[str, Any] = load_config()

# 叠加用户通过网页保存的配置（data/user_config.yaml）
def _apply_user_config() -> None:
    try:
        from .settings import load_into_config
        load_into_config()
    except Exception:
        pass
    _fill_vlm_key(CONFIG)


_apply_user_config()


def get(dotted: str, default: Any = None) -> Any:
    return _get_path(CONFIG, dotted) if _get_path(CONFIG, dotted) is not None else default
