"""运行时设置：让用户在网页上配置模型与数据源，无需改文件或重启。

设计要点：
1. 用户配置写到 data/user_config.yaml，与 config.yaml 分离，升级代码不会覆盖用户设置。
2. 读取时的优先级：环境变量 > 用户配置 > config.yaml > 默认值。
3. API key 只写不读——接口返回时打码，避免密钥通过浏览器泄露。
4. 保存后重载配置并清空 VLM 缓存，让新模型立即生效，无需重启进程。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

from . import config as cfg_mod
from .utils import ensure_dir

_LOCK = threading.Lock()

# 允许在页面上修改的字段（白名单，避免任意键注入）
EDITABLE: dict[str, dict[str, Any]] = {
    # ---- 视觉模型 ----
    "vlm.provider": {
        "label": "视觉模型后端", "group": "模型", "type": "select",
        "options": ["mock", "openai", "ollama", "anthropic"],
        "help": "mock=不调用模型（仅 OCR + 图像统计）；openai=任何 OpenAI 兼容接口",
    },
    "vlm.model": {"label": "模型名称", "group": "模型", "type": "text",
                  "placeholder": "gpt-4o-mini / qwen2.5vl:7b / claude-sonnet-4-5"},
    "vlm.base_url": {"label": "接口地址", "group": "模型", "type": "text",
                     "placeholder": "https://api.openai.com/v1",
                     "help": "OpenAI 兼容服务填到 /v1 为止；Ollama 填 http://127.0.0.1:11434"},
    "vlm.api_key": {"label": "API Key", "group": "模型", "type": "password",
                    "help": "仅保存在本机 data/user_config.yaml，接口返回时打码"},
    "vlm.max_frames": {"label": "送入模型的帧数上限", "group": "模型", "type": "number",
                       "min": 1, "max": 40, "help": "越多越准，也越慢越贵"},
    "vlm.max_image_side": {"label": "单帧最大边长(px)", "group": "模型", "type": "number",
                            "min": 320, "max": 2560},
    "vlm.synthesize": {"label": "跨帧综合", "group": "模型", "type": "bool",
                        "help": "多花一次模型调用，换取跨帧一致性与冲突检查"},

    # ---- 检索区域策略 ----
    "retrieval.region_strategy": {
        "label": "数据源区域策略", "group": "检索", "type": "select",
        "options": ["overseas_first", "domestic_first", "overseas_only", "domestic_only"],
        "help": "overseas_first=境外优先，不可达时自动降级国内（推荐）",
    },
    "retrieval.search_backend": {
        "label": "搜索后端", "group": "检索", "type": "select",
        "options": ["auto", "google", "bing", "duckduckgo", "tavily", "brave", "serper", "sogou", "baidu"],
        "help": "auto=按区域策略自动选择并降级",
    },
    "retrieval.max_search_results": {"label": "每次搜索条数", "group": "检索", "type": "number",
                                     "min": 1, "max": 10},

    # ---- Google ----
    "google.api_key": {"label": "Google API Key", "group": "Google", "type": "password",
                       "help": "用于 Programmable Search（真谷歌搜索），在 Google Cloud 创建"},
    "google.cx": {"label": "搜索引擎 ID (cx)", "group": "Google", "type": "text",
                  "help": "programmablesearchengine.google.com 创建后获得；与 API Key 配对才生效"},
    "google.maps_api_key": {"label": "Google Maps Key", "group": "Google", "type": "password",
                            "help": "用于地理编码与静态地图；不填则用 OpenStreetMap 兜底"},
    "google.maps_enabled": {"label": "启用地图可视化", "group": "Google", "type": "bool"},

    # ---- 其他检索 key ----
    "retrieval.tavily_key": {"label": "Tavily Key", "group": "检索", "type": "password",
                             "help": "可选，境外搜索兜底"},
    "retrieval.brave_key": {"label": "Brave Key", "group": "检索", "type": "password", "help": "可选"},
    "retrieval.serper_key": {"label": "Serper Key", "group": "检索", "type": "password", "help": "可选"},

    # ---- OCR ----
    "ocr.enabled": {"label": "启用 OCR", "group": "其他", "type": "bool",
                    "help": "需要系统安装 tesseract；未安装会自动跳过"},
    "ocr.langs": {"label": "OCR 语言", "group": "其他", "type": "text",
                  "placeholder": "chi_sim+eng"},

    # ---- 推理 ----
    "reasoning.low_confidence": {"label": "低置信度阈值", "group": "其他", "type": "number",
                                  "min": 0.05, "max": 0.9, "step": 0.05,
                                  "help": "低于此值标注为「推测」"},
    "reasoning.allow_abstain": {"label": "允许弃答", "group": "其他", "type": "bool",
                                 "help": "证据不足时明确说不知道，而不是编答案"},
}

# 密钥类字段：接口返回时打码
SECRET_FIELDS = {k for k, v in EDITABLE.items() if v.get("type") == "password"}
# 这些字段虽然不在 EDITABLE 白名单，但需要映射到环境变量风格的 key
_EXTRA_SECRET_KEYS = {"retrieval.tavily_key", "retrieval.brave_key", "retrieval.serper_key"}


def user_config_path() -> Path:
    return Path(cfg_mod.CONFIG["paths"]["data_dir"]) / "user_config.yaml"


def _set_path(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = data
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _get_path(data: dict, dotted: str) -> Any:
    cur: Any = data
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def load_user_config() -> dict[str, Any]:
    p = user_config_path()
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _write_user_config(data: dict[str, Any]) -> None:
    p = user_config_path()
    ensure_dir(p.parent)
    # 只保留有内容的，避免把一堆空字符串写进文件
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        os.chmod(p, 0o600)  # 含密钥，尽量收紧权限
    except Exception:
        pass


def mask(value: str) -> str:
    if not value:
        return ""
    v = str(value)
    if len(v) <= 8:
        return "****"
    return f"{v[:4]}{'*' * min(12, len(v) - 8)}{v[-4:]}"


def _is_secret_set(dotted: str, value: Any) -> bool:
    """密钥类字段是否已配置（从配置或环境变量）。"""
    if value:
        return True
    env_map = {
        "vlm.api_key": ("VFS_VLM_API_KEY", "ARK_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"),
        "google.api_key": ("VFS_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
        "google.maps_api_key": ("VFS_GOOGLE_MAPS_KEY", "GOOGLE_MAPS_API_KEY"),
        "retrieval.tavily_key": ("TAVILY_API_KEY",),
        "retrieval.brave_key": ("BRAVE_API_KEY",),
        "retrieval.serper_key": ("SERPER_API_KEY",),
    }
    return any(os.environ.get(e) for e in env_map.get(dotted, ()))


def schema() -> list[dict[str, Any]]:
    """给前端的字段描述 + 当前值（密钥打码）。"""
    cur = cfg_mod.CONFIG
    out = []
    for dotted, meta in EDITABLE.items():
        raw = _get_path(cur, dotted)
        # 检索 key 存在环境变量或单独位置
        if dotted.endswith("_key") and raw is None:
            env = {"retrieval.tavily_key": "TAVILY_API_KEY",
                   "retrieval.brave_key": "BRAVE_API_KEY",
                   "retrieval.serper_key": "SERPER_API_KEY"}.get(dotted)
            raw = os.environ.get(env) if env else None
        item = {
            "key": dotted,
            "label": meta["label"],
            "group": meta["group"],
            "type": meta["type"],
            "help": meta.get("help", ""),
            "placeholder": meta.get("placeholder", ""),
            "options": meta.get("options"),
            "min": meta.get("min"),
            "max": meta.get("max"),
            "step": meta.get("step"),
            "configured": _is_secret_set(dotted, raw),
        }
        if meta["type"] == "password":
            item["value"] = ""          # 密钥绝不回传明文
            item["masked"] = mask(str(raw or ""))
        else:
            item["value"] = raw if raw is not None else (False if meta["type"] == "bool" else "")
        out.append(item)
    return out


def save(updates: dict[str, Any]) -> dict[str, Any]:
    """保存用户配置并热重载。返回 {saved: [...], ignored: [...]}。"""
    saved, ignored = [], []
    user = load_user_config()

    for dotted, value in (updates or {}).items():
        if dotted not in EDITABLE:
            ignored.append(dotted)
            continue
        meta = EDITABLE[dotted]
        # 密钥留空表示"不修改"，避免用户误清空已有 key
        if meta["type"] == "password" and (value is None or str(value).strip() == ""):
            continue
        coerced = _coerce(value, meta)
        if dotted.endswith("_key"):
            _write_secret(dotted, coerced)
        _set_path(user, dotted, coerced)
        saved.append(dotted)

    with _LOCK:
        _write_user_config(user)
        reload_config()
    return {"saved": saved, "ignored": ignored}


def _coerce(value: Any, meta: dict[str, Any]) -> Any:
    t = meta["type"]
    if t == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if t == "number":
        try:
            f = float(value)
        except (TypeError, ValueError):
            return value
        return int(f) if f.is_integer() and meta.get("step") in (None, 1) else f
    return str(value).strip()


def _write_secret(dotted: str, value: str) -> None:
    """把检索类 key 存到 retrieval 段下，供 search 模块读取。"""
    mapping = {
        "retrieval.tavily_key": "TAVILY_API_KEY",
        "retrieval.brave_key": "BRAVE_API_KEY",
        "retrieval.serper_key": "SERPER_API_KEY",
    }
    env = mapping.get(dotted)
    if env:
        os.environ[env] = value


def reload_config() -> None:
    """重新加载配置，并清掉 VLM 缓存让新模型立即生效。"""
    fresh = cfg_mod.load_config()
    user = load_user_config()
    if user:
        fresh = cfg_mod._merge(fresh, user)
    # 环境变量再次覆盖
    for env_key, dotted in cfg_mod._ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None or not raw.strip():
            continue
        cur = cfg_mod._get_path(fresh, dotted)
        cfg_mod._set_path(fresh, dotted, cfg_mod._coerce(raw, cur))

    cfg_mod._fill_vlm_key(fresh)
    cfg_mod.CONFIG.clear()
    cfg_mod.CONFIG.update(fresh)

    # 清 VLM 缓存，让页面改的模型立刻生效
    try:
        from .vlm.base import _CACHE
        _CACHE.clear()
    except Exception:
        pass

    # 同步检索 key 到环境变量（search 模块从环境读）
    for dotted, env in (("retrieval.tavily_key", "TAVILY_API_KEY"),
                        ("retrieval.brave_key", "BRAVE_API_KEY"),
                        ("retrieval.serper_key", "SERPER_API_KEY")):
        v = _get_path(user, dotted)
        if v:
            os.environ[env] = str(v)


def load_into_config() -> None:
    """启动时把用户配置叠加到全局配置。"""
    user = load_user_config()
    if not user:
        return
    merged = cfg_mod._merge(cfg_mod.CONFIG, user)
    cfg_mod._fill_vlm_key(merged)
    cfg_mod.CONFIG.clear()
    cfg_mod.CONFIG.update(merged)
