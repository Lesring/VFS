"""地理取证 / Planes&Trains 风格约束推理。

原则：只从画面推出可检验的约束，再用公开数据源筛候选。
不得在代码、提示或检索种子中写入任何挑战标答。
"""

from .runner import run_osint_geolocation

__all__ = ["run_osint_geolocation"]
