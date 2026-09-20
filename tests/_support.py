"""测试共享入口：仓库路径常量与按路径加载脚本。

`load_module` 集中原先 9 份逐字相同的 `importlib` 样板（只因模块名不同）。各测试文件保留
自己的薄包装（`load_runner()` 等），调用点不必改。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
TOOLS = REPO / "tools"


def load_module(name: str, path: Path) -> Any:
    """按路径把一个脚本加载成模块（不依赖它是否装进 site-packages）。

    必须在 `exec_module` **之前**写入 `sys.modules`：脚本里的 `@dataclass` 会让标准库按
    `__module__` 反查模块，缺这一步会在装饰期抛 `AttributeError`。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
