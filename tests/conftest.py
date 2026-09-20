"""pytest 入口：统一 `sys.path`，测试文件不再各自插路径。

- 仓库根：让 `from tests._support import ...` 这类跨文件复用成立（原先只靠 `python -m pytest`
  把 CWD 带进 `sys.path`，用 `pytest tests/` 直接跑就会 ImportError）；
- `src/`：让 `import attnview` 指向**仓库内源码**，而不是 site-packages 里那份只含 7 个模块的
  已部署副本；
- `tools/`：让按路径加载的脚本（`importlib`）能 `import _lib`（工具层共享模块）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: 后插入者在 sys.path 更前面，因此最终顺序是 tools → src → 仓库根。
for _path in (REPO, REPO / "src", REPO / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
