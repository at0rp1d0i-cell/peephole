"""工具层共享实现：哈希、证据时间戳、JSON 落盘、venv 安装路径与模型 revision。

为什么单独一个模块：`tools/` 下 28 个脚本各自复制过 `sha256_file`（三种分块写法）、`now_cst`
（其中一处用本机时区而不是 +0800）、`open()` 读 JSON，以及把
`venvs/attnview/lib/python3.12/site-packages` 这种**含 CPython 次版本号**的路径写进代码，
还有 7 处重复的模型 revision 字面量。集中后只有一处要跟着环境变。

导入方式：`import _lib`（相对本目录）。脚本直接运行时 `sys.path[0]` 就是脚本自身目录；
被测试用 `importlib` 按路径加载时，由 `tests/conftest.py` 把 `tools/` 放进 `sys.path`。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: 证据时间戳固定用东八区（`+0800`），不随运行机器的时区变化。
CST = timezone(timedelta(hours=8))

#: 1 MiB：小文件不额外多花系统调用，13 GB 级张量也不会一次读进内存。
HASH_CHUNK = 1 << 20

#: 主模型快照的 revision（`Qwen/Qwen3.8-27B`，阶段 02 已核对 sha）。
#: 换 revision 时**只改这里**；`tools/serve-vanilla.sh` 是同值副本（shell 无法 import），
#: 该脚本注释里指向本常量。
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_cst() -> str:
    """证据时间戳口径：`YYYY-MM-DD HH:MM:SS +0800`（固定东八区）。"""
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S +0800")


def load_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Any, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return target


def venv_site_packages(repo_root: Any) -> Path:
    """解析 `venvs/attnview/lib/python3.*/site-packages`，不把 CPython 次版本写进代码。

    顺序：① glob 现有目录（唯一则用）；② 退化到 venv 自己 `pyvenv.cfg` 的 `version_info`；
    ③ 再退化到当前解释器的 `X.Y`。

    ②③ 返回的路径**可能不存在**，这是有意的：补丁预检需要先构造出目标路径、再逐条报告
    "目标缺失"，而不是在解析阶段就抛异常（旧实现在 venv 不存在时给出的是退出码 4 + 清单）。
    只有"存在多份 site-packages"才判为环境异常。
    """
    venv = Path(repo_root) / "venvs" / "attnview"
    matches = sorted((venv / "lib").glob("python3.*/site-packages"))
    if len(matches) > 1:
        raise RuntimeError(f"venvs/attnview 下有多份 site-packages：{[str(m) for m in matches]}")
    if matches:
        return matches[0]
    minor = _venv_python_minor(venv) or f"{sys.version_info.major}.{sys.version_info.minor}"
    return venv / "lib" / f"python{minor}" / "site-packages"


def _venv_python_minor(venv: Path) -> str | None:
    """从 `pyvenv.cfg` 读 venv 自己的 `X.Y`（uv/pip 都会写；读不到返回 None）。"""
    cfg = Path(venv) / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    for key in ("version_info", "version"):
        for line in cfg.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == key and value.strip():
                return ".".join(value.strip().split(".")[:2])
    return None


def venv_vllm_root(repo_root: Any) -> Path:
    """venv 内已安装的 vLLM 包根（补丁部署目标之一）。"""
    return venv_site_packages(repo_root) / "vllm"
