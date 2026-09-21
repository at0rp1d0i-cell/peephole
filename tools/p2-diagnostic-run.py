#!/usr/bin/env python3
"""一次有界诊断事务：apply -> verify -> run -> finally revert。

配置只描述"跑哪个臂、用什么固定输入"；命令构造、阶段顺序、看门狗与
`vllm-patch/deployed.json` 归属检查都在本工具内，arms 不各自复制一份外壳。
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="新事务目录；已存在即拒绝")
    parser.add_argument("--config", type=Path,
                        default=REPO / "configs/p2-masked-smoke/diagnostic.json",
                        help="诊断配置（arm + 固定输入）；默认 masked 候选诊断")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    # 配置只允许出现本工具真正传给驱动的键：多出来的键不会被消费，却会让人以为存在旋钮。
    keys = ("arm", "max_tokens", "doc_fixture", "expect_fixture", "timeline_config", "force_trajectory")
    missing = [key for key in keys if key not in config]
    unknown = sorted(set(config) - set(keys))
    if missing or unknown:
        raise SystemExit(f"配置 {config_path} 缺失 {missing} / 多余 {unknown}")
    command = [sys.executable, "tools/p2-calib-run.py", "--arm", config["arm"],
               "--out", str(out / "run"), "--max-tokens", str(config["max_tokens"]),
               "--record-layers", "--cleanup-check"]
    for key in keys[2:]:
        command += ["--" + key.replace("_", "-"), config[key]]
    command += ["--compare-to", config["force_trajectory"]]
    state = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "command": command, "started_unix": time.time(), "phases": {},
        "config_path": os.path.relpath(config_path, REPO), "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "diagnostic_only": True,
    }
    def save():
        (out / "transaction.json").write_text(json.dumps(state, indent=2))

    def run(name, cmd, timeout=120):
        state["phases"][name] = {"command": cmd, "started_unix": time.time()}
        save()
        with (out / (name + ".log")).open("w") as stream:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            state["phases"][name]["pid"] = proc.pid
            save()
            try:
                code = proc.wait(timeout=timeout)
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
            finally:
                state["phases"][name].update(exit_code=proc.returncode, ended_unix=time.time())
                save()
        return code

    code = 1
    journal = REPO / "vllm-patch/deployed.json"
    if journal.exists():
        state["error"] = "pre-existing deployment; refusing to own or revert another transaction"
        save()
        return 1
    patch_tool = [sys.executable, "tools/p2-apply-patch.py"]
    try:
        for phase in ("apply", "verify"):
            if run(phase, patch_tool + [phase]):
                raise RuntimeError(f"{phase} failed")
        if not journal.exists() or json.loads(journal.read_text())["state"] != "deployed":
            raise RuntimeError("verify did not establish deployed state")
        code = run("gpu", command, timeout=900 + 180 + 120 + 60)
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if journal.exists():
            state["revert_exit_code"] = run("revert", patch_tool + ["revert"])
            if state["revert_exit_code"]:
                code = 1
        state.update(exit_code=code, ended_unix=time.time(), deployment_journal_remaining=journal.exists())
        save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
