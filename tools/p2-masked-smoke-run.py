#!/usr/bin/env python3
"""One bounded diagnostic transaction: apply -> verify -> run -> finally revert."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_path = REPO / "configs/p2-masked-smoke/diagnostic.json"
    config = json.loads(config_path.read_text())
    command = [sys.executable, "tools/p2-calib-run.py", "--arm", config["arm"],
               "--out", str(out / "run"), "--max-tokens", str(config["max_tokens"]),
               "--record-layers", "--cleanup-check"]
    for key in ("doc_fixture", "expect_fixture", "timeline_config", "force_trajectory"):
        command += ["--" + key.replace("_", "-"), config[key]]
    state = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "command": command, "started_unix": time.time(), "phases": {},
        "config": config, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
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
