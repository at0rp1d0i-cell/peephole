"""阶段 05 窄 patch 的静态不变量与清单自洽（CPU，不部署、不导入 vLLM）。

覆盖 SUP-004-R1 §2 的：真实签名传递、非 FA metadata 不变、部署脚本非破坏性。
本文件只读仓库文件；`apply` 不会被调用。
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIN_PKG = REPO / "vllm" / "vllm"
INSTALL_PKG = REPO / "venvs/attnview/lib/python3.12/site-packages/vllm"
PATCH_ROOT = REPO / "vllm-patch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    path = PATCH_ROOT / "manifest.json"
    if not path.is_file():
        raise unittest.SkipTest("缺少 vllm-patch/manifest.json（先运行 tools/p2-gen-patch.py）")
    return json.loads(path.read_text())


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def find_func(tree: ast.Module, name: str, *, cls: str | None = None) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            continue
        if cls is None:
            return node
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef) and parent.name == cls and node in parent.body:
                return node
    raise AssertionError(f"未找到函数 {cls + '.' if cls else ''}{name}")


def func_arg_names(node: ast.FunctionDef) -> set[str]:
    args = node.args
    names = {a.arg for a in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs)}
    if args.vararg:
        names.add("*" + args.vararg.arg)
    if args.kwarg:
        names.add("**" + args.kwarg.arg)
    return names


def assigned_names(node: ast.AST) -> set[str]:
    """函数体内被赋值的名字（含 for/with/comprehension 目标）。"""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            targets = sub.targets
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            targets = [sub.target]
        elif isinstance(sub, (ast.For, ast.AsyncFor)):
            targets = [sub.target]
        elif isinstance(sub, (ast.With, ast.AsyncWith)):
            targets = [i.optional_vars for i in sub.items if i.optional_vars is not None]
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out.add(t.id)
            elif isinstance(t, ast.Attribute):
                out.add("attr:" + t.attr)
    return out


class ManifestConsistencyTest(unittest.TestCase):
    """清单必须能在仓库内自证：前置哈希指向 pin、后置哈希指向 patched 产物。"""

    def setUp(self) -> None:
        self.manifest = load_manifest()

    def test_edited_files_pre_hashes_match_pin(self) -> None:
        self.assertTrue(self.manifest["edits"], "edits 为空")
        for rel, info in self.manifest["edits"].items():
            with self.subTest(rel=rel):
                self.assertEqual(sha256(PIN_PKG / rel), info["pre_sha256"])

    def test_edited_files_post_hashes_match_patched_artifacts(self) -> None:
        for rel, info in self.manifest["edits"].items():
            with self.subTest(rel=rel):
                artifact = REPO / info["patched_path"]
                self.assertTrue(artifact.is_file(), f"缺少 {artifact}")
                self.assertEqual(sha256(artifact), info["post_sha256"])
                self.assertNotEqual(info["pre_sha256"], info["post_sha256"])

    def test_new_and_package_file_hashes_match_sources(self) -> None:
        self.assertTrue(self.manifest["new_files"], "new_files 为空")
        for item in self.manifest["new_files"]:
            with self.subTest(src=item["src"]):
                self.assertEqual(sha256(REPO / item["src"]), item["sha256"])
        self.assertTrue(self.manifest["package_files"], "package_files 为空")
        for item in self.manifest["package_files"]:
            with self.subTest(file=item["file"]):
                self.assertEqual(sha256(REPO / item["file"]), item["sha256"])

    def test_pin_commit_recorded(self) -> None:
        self.assertEqual(
            self.manifest.get("pin_commit"), "98dff2a81d747d1dba01a47f939f48c3526d4206"
        )


class PatchShapeTest(unittest.TestCase):
    """补丁内容必须与设计一致：参数通道齐备、落位只在 FA 组、守卫在位。"""

    @classmethod
    def setUpClass(cls) -> None:
        m = load_manifest()
        cls.p = {rel: REPO / info["patched_path"] for rel, info in m["edits"].items()}
        cls.orig = {rel: PIN_PKG / rel for rel in m["edits"]}

    def test_build_attn_metadata_takes_override(self) -> None:
        node = find_func(parse(self.p["v1/worker/gpu/attn_utils.py"]), "build_attn_metadata")
        self.assertIn("da_fa_override", func_arg_names(node))
        # 默认值为 None：不传即原行为
        defaults = dict(zip([a.arg for a in node.args.args[-len(node.args.defaults):]], node.args.defaults))
        self.assertIn("da_fa_override", defaults)
        self.assertIsInstance(defaults["da_fa_override"], ast.Constant)
        self.assertIsNone(defaults["da_fa_override"].value)

    def test_only_fa_group_metadata_is_replaced(self) -> None:
        patched = find_func(parse(self.p["v1/worker/gpu/attn_utils.py"]), "build_attn_metadata")
        pristine = find_func(parse(self.orig["v1/worker/gpu/attn_utils.py"]), "build_attn_metadata")
        new_targets = assigned_names(patched) - assigned_names(pristine)
        self.assertEqual(
            new_targets,
            {"da_group_active", "group_seq_lens", "group_seq_lens_cpu_upper_bound", "group_max_seq_len"},
            f"补丁引入了非预期赋值目标：{sorted(new_targets)}",
        )
        # 写入侧与调度侧字段一律不得被补丁改名/改写
        for forbidden in ("slot_mapping", "query_start_loc_gpu", "query_start_loc_cpu", "dcp_local_seq_lens"):
            with self.subTest(name=forbidden):
                self.assertNotIn(forbidden, new_targets)

    def test_override_is_gated_by_group_index_and_capture(self) -> None:
        src = self.p["v1/worker/gpu/attn_utils.py"].read_text()
        self.assertIn("i == da_fa_override.group_index", src)
        self.assertIn("not for_cudagraph_capture", src)

    def test_mamba_hybrid_forwards_override(self) -> None:
        node = find_func(
            parse(self.p["v1/worker/gpu/model_states/mamba_hybrid.py"]),
            "prepare_attn",
            cls="MambaHybridModelState",
        )
        self.assertIn("da_fa_override", func_arg_names(node))
        calls = [
            c
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and ((isinstance(c.func, ast.Name) and c.func.id == "build_attn_metadata")
                 or (isinstance(c.func, ast.Attribute) and c.func.attr == "build_attn_metadata"))
        ]
        self.assertTrue(calls, "prepare_attn 内没有 build_attn_metadata 调用")
        for call in calls:
            kws = {k.arg for k in call.keywords}
            self.assertIn("da_fa_override", kws, "prepare_attn 未把覆写转传给 build_attn_metadata")

    def test_model_runner_builds_override_only_when_plans(self) -> None:
        node = find_func(parse(self.p["v1/worker/gpu/model_runner.py"]), "execute_model", cls="GPUModelRunner")
        src = ast.unparse(node)
        self.assertIn("fa_override_for_step", src)
        self.assertIn("da_step_plans", src)
        self.assertIn("not dummy_run", src)
        self.assertIn("**prepare_attn_kwargs", src)

    def test_engine_core_hooks_present(self) -> None:
        tree = parse(self.p["v1/engine/core.py"])
        step_src = ast.unparse(find_func(tree, "step", cls="EngineCore"))
        self.assertIn("attach_plans", step_src)
        self.assertIn("on_step_outputs", step_src)
        batch_src = ast.unparse(find_func(tree, "step_with_batch_queue", cls="EngineCore"))
        self.assertIn(
            "refuse_unsupported_step",
            batch_src,
            "批次队列路径缺少拒绝守卫（会静默退化为原版读取）",
        )

    def test_scheduler_output_has_plan_field(self) -> None:
        tree = parse(self.p["v1/core/sched/output.py"])
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SchedulerOutput")
        fields = {}
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields[stmt.target.id] = stmt.value
        self.assertIn("da_step_plans", fields)
        self.assertIsNone(getattr(fields["da_step_plans"], "value", "missing"))

    def test_new_modules_parse_and_stay_torch_free_on_engine_side(self) -> None:
        m = load_manifest()
        for item in m["new_files"]:
            with self.subTest(dest=item["dest"]):
                tree = parse(REPO / item["src"])
                self.assertIsInstance(tree, ast.Module)
        engine_path = REPO / "vllm-patch/files/vllm/v1/engine/attnview_engine.py"
        imported: set[str] = set()
        for node in ast.walk(parse(engine_path)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported, "引擎侧模块不得 import torch（跨进程边界）")
        self.assertNotIn("vllm", imported, "引擎侧模块不得 import vllm（跨进程边界）")
        self.assertIn("attnview", imported, "引擎侧模块应复用 attnview 纯逻辑包")


class DeployScriptSafetyTest(unittest.TestCase):
    """未部署状态下 verify/revert 必须是非破坏性的（本测试不调用 apply）。"""

    SAMPLES = None

    def setUp(self) -> None:
        if (PATCH_ROOT / "deployed.json").is_file():
            self.skipTest("当前处于部署状态：本测试只在未部署时断言非破坏性（避免意外撤销）")
        m = load_manifest()
        self.samples: list[Path] = []
        for rel in m["edits"]:
            for root in (PIN_PKG, INSTALL_PKG):
                target = root / rel
                if target.is_file():
                    self.samples.append(target)
        if not self.samples:
            self.skipTest("没有可抽样的目标文件")

    def _run(self, action: str) -> subprocess.CompletedProcess:
        env_python = sys.executable
        return subprocess.run(
            [env_python, str(REPO / "tools/p2-apply-patch.py"), action],
            cwd=REPO,
            capture_output=True,
            text=True,
        )

    def test_verify_and_revert_are_non_destructive_when_undeployed(self) -> None:
        before = {str(p): sha256(p) for p in self.samples}
        for action in ("verify", "revert"):
            with self.subTest(action=action):
                result = self._run(action)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = {str(p): sha256(p) for p in self.samples}
        self.assertEqual(before, after, "未部署状态下 verify/revert 改动了目标文件")


if __name__ == "__main__":
    unittest.main()
