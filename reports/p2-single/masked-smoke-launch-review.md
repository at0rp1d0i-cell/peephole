# First masked diagnostic: focused launch review

User directly delegated launch review to the current implementer on 2026-09-21. The old local READY dependency is superseded; the old OMP development session remains stopped.

Source commit: `bfa728a`. All five implementation/configuration/test paths were explicitly staged, committed, and checked with `git show --stat` and an empty task-path diff before testing. User publication work independently advanced HEAD from `7b0596f` to `57f1871` before this commit; no user changes were included.

## Launch decision: ready for one diagnostic

- CPU: 85 tests passed. Raw command, HEAD, exit code and log: `evidence/p3-masked-smoke/cpu-20260920T170715Z/`. This covers old calibration hooks and affected smoke suites. Real capture begin/end control flow is exercised with FA group 1, noncontiguous physical blocks, stale previous execute state, and subsequent buffer mutation. Fifteen independent corruption controls fail. The real `_run` is exercised with a stub model/engine: normal evidence returns 0; corrupted length returns 1 and cleanup still executes. This is CPU integration evidence, not GPU evidence.
- Independent expected modes/blocks use declared token segments and original renderer spans. Every actual FA layer gets owned per-forward snapshots of canonical allocation, read length/physical row, original positions and current forward-context slots. Missing request/layer/step/trace evidence fails. Restricted override records and enforce-global marks have separate criteria.
- Old arms retain their inputs and canonical-view assertions. Masked alone uses restricted trace and exact structural checks. Original has no patch; disabled has no payload; global has payload with enforce_global=True; masked uses False.
- Deployment import closure: the manifest's seven attnview modules cover the two production adapters and their imports. Host diagnostic modules load from repository `src/`; no reference replacement hook is installed. Generator `--check` passes. Initial deployment has no journal; verify's undeployed exit 0 is not treated as a deployed check. The transaction will apply, then verify, then assert deployed state before model launch, and revert in finally.
- Configuration is `configs/p2-masked-smoke/diagnostic.json`: real 7834-token fixture, 29 samples/28 consumed generated tokens, representative decode 6/7/20/25. Pre-launch input hashes and exact script/trajectory comparison are in `launch-inputs.json` beside the CPU log.
- Pin remains vLLM `98dff2a81d747d1dba01a47f939f48c3526d4206`, model revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Driver uses BF16/TP1/FA2, eager, synchronous scheduling, max_num_seqs=1, max length and prefill budget 8192, prefix cache off; no speculative configuration. Startup/request/cleanup watchdogs remain 900/180/120 seconds.
- Machine snapshot: labops `20260921T005712+0800-07ebdb54ec27`, one RTX PRO 6000 Blackwell 96GB, driver 580.95.05, 22 CPU cores quota, 110 GiB RAM. No GPU compute processes before launch. Preflight passes: 122.82 GiB free versus 3 GiB estimated output plus 5 GiB reserve. No prior labctl baseline exists, so no performance comparability claim.

## Command and boundary

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 "$ATTNVIEW_PYTHON" tools/p2-masked-smoke-run.py \
  --out evidence/p3-masked-smoke/diagnostic-20260921-first
```

Always use a fresh output directory. The transaction records exact commands, phase exit codes, PID, HEAD and configuration hash. The driver snapshots all attnview source modules and records input hashes. Prefill stores K/V only once per capture stream; decode stores incremental K/V and only four selected Q/out points. A merged archive may duplicate stored evidence on disk but does not recopy historical KV from GPU.

This is a synchronized diagnostic, not a timing measurement. Structure, capture ownership, finite outputs and failures are observable. Full independent masked attention/logits execution, model quality and net benefit remain unaccepted. No new numeric thresholds or P2 acceptance are introduced. Stop after the diagnostic result audit and closeout.

## First attempt and narrowly scoped retry

The first attempt at `ba05322` applied/verified successfully, loaded FA2 with block size 784, then failed before decode 6 FA forward: `calibration_note_override` accessed `_version` on a PyTorch inference tensor. The exact error is independently reproducible on CPU with `torch.inference_mode()`. `diagnostic-20260921-first/run/error.txt` and forwards 1–6 are preserved. Revert returned 0 for all 27 targets; no remaining deployment journal or GPU process; GPU returned to 0 MiB. The in-process lifecycle cleanup was not reached in this failed attempt; process exit and deployment rollback are the observed cleanup evidence.

The work order permits a narrow repair of the same diagnostic fault. The repair reports inference-tensor version as null/unobservable instead of claiming a counter exists. No candidate read/write arithmetic changes. The driver also preserves partial capture/engine trace and attempts bounded target cancellation on a request exception. A CPU regression calls the actual trace function inside inference mode; `_run` exception coverage verifies cancellation and failure evidence. After committing and running these affected checks, use fresh directory `diagnostic-20260921-retry1`; no third launch is planned. The first failure is not replaced or relabeled as success.
