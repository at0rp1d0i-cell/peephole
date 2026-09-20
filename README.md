# peephole

**Declarative attention read views for vLLM** — the model declares which KV regions it reads at each
decode step; the engine reads only those.

> **Status: work in progress.** Construction, correctness checks and measurement are still running; no
> speedup is claimed here. Both positive and negative results are committed as raw artifacts under
> [`evidence/`](evidence/) with commentary under [`reports/`](reports/). Claims that are not
> reproducible from those artifacts should be treated as unverified.

## What this is

On a pinned vLLM revision, this project adds a request-level **attention read view** runtime: the
model emits declarative tags in its own generation stream, the runtime parses them incrementally and
turns them into a real paged-KV read view for the next decode step.

- **Declared, not inferred.** The model writes `<global>` / `<focus magic_chunks="K">…</focus>` /
  `<local>`; the runtime follows a frozen state machine (parse on step *t*, take effect on step
  *t+1*). Nothing is decided from observed attention statistics.
- **Reads shrink, storage does not.** Only *reads* are narrowed: canonical block allocation, reference
  counts and write positions are untouched, and the full KV stays resident. The next append still goes
  to the canonical write position, not to a compacted one.
- **Three regions are always visible:** attention sink, the local window (question / instruction), and
  the tokens already generated. `focus` adds the declared segments; `local` is only those three.
- **Quantities that must not be conflated:** logical history length, visible token count, and number of
  blocks read are three different numbers and are reported as such.
- **Recovery is not a rollback.** Returning to `global` restores full reads from that point on; it
  neither undoes earlier reads nor rewrites prior output.
- **Malformed declarations do not silently degrade:** the current mode is kept, the event is traced and
  counted as a failure. There is no implicit fallback to global (an explicit switch may force one, and
  those samples are reported separately).
- **The public surface is the result.** The server side wraps the internal generation stream: control
  tags and internal analysis are not forwarded to callers (first version: non-streaming, mechanical
  extraction of the answer region). Treating the tag syntax as reserved means text that emits those
  literal tags is a documented compatibility limit, not a solved isolation guarantee.

## The name

Literal: the model moves a peephole across its own paged KV cache and the engine reads only inside that
window — the way a *peephole optimizer* rewrites a handful of instructions instead of the whole
program. In Chinese, the closest classical phrasing is 以管窥天（《庄子·秋水》）.

## Layout

| Path | Contents |
| --- | --- |
| `src/attnview/` | CPU-side core: segmentation, incremental tag parsing, protocol state, read-view construction, view → physical read-table conversion, per-step plan |
| `vllm-patch/` | Version-locked patch: modified upstream files, the two files added inside the vLLM package, `manifest.json` with per-file SHA-256 and the pinned revision, and the deployment journal |
| `tests/` | CPU-runnable tests; they reach the core through `src/` on `sys.path` |
| `tools/` | Patch generation/deployment, in-kernel calibration drivers, probes and oracles |
| `reports/`, `evidence/` | Reports and raw artifacts (per-request traces, calibration output, timing runs) with an evidence index |
| `configs/` | Experiment configurations |
| `env.sh`, `install-runtime.sh`, `verify-runtime.sh`, `setup-local-cuda.sh` | Environment definition, installation, and the runtime verification entry point |

## Reproducing

```bash
bash install-runtime.sh          # environment: pinned runtimes, dependency freeze, vLLM checkout
bash verify-runtime.sh           # runtime checks; console output kept
python tools/p2-gen-patch.py     # regenerate the patch manifest from sources
python tools/p2-apply-patch.py apply     # deploy (pre-checks, journal, expected digests)
python tools/p2-apply-patch.py verify    # verify deployed digests
python tools/p2-apply-patch.py revert    # transactional rollback
python -m pytest tests/ -q       # CPU-side test suite
```

Notes: `install-runtime.sh` and `setup-local-cuda.sh` were written for the container environment in
which the experiments were run (mirror-based package access, container-local CUDA toolchain, large
model cache on a separate data disk); outside that environment expect to adjust paths and mirrors.
Environment variables and directory locations are defined in `env.sh`, which is the source of truth.

Freeze authority: the repository-root `requirements.freeze.txt` is the authoritative dependency freeze
for the current environment; `requirements.freeze.stage-01.txt` and `reports/requirements.freeze.txt`
are historical snapshots kept as records (197 / 197 / 202 lines respectively, with differing
contents).

## Not supported (first version)

Single GPU, one primary model/backend, text only, single requests and limited mixed batching. The
following are **out of scope and must not be described as supported**: multiple GPUs, 1M-token
context, training, KV offloading, quantization, speculative decoding, a second model architecture, a
web product, prefix caching, state semantics under request preemption, and view updates under CUDA
Graph `FULL`. The current path commits to FlashAttention only, and the patched engine rejects
unsupported configurations at start-up (asynchronous scheduling, non-eager / graph modes, prefix
caching, speculative or MTP) instead of degrading quietly.

## Evidence

Raw per-request and per-run artifacts are kept, not summarized away: `evidence/` holds traces, JSONL /
JSON outputs, logs and the environment probes; `reports/` holds the interpretation, including the
cases where the mechanism did not pay off and the differences between the protocol arm and the
baseline. A claim of speedup requires a quality-constrained measurement, not a drop in counters.

## Provenance

Paths of the form `material/attnview/...` that appear in the records denote a **private materials
repository snapshot that is not distributed with this repository** (`material/` is excluded from the
published tree). Reference files that a shipped tool actually needs are copied into
[`docs/references/`](docs/references/) — currently `da-paper-extract.md` (paper appendix B and F
excerpt, carrying its own source attribution), the input of `tools/p1cpu-check-prompt-fidelity.py`.
Occurrences of the private paths inside `evidence/` and `reports/` (command lines, logs, manifests)
are historical records and are left byte-identical on purpose: they document how the artifacts were
produced, they do not resolve inside a public checkout.

## Design documents

The plan, protocol contract, read-view specification and the open-gap ledger currently live in the
project's materials repository and are being merged into `docs/` here as part of publication.

## License and attribution

Apache-2.0 — see [`LICENSE`](LICENSE). Modified vLLM files and files added inside the vLLM package,
the pinned revision they derive from, and the third-party references are declared in
[`NOTICE`](NOTICE).

## Working with agents

Contributors and agents should start from the material that ships in this repository:
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for licensing and third-party attribution, and
[`reports/`](reports/) for the current interpretation, the open gaps and the evidence index. The
verification entry point is `bash verify-runtime.sh` (runtime checks) followed by
`python -m pytest tests/ -q` (CPU-side tests); run both before treating a change as done.
