# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

> `AGENTS.md` above is the upstream vLLM contribution policy (mandatory). The rest of
> this file documents what is specific to **this Windows fork** and is not derivable
> from the code alone.

## What this repository is

This is **vllm-windows** — vLLM packaged so it builds and runs natively on Windows
(MSVC + CUDA), not just Linux/WSL. It is a fork-of-a-fork:

- `origin` → `github.com/davidcforbes/vllm-windows` (this working fork)
- `upstream` → `github.com/SystemPanic/vllm-windows` (the Windows port project)
- The real vLLM (`github.com/vllm-project/vllm`) is periodically merged into `main`.

### The two-branch model (most important thing to know)

The Windows patches do **not** live on `main`.

- **`main`** tracks `vllm-project/vllm` `main` *verbatim*. It contains no Windows build
  support — `setup.py` here will warn that only Linux/macOS are supported, and the
  Windows helper scripts (`fix_cuda_13_align.py`, `fix_cutlass_msvc.py`,
  `requirements/windows.txt`) do **not exist** on this branch. Use `main` only to track
  upstream and stage merges.
- **`vllm-for-windows`** is the actual product branch. ~60 files differ from `main`:
  build system, MSVC C++ compatibility fixes, and Windows multiprocessing/IPC patches.
  The README's build instructions assume you cloned this branch.

When asked to "build on Windows", "fix a Windows issue", or "port an upstream change",
work happens on `vllm-for-windows`. Inspect the delta with:

```bash
git fetch upstream vllm-for-windows
git diff --stat main...upstream/vllm-for-windows
```

## Building on Windows (from `vllm-for-windows`)

Requires Visual Studio 2019+ (for the x64 `vcvarsall.bat` environment) and a CUDA
toolkit (auto-detected via PATH or `CUDA_ROOT`/`CUDA_HOME`/`CUDA_PATH`). All steps run
in `cmd.exe`, not PowerShell, after `vcvarsall.bat x64`:

```bat
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10                          REM parallel compile jobs

REM CUDA 13.0–13.2 only: patch the 128-byte CUtensorMap alignment MSVC can't pass by
REM value. Run ONCE from an elevated cmd before building (no-op on CUDA 13.3+):
python fix_cuda_13_align.py

pip install torch==2.11+cu130 torchaudio==2.11+cu130 torchvision==0.26.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/build/cuda.txt
pip install -r requirements/cuda.txt
pip install -r requirements/windows.txt   REM winloop, triton-windows, xformers, portalocker
pip install . --no-build-isolation -vvv
```

To consume a release instead of building, install the prebuilt wheel from the
[releases page](https://github.com/SystemPanic/vllm-windows/releases/latest) — the
wheel's Python/Torch/CUDA versions must match your environment exactly.

### Build gotchas baked into the fork

- **FlashAttention 3 is disabled** on Windows and WSL2 (`setup.py` sets
  `VLLM_DISABLE_FA3_BUILD=1`): it crashes MSVC and Hopper isn't available on Windows
  anyway. Force it with `set VLLM_FORCE_FA3_WINDOWS_BUILD=1` only if you know why.
- **ccache is preferred over sccache** on Windows (more stable, cache is clearable).
- `fix_cutlass_msvc.py <cutlass_dir>` patches CUTLASS's `platform.h` so the C++17
  guard also fires under `_MSC_VER`. `setup.py` runs the relevant patches; the
  `fix_cuda_*` script is the one manual step.
- Optional libs are opt-in via env vars: `USE_CUDNN`/`USE_CUDSS`/`USE_CUSPARSELT`
  with matching `*_INCLUDE_PATH` / `*_LIBRARY_PATH`.

## Where the Windows patches live (anatomy of the diff)

When porting an upstream change or debugging a Windows-only failure, these are the
categories that carry fork-specific edits:

- **Build** — `setup.py` (`IS_WINDOWS`/`IS_WSL` branches, CMake path normalization),
  `cmake/external_projects/*` (qutlass, triton_kernels, vllm_flash_attn),
  `requirements/{windows.txt,build/cuda.txt,cuda.txt}`.
- **MSVC C++/CUDA compatibility** — almost everything under `csrc/libtorch_stable/`
  plus `csrc/quantization/`, `csrc/moe/`, `csrc/spinloop.cpp`, `csrc/core/math.hpp`.
  These are alignment/intrinsic/`__attribute__` fixes for the MSVC compiler; keep them
  when merging upstream kernel changes.
- **Windows multiprocessing & IPC** (the runtime patches, where most Windows *runtime*
  bugs originate) — `vllm/distributed/parallel_state.py`,
  `vllm/distributed/device_communicators/shm_broadcast.py`,
  `vllm/v1/executor/multiproc_executor.py`, `vllm/v1/{utils,engine/utils}.py`,
  `vllm/utils/system_utils.py`, and `vllm/entrypoints/cli/{serve,launch,openai}.py` /
  `grpc_server.py` / `openai/api_server.py`. Windows uses spawn (not fork) and lacks
  POSIX shared-memory/signals, so these adapt process startup, IPC, and socket handling.

Multi-GPU (tensor/pipeline parallel) needs a Windows NCCL build pointed at via
`VLLM_NCCL_SO_PATH=...\nccl.dll`.

## Dev workflow on this fork

`AGENTS.md` prescribes a POSIX `uv` + `.venv/bin/python` + `pre-commit` workflow. That
is the upstream contribution standard and is correct for **lint and Python-only test
runs**. On Windows native builds the install commands differ (use the Windows build
recipe above, not `VLLM_USE_PRECOMPILED=1 uv pip install -e .`). Lint/type rules
(ruff, mypy, 88-char lines, Google-style docstrings) and the contribution policy in
`AGENTS.md` apply unchanged.

```bash
# Single test file (works cross-platform once vLLM is importable):
.venv/Scripts/python -m pytest tests/path/to/test_file.py -v   # Windows venv layout
pre-commit run --all-files                                     # lint everything
```

## vLLM architecture (orientation)

Engine internals live under `vllm/`. The current generation is **v1**
(`vllm/v1/`): the engine core, scheduler, KV-cache manager, and the
executor/worker stack (`v1/executor/`, `v1/worker/`) that the Windows multiprocessing
patches target. Request entry points are in `vllm/entrypoints/` (OpenAI-compatible
server in `entrypoints/openai/`, CLI in `entrypoints/cli/`, plus a Rust frontend under
`rust/`). Model definitions are in `vllm/model_executor/` and `vllm/models/`; C++/CUDA
kernels are in `csrc/` (built via `cmake/`); hardware abstraction is in
`vllm/platforms/`. For deep architecture, see `docs/design/arch_overview.md`.
