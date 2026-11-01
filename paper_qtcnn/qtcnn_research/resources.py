"""Resource guards so paper experiments never starve the machine's other jobs.

This box has one RTX 3090 shared by many sessions, and 24 CPU cores shared with
several always-on services. GPU arbitration belongs to the ``gq`` scheduler:
submit GPU work as ``gq run -g <vram> -- <command>`` (or ``gq submit``) and
declare the resources honestly, since gq admits jobs on the declared figure.

What stays here is everything gq does not cover: :func:`claim_cpu` caps our
thread budget and drops priority, :func:`hide_gpu` makes CUDA invisible so a
CUDA-enabled torch cannot take VRAM the scheduler never granted, and
:func:`wait_for_cpu` yields while other people's jobs have the machine
oversubscribed. :func:`query_gpus` remains useful for reporting.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

DEFAULT_CPU_THREADS = 8
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def claim_cpu(threads: int = DEFAULT_CPU_THREADS, niceness: int = 10) -> int:
    """Pin this process to a modest CPU budget and drop its scheduling priority.

    Must run before numpy/torch import their thread pools to have full effect,
    hence the env vars rather than torch.set_num_threads alone.
    """
    threads = max(1, min(int(threads), os.cpu_count() or 1))
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(threads))
    with contextlib.suppress(OSError):
        os.nice(niceness)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # set_num_interop_threads raises if the pool is already initialised.
        pass
    return threads


def hide_gpu() -> None:
    """Make CUDA invisible, so a CUDA-enabled torch cannot grab the busy card."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_per_core() -> float:
    """1-minute load average divided by core count."""
    return os.getloadavg()[0] / max(1, os.cpu_count() or 1)


def wait_for_cpu(
    max_load_per_core: float = 1.2,
    poll_seconds: int = 60,
    max_wait_seconds: Optional[int] = 3600,
    log=print,
) -> bool:
    """Block while the machine is oversubscribed by someone else's jobs.

    This box regularly runs other people's training at a 1-minute load average
    above 50 on 24 cores, under which our own throughput collapses by two orders
    of magnitude and we would only make their jobs slower. Returns True if the
    machine became quiet, False if the wait budget ran out (caller may proceed
    at low priority anyway).
    """
    deadline = None if max_wait_seconds is None else time.time() + max_wait_seconds
    while True:
        load = load_per_core()
        if load <= max_load_per_core:
            return True
        if deadline is not None and time.time() >= deadline:
            log(f"[wait_for_cpu] still busy (load/core {load:.2f}); proceeding at low priority")
            return False
        log(f"[wait_for_cpu] load/core {load:.2f} > {max_load_per_core}; waiting {poll_seconds}s")
        time.sleep(poll_seconds)


# ----------------------------------------------------------------------------
# GPU queue
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuState:
    index: int
    used_mib: int
    total_mib: int
    utilisation: int
    compute_pids: List[int]

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


def query_gpus() -> List[GpuState]:
    """Read GPU occupancy via nvidia-smi; empty list when there is no GPU."""
    try:
        summary = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        procs = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    # nvidia-smi reports compute apps by UUID; index order matches the gpu query,
    # and this box has a single card, so map every compute pid onto each device.
    pids = [int(line.split(",")[1]) for line in procs.splitlines() if line.strip()]

    states: List[GpuState] = []
    for line in summary.splitlines():
        if not line.strip():
            continue
        idx, used, total, util = (part.strip() for part in line.split(","))
        states.append(
            GpuState(
                index=int(idx),
                used_mib=int(used),
                total_mib=int(total),
                utilisation=int(util),
                compute_pids=pids,
            )
        )
    return states


def gpu_is_free(
    state: GpuState,
    min_free_mib: int = 18000,
    max_utilisation: int = 25,
    ignore_pids: Optional[List[int]] = None,
) -> bool:
    """A card counts as free only if it is both idle and mostly unallocated.

    Desktop compositors keep a permanent slice of memory, which is why the
    default threshold is on *free* memory rather than on zero usage.
    """
    ignore = set(ignore_pids or [])
    foreign = [pid for pid in state.compute_pids if pid not in ignore and pid != os.getpid()]
    return not foreign and state.free_mib >= min_free_mib and state.utilisation <= max_utilisation


@contextlib.contextmanager
def gpu_queue(  # noqa: D401 -- superseded by the gq scheduler; see module docstring
    min_free_mib: int = 18000,
    max_utilisation: int = 25,
    poll_seconds: int = 120,
    max_wait_seconds: Optional[int] = None,
    lock_dir: Optional[Path] = None,
    log=print,
) -> Iterator[Optional[int]]:
    """Deprecated: use ``gq run -g <vram> -- ...`` instead.

    Kept only for offline diagnostics on a machine without gq. Doing our own
    polling alongside the scheduler would double-book the card, because gq
    admits jobs against declared VRAM rather than observed free memory.

    Yields ``None`` when no GPU exists or the wait budget runs out, so callers
    can fall back to CPU rather than fail.
    """
    log("[gpu_queue] DEPRECATED: submit GPU work with `gq run -g <vram> -- ...` instead")
    lock_dir = lock_dir or Path.home() / ".cache" / "qtcnn_paper"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "gpu.lock"

    deadline = None if max_wait_seconds is None else time.time() + max_wait_seconds
    lock_fd = None
    waited = 0

    try:
        while True:
            states = query_gpus()
            if not states:
                log("[gpu_queue] no GPU visible; running on CPU")
                yield None
                return

            free = [s for s in states if gpu_is_free(s, min_free_mib, max_utilisation)]
            if free:
                try:
                    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(lock_fd, f"{os.getpid()}\n".encode())
                except FileExistsError:
                    log("[gpu_queue] another paper run holds the GPU lock; waiting")
                else:
                    chosen = free[0]
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen.index)
                    log(
                        f"[gpu_queue] acquired GPU {chosen.index} "
                        f"({chosen.free_mib} MiB free) after {waited}s"
                    )
                    yield chosen.index
                    return
            else:
                busy = states[0]
                log(
                    f"[gpu_queue] GPU busy (util {busy.utilisation}%, "
                    f"{busy.free_mib} MiB free, pids={busy.compute_pids}); "
                    f"waited {waited}s"
                )

            if deadline is not None and time.time() >= deadline:
                log("[gpu_queue] wait budget exhausted; falling back to CPU")
                yield None
                return

            time.sleep(poll_seconds)
            waited += poll_seconds
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            with contextlib.suppress(OSError):
                lock_path.unlink()


def describe_environment() -> str:
    """One-block provenance dump for the reproducibility section of the paper."""
    import platform

    lines = [
        f"python      : {platform.python_version()}",
        f"platform    : {platform.platform()}",
        f"cpu_count   : {os.cpu_count()}",
        f"cpu_threads : {os.environ.get('OMP_NUM_THREADS', 'unset')}",
        f"load/core   : {load_per_core():.2f}",
    ]
    for module in ("numpy", "pandas", "torch", "pennylane", "sklearn", "lightgbm", "statsmodels"):
        try:
            mod = __import__(module)
            lines.append(f"{module:12s}: {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            lines.append(f"{module:12s}: not installed")
    for state in query_gpus():
        lines.append(
            f"gpu{state.index}        : {state.total_mib} MiB total, "
            f"{state.used_mib} MiB used, util {state.utilisation}%"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    claim_cpu()
    print(describe_environment())
