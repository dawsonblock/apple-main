"""
RFSN v10.2 — Unified Mac launcher
=================================

This launcher detects available local backends and runs a small smoke path.
It reports backend availability only. It does not claim ANE execution.

Priority order:
1. MLX available
2. PyTorch MPS available
3. PyTorch CPU available
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RFSN_LAUNCHER")


@dataclass
class BackendInfo:
    name: str
    priority: int
    device_str: str
    is_available: bool
    reason: str


def detect_backends() -> List[BackendInfo]:
    backends: List[BackendInfo] = []

    mlx_available = False
    mlx_reason = "MLX not installed"
    try:
        import mlx.core as mx

        _ = mx.zeros((1,), dtype=mx.float16)
        mlx_available = True
        mlx_reason = "MLX available"
    except Exception as exc:  # pragma: no cover - environment dependent
        mlx_reason = f"MLX unavailable: {exc}"
    backends.append(BackendInfo("mlx", 1, "mlx", mlx_available, mlx_reason))

    mps_available = False
    mps_reason = "PyTorch not installed"
    try:
        import torch

        if torch.backends.mps.is_available():
            _ = torch.zeros((1,), device="mps")
            mps_available = True
            mps_reason = f"PyTorch MPS available (torch {torch.__version__})"
        else:
            mps_reason = f"PyTorch installed but MPS unavailable (torch {torch.__version__})"
    except Exception as exc:  # pragma: no cover - environment dependent
        mps_reason = f"PyTorch MPS unavailable: {exc}"
    backends.append(BackendInfo("pytorch_mps", 2, "mps", mps_available, mps_reason))

    cpu_available = False
    cpu_reason = "PyTorch not installed"
    try:
        import torch

        _ = torch.zeros((1,), device="cpu")
        cpu_available = True
        cpu_reason = f"PyTorch CPU available (torch {torch.__version__})"
    except Exception as exc:  # pragma: no cover - environment dependent
        cpu_reason = f"PyTorch CPU unavailable: {exc}"
    backends.append(BackendInfo("pytorch_cpu", 3, "cpu", cpu_available, cpu_reason))

    return backends


def select_best_backend(backends: List[BackendInfo]) -> BackendInfo:
    available = [b for b in backends if b.is_available]
    if not available:
        raise RuntimeError("No supported backend is available")
    return sorted(available, key=lambda item: item.priority)[0]


def run_mlx() -> None:
    try:
        from rfsn_v10_mlx_ane_complete import run_tests
    except ImportError as exc:
        logger.error("Failed to import MLX prototype module: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Running MLX prototype test suite")
    ok = run_tests()
    if not ok:
        raise SystemExit(1)


def run_pytorch_fallback() -> None:
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Running PyTorch fallback smoke test on %s", device)

    batch = 2
    heads = 4
    dim = 128
    seq = 10

    q = torch.randn(batch, heads, dim, device=device, dtype=torch.float32)
    k = torch.randn(seq, heads, dim, device=device, dtype=torch.float32)
    v = torch.randn(seq, heads, dim, device=device, dtype=torch.float32)

    scale = dim ** -0.5
    scores = torch.einsum("bhd,shd->bhs", q, k) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,shd->bhd", weights, v)

    assert out.shape == (batch, heads, dim)
    assert torch.isfinite(out).all()

    if device.type == "mps":
        torch.mps.synchronize()
    logger.info("PyTorch fallback smoke test passed")


def main() -> None:
    logger.info("=" * 64)
    logger.info("RFSN v10.2 unified launcher")
    logger.info("Backend selection proves availability only, not ANE execution")
    logger.info("=" * 64)

    try:
        backends = detect_backends()
        for backend in backends:
            status = "AVAILABLE" if backend.is_available else "UNAVAILABLE"
            logger.info("%-14s %-11s %s", backend.name, status, backend.reason)

        best = select_best_backend(backends)
        logger.info("Selected backend: %s", best.name)

        start = time.time()
        if best.name == "mlx":
            run_mlx()
        else:
            run_pytorch_fallback()
        elapsed_ms = (time.time() - start) * 1000.0
        logger.info("Completed successfully in %.1f ms", elapsed_ms)
    except Exception as exc:
        logger.error("Launcher failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
