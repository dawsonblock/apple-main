from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Optional

import numpy as np
from numpy.typing import DTypeLike

_MLX_IMPORT_ERROR: Exception | None = None

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except Exception as exc:  # pragma: no cover - environment dependent
    mx = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    HAS_MLX = False
    _MLX_IMPORT_ERROR = exc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RFSN_MLX")


@dataclass
class RFSNConfig:
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32

    num_subspaces: int = 8
    pq_bits: int = 8
    subspace_dim: int = 16

    num_rvq_layers: int = 4
    rvq_codebook_size: int = 4096
    rvq_sparsity_threshold: float = 0.005

    hot_capacity: int = 8192
    warm_capacity: int = 65536
    cold_capacity: int = 2_000_000

    block_size_seq: int = 128

    disk_cache_dir: str = "./rfsn_disk_cache"
    max_open_files: int = 8
    prefetch_throttle_s: float = 0.25

    cpu_threads: int = 4
    hot_cache_dtype: str = "float16"
    rvq_max_active: int = -1

    def __post_init__(self) -> None:
        expected_dim = self.num_subspaces * self.subspace_dim
        if expected_dim != self.head_dim:
            raise ValueError(
                f"head_dim must equal num_subspaces * subspace_dim; got "
                f"{self.head_dim} vs {self.num_subspaces}*{self.subspace_dim}={expected_dim}"
            )


def _require_mlx() -> None:
    if not HAS_MLX:
        raise RuntimeError(
            "MLX is required for this module. Install mlx on Apple Silicon to use it. "
            f"Original import error: {_MLX_IMPORT_ERROR}"
        )


def _mx_dtype(name: str):
    _require_mlx()
    return getattr(mx, name)


def _resolve_cache_storage_dtype(name: str):
    _require_mlx()
    if not hasattr(mx, name):
        raise RuntimeError(
            f"Requested cache dtype {name!r} is not available in this MLX runtime"
        )
    return getattr(mx, name)


def _dtype_nbytes(name: str) -> int:
    if name.startswith("float8"):
        return 1
    if name in {"bool_", "uint8", "int8"}:
        return 1
    if name in {"float16", "bfloat16", "uint16", "int16"}:
        return 2
    if name in {"float32", "uint32", "int32"}:
        return 4
    raise ValueError(f"Unsupported dtype name for byte accounting: {name}")


def _mx_to_np(x: Any, dtype: DTypeLike | None = None) -> np.ndarray:
    arr = np.array(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _np_to_mx(x: np.ndarray, dtype=None):
    _require_mlx()
    arr = mx.array(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _stable_softmax_np(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    max_scores = np.max(scores, axis=axis, keepdims=True)
    shifted = scores - max_scores
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores, axis=axis, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_scores / denom


def _force_eval(*tensors) -> None:
    _require_mlx()
    if tensors:
        mx.eval(*tensors)


def _stable_softmax_mx(scores, axis: int = -1):
    max_scores = mx.max(scores, axis=axis, keepdims=True)
    shifted = scores - max_scores
    exp_scores = mx.exp(shifted)
    denom = mx.sum(exp_scores, axis=axis, keepdims=True)
    denom = mx.where(denom == 0.0, mx.ones_like(denom), denom)
    return exp_scores / denom
