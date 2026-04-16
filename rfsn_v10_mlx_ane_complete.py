"""
RFSN v10.2 — MLX experimental prototype for tiered KV caching
===============================================================

This module is an experimental Apple-Silicon-oriented prototype built around
MLX arrays, dense PQ/RVQ codebooks, and a tiered KV cache.

What is implemented:
- product quantization encode/decode
- residual vector quantization on PQ residuals
- hot and warm KV tiers
- cold spill to .npz chunks
- dense reference attention over hot + reconstructed warm tiers
- deterministic correctness tests against a dense reference path

What is not implemented:
- verified ANE execution
- quantized-codebook fast paths
- cold-tier read attention during inference
- serving/runtime integration beyond local tests

The code favors correctness and explicit behavior over speculative acceleration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


def _mx_to_np(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
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


def dense_attention_reference_np(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    q: [B, H, D]
    k: [S, H, D]
    v: [S, H, D]
    returns: [B, H, D]
    """
    if k.shape[0] == 0:
        return np.zeros_like(q, dtype=np.float32)
    scale = k.shape[-1] ** -0.5
    scores = np.einsum("bhd,shd->bhs", q.astype(np.float32), k.astype(np.float32)) * scale
    weights = _stable_softmax_np(scores, axis=-1)
    out = np.einsum("bhs,shd->bhd", weights, v.astype(np.float32))
    return out.astype(np.float32)


def _streaming_attention_update_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    running_max: np.ndarray,
    running_sum: np.ndarray,
    running_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k.shape[0] == 0:
        return running_max, running_sum, running_out

    scale = k.shape[-1] ** -0.5
    scores = np.einsum("bhd,shd->bhs", q.astype(np.float32), k.astype(np.float32)) * scale
    chunk_max = np.max(scores, axis=-1)
    new_max = np.maximum(running_max, chunk_max)

    prev_rescale = np.exp(running_max - new_max)
    chunk_weights = np.exp(scores - new_max[:, :, None])

    running_sum = running_sum * prev_rescale + np.sum(chunk_weights, axis=-1)
    running_out = (
        running_out * prev_rescale[:, :, None]
        + np.einsum("bhs,shd->bhd", chunk_weights, v.astype(np.float32))
    )
    return new_max, running_sum, running_out


def _pq_decode_np(
    codes_np: np.ndarray,
    codebooks_np: np.ndarray,
    num_subspaces: int,
    subspace_dim: int,
) -> np.ndarray:
    n = codes_np.shape[0]
    out = np.zeros((n, num_subspaces * subspace_dim), dtype=np.float32)
    codes_int = codes_np.astype(np.int32, copy=False)

    for sub in range(num_subspaces):
        s0 = sub * subspace_dim
        s1 = s0 + subspace_dim
        out[:, s0:s1] = codebooks_np[sub][codes_int[:, sub]]

    return out


def _rvq_decode_correction_np(
    total_rows: int,
    rvq_codes_np: np.ndarray,
    rvq_offsets_np: np.ndarray,
    codebooks_np: np.ndarray,
    head_dim: int,
) -> np.ndarray:
    out = np.zeros((total_rows, head_dim), dtype=np.float32)
    if rvq_offsets_np.size == 0 or codebooks_np.shape[0] == 0:
        return out

    if np.any(rvq_offsets_np < 0) or np.any(rvq_offsets_np >= total_rows):
        raise ValueError("RVQ offsets must be valid row indices")

    codes_int = rvq_codes_np.astype(np.int32, copy=False)
    for layer in range(codebooks_np.shape[0]):
        out[rvq_offsets_np] += codebooks_np[layer][codes_int[:, layer]]

    return out


def _hybrid_decode_np(
    pq_codes_np: np.ndarray,
    rvq_codes_np: np.ndarray,
    rvq_offsets_np: np.ndarray,
    pq_codebooks_np: np.ndarray,
    rvq_codebooks_np: np.ndarray,
    num_subspaces: int,
    subspace_dim: int,
    head_dim: int,
) -> np.ndarray:
    pq_recon = _pq_decode_np(pq_codes_np, pq_codebooks_np, num_subspaces, subspace_dim)
    correction = _rvq_decode_correction_np(
        pq_codes_np.shape[0],
        rvq_codes_np,
        rvq_offsets_np,
        rvq_codebooks_np,
        head_dim,
    )
    return pq_recon + correction


class ProductQuantizerMLX(nn.Module):
    def __init__(self, config: RFSNConfig):
        _require_mlx()
        super().__init__()
        self.config = config
        self.num_subspaces = config.num_subspaces
        self.subspace_dim = config.subspace_dim
        self.codebook_size = 1 << config.pq_bits

        scale = (2.0 / self.subspace_dim) ** 0.5
        codebooks = np.random.randn(
            self.num_subspaces, self.codebook_size, self.subspace_dim
        ).astype(np.float32) * scale
        self.codebooks = _np_to_mx(codebooks, dtype=_mx_dtype("float16"))

    def quantize(self, vectors) -> Tuple[Any, Any]:
        vectors_np = _mx_to_np(vectors, np.float32)
        n, d = vectors_np.shape
        if d != self.num_subspaces * self.subspace_dim:
            raise ValueError(f"Expected dim {self.num_subspaces * self.subspace_dim}, got {d}")

        codebooks_np = _mx_to_np(self.codebooks, np.float32)
        codes = np.zeros((n, self.num_subspaces), dtype=np.uint8)
        reconstructed = np.zeros_like(vectors_np, dtype=np.float32)

        for sub in range(self.num_subspaces):
            s0 = sub * self.subspace_dim
            s1 = s0 + self.subspace_dim
            v_sub = vectors_np[:, s0:s1]
            cb = codebooks_np[sub]
            dists = np.sum((v_sub[:, None, :] - cb[None, :, :]) ** 2, axis=-1)
            idx = np.argmin(dists, axis=1).astype(np.uint8)
            codes[:, sub] = idx
            reconstructed[:, s0:s1] = cb[idx]

        residuals = vectors_np - reconstructed
        return _np_to_mx(codes, dtype=_mx_dtype("uint8")), _np_to_mx(residuals, dtype=vectors.dtype)

    def decode(self, codes):
        codes_np = _mx_to_np(codes, np.int32)
        n = codes_np.shape[0]
        out = np.zeros((n, self.num_subspaces * self.subspace_dim), dtype=np.float32)
        codebooks_np = _mx_to_np(self.codebooks, np.float32)

        for sub in range(self.num_subspaces):
            s0 = sub * self.subspace_dim
            s1 = s0 + self.subspace_dim
            out[:, s0:s1] = codebooks_np[sub][codes_np[:, sub]]

        return _np_to_mx(out, dtype=_mx_dtype("float16"))


class ResidualVQMLX(nn.Module):
    def __init__(self, config: RFSNConfig):
        _require_mlx()
        super().__init__()
        self.config = config
        self.num_layers = config.num_rvq_layers
        self.codebook_size = config.rvq_codebook_size
        self.head_dim = config.head_dim
        self.sparsity_threshold = config.rvq_sparsity_threshold

        scale = (2.0 / self.head_dim) ** 0.5
        codebooks = np.random.randn(
            self.num_layers, self.codebook_size, self.head_dim
        ).astype(np.float32) * scale
        self.codebooks = _np_to_mx(codebooks, dtype=_mx_dtype("float16"))

    def encode(self, residuals) -> Tuple[Any, Any, Any]:
        residuals_np = _mx_to_np(residuals, np.float32)
        norms = np.linalg.norm(residuals_np, axis=1)
        mask = norms > self.sparsity_threshold
        offsets = np.where(mask)[0].astype(np.int32)

        if offsets.size == 0:
            empty_codes = np.zeros((0, self.num_layers), dtype=np.uint16)
            return (
                _np_to_mx(empty_codes, dtype=_mx_dtype("uint16")),
                _np_to_mx(mask.astype(np.bool_), dtype=_mx_dtype("bool_")),
                _np_to_mx(offsets, dtype=_mx_dtype("int32")),
            )

        active = residuals_np[offsets].copy()
        codebooks_np = _mx_to_np(self.codebooks, np.float32)
        codes = np.zeros((offsets.shape[0], self.num_layers), dtype=np.uint16)

        for layer in range(self.num_layers):
            cb = codebooks_np[layer]
            dists = np.sum((active[:, None, :] - cb[None, :, :]) ** 2, axis=-1)
            idx = np.argmin(dists, axis=1).astype(np.uint16)
            codes[:, layer] = idx
            active -= cb[idx.astype(np.int32)]

        return (
            _np_to_mx(codes, dtype=_mx_dtype("uint16")),
            _np_to_mx(mask.astype(np.bool_), dtype=_mx_dtype("bool_")),
            _np_to_mx(offsets, dtype=_mx_dtype("int32")),
        )

    def decode_correction(self, total_rows: int, rvq_codes, rvq_offsets):
        codes_np = _mx_to_np(rvq_codes, np.int32)
        offsets_np = _mx_to_np(rvq_offsets, np.int32)
        out = np.zeros((total_rows, self.head_dim), dtype=np.float32)
        if offsets_np.size == 0:
            return _np_to_mx(out, dtype=_mx_dtype("float16"))

        if np.any(offsets_np < 0) or np.any(offsets_np >= total_rows):
            raise ValueError("RVQ offsets must be valid row indices")

        codebooks_np = _mx_to_np(self.codebooks, np.float32)
        for i, row_idx in enumerate(offsets_np.tolist()):
            correction = np.zeros((self.head_dim,), dtype=np.float32)
            for layer in range(self.num_layers):
                correction += codebooks_np[layer, codes_np[i, layer]]
            out[row_idx] = correction

        return _np_to_mx(out, dtype=_mx_dtype("float16"))


class HybridQuantizerMLX(nn.Module):
    def __init__(self, config: RFSNConfig):
        _require_mlx()
        super().__init__()
        self.config = config
        self.pq = ProductQuantizerMLX(config)
        self.rvq = ResidualVQMLX(config)

    def encode(self, vectors) -> Tuple[Any, Any, Any, Any]:
        pq_codes, residuals = self.pq.quantize(vectors)
        rvq_codes, rvq_mask, rvq_offsets = self.rvq.encode(residuals)
        return pq_codes, rvq_codes, rvq_mask, rvq_offsets

    def decode(self, pq_codes, rvq_codes, rvq_mask, rvq_offsets):
        del rvq_mask
        pq_recon = self.pq.decode(pq_codes).astype(_mx_dtype("float32"))
        correction = self.rvq.decode_correction(pq_codes.shape[0], rvq_codes, rvq_offsets).astype(_mx_dtype("float32"))
        return (pq_recon + correction).astype(_mx_dtype("float16"))


class RFSNHybridAttentionMLX(nn.Module):
    """Dense reference attention used by the cache for correctness tests."""

    def __init__(self, config: RFSNConfig):
        _require_mlx()
        super().__init__()
        self.config = config

    def __call__(self, q, keys, values):
        q_np = _mx_to_np(q, np.float32)
        k_np = _mx_to_np(keys, np.float32)
        v_np = _mx_to_np(values, np.float32)
        out = dense_attention_reference_np(q_np, k_np, v_np)
        return _np_to_mx(out, dtype=_mx_dtype("float16"))


class RFSNv10KVCacheMLX:
    def __init__(self, config: RFSNConfig, layer_idx: int = 0):
        _require_mlx()
        self.config = config
        self.layer_idx = layer_idx
        self.hot_capacity = config.hot_capacity
        self.warm_capacity = config.warm_capacity

        self.hot_keys = mx.zeros((0, config.num_heads, config.head_dim), dtype=_mx_dtype("float16"))
        self.hot_values = mx.zeros((0, config.num_heads, config.head_dim), dtype=_mx_dtype("float16"))

        self.warm_key_pq_codes = mx.zeros((0, config.num_heads, config.num_subspaces), dtype=_mx_dtype("uint8"))
        self.warm_key_rvq_codes = mx.zeros((0, config.num_rvq_layers), dtype=_mx_dtype("uint16"))
        self.warm_key_rvq_mask = mx.zeros((0, config.num_heads), dtype=_mx_dtype("bool_"))
        self.warm_key_rvq_offsets = mx.zeros((0,), dtype=_mx_dtype("int32"))

        self.warm_value_pq_codes = mx.zeros((0, config.num_heads, config.num_subspaces), dtype=_mx_dtype("uint8"))
        self.warm_value_rvq_codes = mx.zeros((0, config.num_rvq_layers), dtype=_mx_dtype("uint16"))
        self.warm_value_rvq_mask = mx.zeros((0, config.num_heads), dtype=_mx_dtype("bool_"))
        self.warm_value_rvq_offsets = mx.zeros((0,), dtype=_mx_dtype("int32"))

        self.cold_chunk_paths: List[Path] = []
        self.num_hot = 0
        self.num_warm = 0
        self.num_cold = 0
        self.total_tokens = 0
        self.attention = RFSNHybridAttentionMLX(config)

    @property
    def current_tier(self) -> str:
        if self.num_hot < self.hot_capacity:
            return "hot"
        if self.num_warm < self.warm_capacity:
            return "warm"
        return "cold"

    def update(self, new_keys, new_values, quantizer: HybridQuantizerMLX, disk_dir: Optional[Path] = None) -> None:
        if new_keys.shape != new_values.shape:
            raise ValueError("Keys and values must have identical shapes")

        start = 0
        total = new_keys.shape[0]

        hot_space = max(0, self.hot_capacity - self.num_hot)
        hot_take = min(total - start, hot_space)
        if hot_take > 0:
            self.hot_keys = mx.concatenate([self.hot_keys, new_keys[start:start + hot_take]], axis=0)
            self.hot_values = mx.concatenate([self.hot_values, new_values[start:start + hot_take]], axis=0)
            self.num_hot += hot_take
            start += hot_take

        warm_space = max(0, self.warm_capacity - self.num_warm)
        warm_take = min(total - start, warm_space)
        if warm_take > 0:
            self._add_to_warm(new_keys[start:start + warm_take], new_values[start:start + warm_take], quantizer)
            start += warm_take

        if start < total:
            self._add_to_cold(new_keys[start:], new_values[start:], quantizer, disk_dir)

        self.total_tokens = self.num_hot + self.num_warm + self.num_cold

    def _encode_warm_batch(self, batch, quantizer: HybridQuantizerMLX, base_token_offset: int) -> Tuple[Any, Any, Any, Any]:
        s, h, d = batch.shape
        flat = batch.reshape(s * h, d)
        pq_codes, rvq_codes, rvq_mask, rvq_offsets = quantizer.encode(flat)
        pq_codes_reshaped = pq_codes.reshape(s, h, self.config.num_subspaces)
        rvq_mask_reshaped = rvq_mask.reshape(s, h)
        shifted_offsets = rvq_offsets + int(base_token_offset * h)
        return pq_codes_reshaped, rvq_codes, rvq_mask_reshaped, shifted_offsets

    def _add_to_warm(self, keys, values, quantizer: HybridQuantizerMLX) -> None:
        s = keys.shape[0]
        base_offset = self.num_warm

        key_pq, key_rvq, key_mask, key_offsets = self._encode_warm_batch(keys, quantizer, base_offset)
        value_pq, value_rvq, value_mask, value_offsets = self._encode_warm_batch(values, quantizer, base_offset)

        self.warm_key_pq_codes = mx.concatenate([self.warm_key_pq_codes, key_pq], axis=0)
        self.warm_key_rvq_codes = mx.concatenate([self.warm_key_rvq_codes, key_rvq], axis=0)
        self.warm_key_rvq_mask = mx.concatenate([self.warm_key_rvq_mask, key_mask], axis=0)
        self.warm_key_rvq_offsets = mx.concatenate([self.warm_key_rvq_offsets, key_offsets], axis=0)

        self.warm_value_pq_codes = mx.concatenate([self.warm_value_pq_codes, value_pq], axis=0)
        self.warm_value_rvq_codes = mx.concatenate([self.warm_value_rvq_codes, value_rvq], axis=0)
        self.warm_value_rvq_mask = mx.concatenate([self.warm_value_rvq_mask, value_mask], axis=0)
        self.warm_value_rvq_offsets = mx.concatenate([self.warm_value_rvq_offsets, value_offsets], axis=0)

        self.num_warm += s

    def _add_to_cold(self, keys, values, quantizer: HybridQuantizerMLX, disk_dir: Optional[Path]) -> None:
        disk_dir = Path(self.config.disk_cache_dir) if disk_dir is None else Path(disk_dir)
        disk_dir.mkdir(parents=True, exist_ok=True)

        s, h, d = keys.shape
        key_flat = keys.reshape(s * h, d)
        value_flat = values.reshape(s * h, d)

        key_pq, key_rvq, key_mask, key_offsets = quantizer.encode(key_flat)
        value_pq, value_rvq, value_mask, value_offsets = quantizer.encode(value_flat)

        chunk_id = len(self.cold_chunk_paths)
        chunk_path = disk_dir / f"layer{self.layer_idx}_chunk{chunk_id}.npz"
        np.savez_compressed(
            chunk_path,
            seq_tokens=np.array([s], dtype=np.int32),
            num_heads=np.array([h], dtype=np.int32),
            head_dim=np.array([d], dtype=np.int32),
            key_pq_codes=_mx_to_np(key_pq, np.uint8),
            key_rvq_codes=_mx_to_np(key_rvq, np.uint16),
            key_rvq_mask=_mx_to_np(key_mask, np.bool_),
            key_rvq_offsets=_mx_to_np(key_offsets, np.int32),
            value_pq_codes=_mx_to_np(value_pq, np.uint8),
            value_rvq_codes=_mx_to_np(value_rvq, np.uint16),
            value_rvq_mask=_mx_to_np(value_mask, np.bool_),
            value_rvq_offsets=_mx_to_np(value_offsets, np.int32),
        )
        self.cold_chunk_paths.append(chunk_path)
        self.num_cold += s

    def load_cold_chunk(self, chunk_id: int) -> Dict[str, Any]:
        path = self.cold_chunk_paths[chunk_id]
        data = np.load(path)
        return {k: _np_to_mx(np.array(v)) for k, v in data.items()}

    def _validate_warm_block_token_ranges(
        self,
        warm_block_token_ranges: Optional[List[Tuple[int, int]]],
    ) -> None:
        if warm_block_token_ranges is None:
            return

        previous_end = 0
        for index, block_range in enumerate(warm_block_token_ranges):
            if len(block_range) != 2:
                raise ValueError(f"Warm block range {index} must contain exactly two bounds")

            start_token, end_token = block_range
            if start_token < 0 or end_token > self.num_warm:
                raise ValueError(f"Warm block range {index} is out of bounds for num_warm={self.num_warm}")
            if start_token >= end_token:
                raise ValueError(f"Warm block range {index} must have start < end")
            if index > 0 and start_token < previous_end:
                raise ValueError("Warm block token ranges must be sorted and non-overlapping")

            previous_end = end_token

    def _default_warm_block_token_ranges(self, block_size_tokens: int) -> List[Tuple[int, int]]:
        return [
            (start_token, min(self.num_warm, start_token + block_size_tokens))
            for start_token in range(0, self.num_warm, block_size_tokens)
        ]

    def _build_blockwise_numpy_decode_state(self, quantizer: HybridQuantizerMLX) -> Dict[str, Any]:
        return {
            "pq_codebooks": _mx_to_np(quantizer.pq.codebooks, np.float32),
            "rvq_codebooks": _mx_to_np(quantizer.rvq.codebooks, np.float32),
            "key": {
                "pq_codes": _mx_to_np(self.warm_key_pq_codes, np.uint8),
                "rvq_codes": _mx_to_np(self.warm_key_rvq_codes, np.int32),
                "rvq_offsets": _mx_to_np(self.warm_key_rvq_offsets, np.int32),
            },
            "value": {
                "pq_codes": _mx_to_np(self.warm_value_pq_codes, np.uint8),
                "rvq_codes": _mx_to_np(self.warm_value_rvq_codes, np.int32),
                "rvq_offsets": _mx_to_np(self.warm_value_rvq_offsets, np.int32),
            },
        }

    def _reconstruct_warm_component_block_numpy(
        self,
        component_state: Dict[str, np.ndarray],
        decode_state: Dict[str, Any],
        start_token: int,
        end_token: int,
    ) -> np.ndarray:
        if start_token < 0 or end_token < start_token or end_token > int(component_state["pq_codes"].shape[0]):
            raise ValueError("Warm block token range is invalid")

        token_count = end_token - start_token
        if token_count == 0:
            return np.zeros((0, self.config.num_heads, self.config.head_dim), dtype=np.float32)

        row_start = start_token * self.config.num_heads
        row_end = end_token * self.config.num_heads
        block_pq_codes = component_state["pq_codes"][start_token:end_token].reshape(
            token_count * self.config.num_heads,
            self.config.num_subspaces,
        )

        offsets_np = component_state["rvq_offsets"]
        left = int(np.searchsorted(offsets_np, row_start, side="left"))
        right = int(np.searchsorted(offsets_np, row_end, side="left"))
        local_offsets = offsets_np[left:right] - row_start
        if np.any(local_offsets < 0) or np.any(local_offsets >= token_count * self.config.num_heads):
            raise ValueError("Local RVQ offsets must map inside the reconstructed warm block")

        local_rvq_codes = component_state["rvq_codes"][left:right]
        decoded = _hybrid_decode_np(
            block_pq_codes,
            local_rvq_codes,
            local_offsets,
            decode_state["pq_codebooks"],
            decode_state["rvq_codebooks"],
            self.config.num_subspaces,
            self.config.subspace_dim,
            self.config.head_dim,
        )
        return decoded.reshape(token_count, self.config.num_heads, self.config.head_dim)

    def reconstruct_warm_keys(self, quantizer: HybridQuantizerMLX):
        if self.num_warm == 0:
            return mx.zeros((0, self.config.num_heads, self.config.head_dim), dtype=_mx_dtype("float16"))
        flat_codes = self.warm_key_pq_codes.reshape(self.num_warm * self.config.num_heads, self.config.num_subspaces)
        flat_mask = self.warm_key_rvq_mask.reshape(self.num_warm * self.config.num_heads)
        recon = quantizer.decode(flat_codes, self.warm_key_rvq_codes, flat_mask, self.warm_key_rvq_offsets)
        return recon.reshape(self.num_warm, self.config.num_heads, self.config.head_dim)

    def reconstruct_warm_values(self, quantizer: HybridQuantizerMLX):
        if self.num_warm == 0:
            return mx.zeros((0, self.config.num_heads, self.config.head_dim), dtype=_mx_dtype("float16"))
        flat_codes = self.warm_value_pq_codes.reshape(self.num_warm * self.config.num_heads, self.config.num_subspaces)
        flat_mask = self.warm_value_rvq_mask.reshape(self.num_warm * self.config.num_heads)
        recon = quantizer.decode(flat_codes, self.warm_value_rvq_codes, flat_mask, self.warm_value_rvq_offsets)
        return recon.reshape(self.num_warm, self.config.num_heads, self.config.head_dim)

    def _reconstruct_warm_component_block(
        self,
        pq_codes,
        rvq_codes,
        rvq_mask,
        rvq_offsets,
        quantizer: HybridQuantizerMLX,
        start_token: int,
        end_token: int,
    ):
        if start_token < 0 or end_token < start_token or end_token > int(pq_codes.shape[0]):
            raise ValueError("Warm block token range is invalid")

        token_count = end_token - start_token
        if token_count == 0:
            return mx.zeros((0, self.config.num_heads, self.config.head_dim), dtype=_mx_dtype("float16"))

        row_start = start_token * self.config.num_heads
        row_end = end_token * self.config.num_heads
        block_pq_codes = pq_codes[start_token:end_token].reshape(token_count * self.config.num_heads, self.config.num_subspaces)
        block_mask = rvq_mask[start_token:end_token].reshape(token_count * self.config.num_heads)

        offsets_np = _mx_to_np(rvq_offsets, np.int32)
        selected = (offsets_np >= row_start) & (offsets_np < row_end)
        local_offsets = offsets_np[selected] - row_start
        if np.any(local_offsets < 0) or np.any(local_offsets >= token_count * self.config.num_heads):
            raise ValueError("Local RVQ offsets must map inside the reconstructed warm block")

        rvq_codes_np = _mx_to_np(rvq_codes, np.uint16)
        local_rvq_codes = rvq_codes_np[selected]
        decoded = quantizer.decode(
            block_pq_codes,
            _np_to_mx(local_rvq_codes, dtype=_mx_dtype("uint16")),
            block_mask,
            _np_to_mx(local_offsets, dtype=_mx_dtype("int32")),
        )
        return decoded.reshape(token_count, self.config.num_heads, self.config.head_dim)

    def reconstruct_warm_key_block(self, quantizer: HybridQuantizerMLX, start_token: int, end_token: int):
        return self._reconstruct_warm_component_block(
            self.warm_key_pq_codes,
            self.warm_key_rvq_codes,
            self.warm_key_rvq_mask,
            self.warm_key_rvq_offsets,
            quantizer,
            start_token,
            end_token,
        )

    def reconstruct_warm_value_block(self, quantizer: HybridQuantizerMLX, start_token: int, end_token: int):
        return self._reconstruct_warm_component_block(
            self.warm_value_pq_codes,
            self.warm_value_rvq_codes,
            self.warm_value_rvq_mask,
            self.warm_value_rvq_offsets,
            quantizer,
            start_token,
            end_token,
        )

    def _attention_forward_impl(
        self,
        q,
        quantizer: HybridQuantizerMLX,
        warm_read_mode: str = "full",
        warm_block_size_tokens: Optional[int] = None,
        warm_block_token_ranges: Optional[List[Tuple[int, int]]] = None,
        collect_metrics: bool = False,
    ) -> Tuple[Any, Dict[str, float | int | str]]:
        if warm_read_mode not in {"full", "blockwise"}:
            raise ValueError(f"Unsupported warm read mode: {warm_read_mode}")

        block_size_tokens = self.config.block_size_seq if warm_block_size_tokens is None else warm_block_size_tokens
        if block_size_tokens <= 0:
            raise ValueError("warm_block_size_tokens must be positive")
        self._validate_warm_block_token_ranges(warm_block_token_ranges)

        key_parts: List[np.ndarray] = []
        value_parts: List[np.ndarray] = []
        q_np = _mx_to_np(q, np.float32)
        metrics: Dict[str, float | int | str] = {}

        if collect_metrics:
            stored = self.memory_usage_bytes()
            metrics = {
                "warm_read_mode": warm_read_mode,
                "warm_block_size_tokens": int(block_size_tokens),
                "query_batch": int(q_np.shape[0]),
                "hot_tokens": int(self.num_hot),
                "warm_tokens": int(self.num_warm),
                "cold_tokens": int(self.num_cold),
                "warm_active": int(self.num_warm > 0),
                "warm_key_rvq_rows": int(self.warm_key_rvq_codes.shape[0]),
                "warm_value_rvq_rows": int(self.warm_value_rvq_codes.shape[0]),
                "warm_reconstruct_ms": 0.0,
                "concat_ms": 0.0,
                "attention_ms": 0.0,
                "total_ms": 0.0,
                "warm_blocks": 0,
                "warm_decode_tokens": 0,
                "warm_reconstruction_fp16_bytes": 0,
                "warm_reconstruction_fp32_bytes": 0,
                "dense_kv_fp32_bytes": 0,
                "query_fp32_bytes": int(q_np.nbytes),
                "stored_bytes": int(
                    stored["hot_bytes"]
                    + stored["warm_key_pq_bytes"]
                    + stored["warm_value_pq_bytes"]
                    + stored["warm_key_rvq_bytes"]
                    + stored["warm_value_rvq_bytes"]
                ),
            }
            total_start = time.perf_counter()

        hot_keys_np: Optional[np.ndarray] = None
        hot_values_np: Optional[np.ndarray] = None

        if self.num_hot > 0:
            hot_keys_np = _mx_to_np(self.hot_keys, np.float32)
            hot_values_np = _mx_to_np(self.hot_values, np.float32)

            if warm_read_mode == "full":
                key_parts.append(hot_keys_np)
                value_parts.append(hot_values_np)

        if warm_read_mode == "full":
            if self.num_warm > 0:
                if collect_metrics:
                    reconstruct_start = time.perf_counter()

                warm_keys = self.reconstruct_warm_keys(quantizer)
                warm_values = self.reconstruct_warm_values(quantizer)
                warm_keys_np = _mx_to_np(warm_keys, np.float32)
                warm_values_np = _mx_to_np(warm_values, np.float32)

                key_parts.append(warm_keys_np)
                value_parts.append(warm_values_np)

                if collect_metrics:
                    metrics["warm_reconstruct_ms"] = (time.perf_counter() - reconstruct_start) * 1000.0
                    metrics["warm_blocks"] = int(self.num_warm > 0)
                    metrics["warm_decode_tokens"] = int(self.num_warm)
                    metrics["warm_reconstruction_fp16_bytes"] = int(self.num_warm * self.config.num_heads * self.config.head_dim * 2 * 2)
                    metrics["warm_reconstruction_fp32_bytes"] = int(warm_keys_np.nbytes + warm_values_np.nbytes)

            if not key_parts:
                out = mx.zeros(q.shape, dtype=_mx_dtype("float16"))
                if collect_metrics:
                    metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
                return out, metrics

            if collect_metrics:
                concat_start = time.perf_counter()

            all_keys = np.concatenate(key_parts, axis=0)
            all_values = np.concatenate(value_parts, axis=0)
            if collect_metrics:
                metrics["concat_ms"] = (time.perf_counter() - concat_start) * 1000.0
                metrics["dense_kv_fp32_bytes"] = int(all_keys.nbytes + all_values.nbytes)
                attention_start = time.perf_counter()

            out = dense_attention_reference_np(q_np, all_keys, all_values)

            if collect_metrics:
                metrics["attention_ms"] = (time.perf_counter() - attention_start) * 1000.0
                metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0

            return _np_to_mx(out, dtype=_mx_dtype("float16")), metrics

        if self.num_hot == 0 and self.num_warm == 0:
            out = mx.zeros(q.shape, dtype=_mx_dtype("float16"))
            if collect_metrics:
                metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return out, metrics

        running_max = np.full((q_np.shape[0], q_np.shape[1]), -np.inf, dtype=np.float32)
        running_sum = np.zeros((q_np.shape[0], q_np.shape[1]), dtype=np.float32)
        running_out = np.zeros((q_np.shape[0], q_np.shape[1], q_np.shape[2]), dtype=np.float32)

        hot_fp32_bytes = 0
        if hot_keys_np is not None and hot_values_np is not None:
            hot_fp32_bytes = int(hot_keys_np.nbytes + hot_values_np.nbytes)
            if collect_metrics:
                metrics["dense_kv_fp32_bytes"] = hot_fp32_bytes
                attention_start = time.perf_counter()
            running_max, running_sum, running_out = _streaming_attention_update_np(
                q_np,
                hot_keys_np,
                hot_values_np,
                running_max,
                running_sum,
                running_out,
            )
            if collect_metrics:
                metrics["attention_ms"] = float(metrics["attention_ms"]) + (time.perf_counter() - attention_start) * 1000.0

        warm_ranges = (
            self._default_warm_block_token_ranges(block_size_tokens)
            if warm_block_token_ranges is None
            else list(warm_block_token_ranges)
        )
        numpy_decode_state = self._build_blockwise_numpy_decode_state(quantizer) if warm_ranges else None

        for start_token, end_token in warm_ranges:
            token_count = end_token - start_token

            if collect_metrics:
                reconstruct_start = time.perf_counter()

            warm_keys_np = self._reconstruct_warm_component_block_numpy(
                numpy_decode_state["key"],
                numpy_decode_state,
                start_token,
                end_token,
            )
            warm_values_np = self._reconstruct_warm_component_block_numpy(
                numpy_decode_state["value"],
                numpy_decode_state,
                start_token,
                end_token,
            )

            if collect_metrics:
                metrics["warm_reconstruct_ms"] = float(metrics["warm_reconstruct_ms"]) + (time.perf_counter() - reconstruct_start) * 1000.0
                metrics["warm_blocks"] = int(metrics["warm_blocks"]) + 1
                metrics["warm_decode_tokens"] = int(metrics["warm_decode_tokens"]) + token_count
                metrics["warm_reconstruction_fp16_bytes"] = max(
                    int(metrics["warm_reconstruction_fp16_bytes"]),
                    token_count * self.config.num_heads * self.config.head_dim * 2 * 2,
                )
                block_fp32_bytes = int(warm_keys_np.nbytes + warm_values_np.nbytes)
                metrics["warm_reconstruction_fp32_bytes"] = max(
                    int(metrics["warm_reconstruction_fp32_bytes"]),
                    block_fp32_bytes,
                )
                metrics["dense_kv_fp32_bytes"] = max(
                    int(metrics["dense_kv_fp32_bytes"]),
                    hot_fp32_bytes + block_fp32_bytes,
                )
                attention_start = time.perf_counter()

            running_max, running_sum, running_out = _streaming_attention_update_np(
                q_np,
                warm_keys_np,
                warm_values_np,
                running_max,
                running_sum,
                running_out,
            )

            if collect_metrics:
                metrics["attention_ms"] = float(metrics["attention_ms"]) + (time.perf_counter() - attention_start) * 1000.0

        denom = np.where(running_sum == 0.0, 1.0, running_sum)
        out = (running_out / denom[:, :, None]).astype(np.float32)

        if collect_metrics:
            metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0

        return _np_to_mx(out, dtype=_mx_dtype("float16")), metrics

    def attention_forward(
        self,
        q,
        quantizer: HybridQuantizerMLX,
        warm_read_mode: str = "full",
        warm_block_size_tokens: Optional[int] = None,
        warm_block_token_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        out, _ = self._attention_forward_impl(
            q,
            quantizer,
            warm_read_mode=warm_read_mode,
            warm_block_size_tokens=warm_block_size_tokens,
            warm_block_token_ranges=warm_block_token_ranges,
            collect_metrics=False,
        )
        return out

    def attention_forward_profile(
        self,
        q,
        quantizer: HybridQuantizerMLX,
        warm_read_mode: str = "full",
        warm_block_size_tokens: Optional[int] = None,
        warm_block_token_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Any, Dict[str, float | int | str]]:
        return self._attention_forward_impl(
            q,
            quantizer,
            warm_read_mode=warm_read_mode,
            warm_block_size_tokens=warm_block_size_tokens,
            warm_block_token_ranges=warm_block_token_ranges,
            collect_metrics=True,
        )

    def memory_usage_bytes(self) -> dict:
        warm_key_rvq_rows = int(self.warm_key_rvq_codes.shape[0])
        warm_value_rvq_rows = int(self.warm_value_rvq_codes.shape[0])
        return {
            "hot_bytes": self.num_hot * self.config.num_heads * self.config.head_dim * 2 * 2,
            "warm_key_pq_bytes": self.num_warm * self.config.num_heads * self.config.num_subspaces,
            "warm_value_pq_bytes": self.num_warm * self.config.num_heads * self.config.num_subspaces,
            "warm_key_rvq_bytes": warm_key_rvq_rows * self.config.num_rvq_layers * 2,
            "warm_value_rvq_bytes": warm_value_rvq_rows * self.config.num_rvq_layers * 2,
            "cold_chunks": len(self.cold_chunk_paths),
            "cold_tokens": self.num_cold,
        }


class AsyncHierarchicalRouterMLX:
    """
    Simple chunk prefetch helper.

    This class does window-based chunk selection and async loading into an in-memory
    cache. It does not claim predictive quality beyond that heuristic.
    """

    def __init__(self, config: RFSNConfig, disk_dir: Optional[Path] = None):
        self.config = config
        self.disk_dir = Path(config.disk_cache_dir) if disk_dir is None else Path(disk_dir)
        self.throttle = config.prefetch_throttle_s
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._max_cache_size = 16
        self._pending_prefetch: set[int] = set()

    async def predict_and_prefetch(self, current_position: int, context_window: int, top_k: int = 2) -> List[int]:
        chunk_ids = self._candidate_chunk_ids(current_position, context_window)[:top_k]
        loaded: List[int] = []
        for cid in chunk_ids:
            if cid in self._cache:
                loaded.append(cid)
                continue
            if cid in self._pending_prefetch:
                continue
            self._pending_prefetch.add(cid)
            try:
                await self._load_chunk(cid)
                loaded.append(cid)
            finally:
                self._pending_prefetch.discard(cid)
            await asyncio.sleep(self.throttle)

        while len(self._cache) > self._max_cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return loaded

    def _candidate_chunk_ids(self, current_position: int, context_window: int) -> List[int]:
        chunk_size = 4096
        start_chunk = max(0, (current_position - context_window) // chunk_size - 1)
        end_chunk = max(start_chunk, (current_position + context_window) // chunk_size + 1)
        ids: List[int] = []
        for cid in range(start_chunk, end_chunk + 1):
            path = self.disk_dir / f"layer0_chunk{cid}.npz"
            if path.exists():
                ids.append(cid)
        ids.sort(key=lambda cid: abs(cid * chunk_size - current_position))
        return ids

    async def _load_chunk(self, chunk_id: int) -> None:
        loop = asyncio.get_event_loop()
        self._cache[chunk_id] = await loop.run_in_executor(None, self._load_chunk_sync, chunk_id)

    def _load_chunk_sync(self, chunk_id: int) -> Dict[str, Any]:
        path = self.disk_dir / f"layer0_chunk{chunk_id}.npz"
        if not path.exists():
            return {}
        data = np.load(path)
        return {k: _np_to_mx(np.array(v)) for k, v in data.items()}


def calibrate_quantizer(quantizer: HybridQuantizerMLX, calibration_vectors, num_iterations: int = 10) -> dict:
    logger.info("Calibrating PQ codebooks with %d vectors for %d iterations", calibration_vectors.shape[0], num_iterations)
    vectors = _mx_to_np(calibration_vectors, np.float32)
    n, _ = vectors.shape
    codebook_size = 1 << quantizer.config.pq_bits
    sub_dim = quantizer.config.subspace_dim

    metrics = {"avg_distortion": []}
    codebooks = _mx_to_np(quantizer.pq.codebooks, np.float32)

    for sub in range(quantizer.config.num_subspaces):
        s0 = sub * sub_dim
        s1 = s0 + sub_dim
        idx = np.random.choice(n, size=codebook_size, replace=n < codebook_size)
        codebooks[sub] = vectors[idx, s0:s1]

    quantizer.pq.codebooks = _np_to_mx(codebooks, dtype=_mx_dtype("float16"))

    for _ in range(num_iterations):
        codes, _ = quantizer.pq.quantize(_np_to_mx(vectors, dtype=_mx_dtype("float16")))
        codes_np = _mx_to_np(codes, np.int32)
        recon = np.zeros_like(vectors, dtype=np.float32)
        codebooks = _mx_to_np(quantizer.pq.codebooks, np.float32)

        for sub in range(quantizer.config.num_subspaces):
            s0 = sub * sub_dim
            s1 = s0 + sub_dim
            sub_vectors = vectors[:, s0:s1]
            for c in range(codebook_size):
                mask = codes_np[:, sub] == c
                if np.any(mask):
                    codebooks[sub, c] = sub_vectors[mask].mean(axis=0)
            recon[:, s0:s1] = codebooks[sub][codes_np[:, sub]]

        quantizer.pq.codebooks = _np_to_mx(codebooks, dtype=_mx_dtype("float16"))
        mse = float(np.mean((vectors - recon) ** 2))
        metrics["avg_distortion"].append(mse)

    return metrics


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name} mismatch; max_abs={max_abs}")


def run_tests() -> bool:
    _require_mlx()
    np.random.seed(0)
    config = RFSNConfig(
        hidden_dim=512,
        num_heads=4,
        head_dim=128,
        num_layers=2,
        hot_capacity=6,
        warm_capacity=8,
        cold_capacity=64,
        rvq_codebook_size=128,
    )

    logger.info("=" * 68)
    logger.info("RFSN v10.2 MLX experimental prototype tests")
    logger.info("=" * 68)

    quantizer = HybridQuantizerMLX(config)
    vectors = _np_to_mx(np.random.randn(40, config.head_dim).astype(np.float32), dtype=_mx_dtype("float16"))

    logger.info("[1] PQ encode/decode")
    pq_codes, residuals = quantizer.pq.quantize(vectors)
    pq_recon = quantizer.pq.decode(pq_codes)
    pq_mse = float(np.mean((_mx_to_np(vectors, np.float32) - _mx_to_np(pq_recon, np.float32)) ** 2))
    assert pq_codes.shape == (40, config.num_subspaces)
    assert residuals.shape == vectors.shape
    assert np.isfinite(pq_mse)

    logger.info("[2] RVQ offsets are valid row indices")
    rvq_codes, rvq_mask, rvq_offsets = quantizer.rvq.encode(residuals)
    offsets_np = _mx_to_np(rvq_offsets, np.int32)
    assert offsets_np.ndim == 1
    assert np.all(offsets_np >= 0)
    assert np.all(offsets_np < vectors.shape[0])
    assert rvq_codes.shape[1] == config.num_rvq_layers if rvq_codes.shape[0] > 0 else True

    logger.info("[3] Hybrid reconstruction improves or matches PQ-only")
    hybrid_recon = quantizer.decode(pq_codes, rvq_codes, rvq_mask, rvq_offsets)
    hybrid_mse = float(np.mean((_mx_to_np(vectors, np.float32) - _mx_to_np(hybrid_recon, np.float32)) ** 2))
    assert hybrid_mse <= pq_mse + 1e-6

    logger.info("[4] Calibration lowers or preserves PQ distortion")
    fresh = HybridQuantizerMLX(config)
    before_codes, _ = fresh.pq.quantize(vectors)
    before_recon = fresh.pq.decode(before_codes)
    before_mse = float(np.mean((_mx_to_np(vectors, np.float32) - _mx_to_np(before_recon, np.float32)) ** 2))
    metrics = calibrate_quantizer(fresh, vectors, num_iterations=4)
    after_codes, _ = fresh.pq.quantize(vectors)
    after_recon = fresh.pq.decode(after_codes)
    after_mse = float(np.mean((_mx_to_np(vectors, np.float32) - _mx_to_np(after_recon, np.float32)) ** 2))
    assert metrics["avg_distortion"]
    assert after_mse <= before_mse + 1e-6

    logger.info("[5] Warm tier stores keys and values separately")
    cache = RFSNv10KVCacheMLX(config, layer_idx=0)

    hot_keys = np.random.randn(4, config.num_heads, config.head_dim).astype(np.float32)
    hot_values = np.random.randn(4, config.num_heads, config.head_dim).astype(np.float32)
    cache.update(_np_to_mx(hot_keys, dtype=_mx_dtype("float16")), _np_to_mx(hot_values, dtype=_mx_dtype("float16")), quantizer)

    warm_keys = np.random.randn(5, config.num_heads, config.head_dim).astype(np.float32)
    warm_values = np.random.randn(5, config.num_heads, config.head_dim).astype(np.float32)
    cache.update(_np_to_mx(warm_keys, dtype=_mx_dtype("float16")), _np_to_mx(warm_values, dtype=_mx_dtype("float16")), quantizer)

    assert cache.warm_key_pq_codes.shape[0] == cache.num_warm
    assert cache.warm_value_pq_codes.shape[0] == cache.num_warm
    assert cache.warm_key_rvq_mask.shape == (cache.num_warm, config.num_heads)
    assert cache.warm_value_rvq_mask.shape == (cache.num_warm, config.num_heads)

    logger.info("[6] Warm reconstruction returns coherent K/V tensors")
    warm_k_recon = _mx_to_np(cache.reconstruct_warm_keys(quantizer), np.float32)
    warm_v_recon = _mx_to_np(cache.reconstruct_warm_values(quantizer), np.float32)
    assert warm_k_recon.shape == (cache.num_warm, config.num_heads, config.head_dim)
    assert warm_v_recon.shape == (cache.num_warm, config.num_heads, config.head_dim)

    logger.info("[7] Hot-only attention matches dense reference")
    hot_only_cache = RFSNv10KVCacheMLX(config, layer_idx=0)
    hot_only_cache.update(_np_to_mx(hot_keys, dtype=_mx_dtype("float16")), _np_to_mx(hot_values, dtype=_mx_dtype("float16")), quantizer)
    q = np.random.randn(2, config.num_heads, config.head_dim).astype(np.float32)
    out_hot = _mx_to_np(hot_only_cache.attention_forward(_np_to_mx(q, dtype=_mx_dtype("float16")), quantizer), np.float32)
    ref_hot = dense_attention_reference_np(q, hot_keys, hot_values)
    _assert_close("hot_attention", out_hot, ref_hot, atol=2e-3, rtol=2e-3)

    logger.info("[8] Warm reconstructed attention matches dense reference built from reconstructed K/V")
    warm_config = RFSNConfig(
        hidden_dim=512,
        num_heads=4,
        head_dim=128,
        num_layers=2,
        hot_capacity=0,
        warm_capacity=8,
        cold_capacity=64,
        rvq_codebook_size=128,
    )
    warm_quantizer = HybridQuantizerMLX(warm_config)
    warm_only_cache = RFSNv10KVCacheMLX(warm_config, layer_idx=0)
    warm_keys_src = np.random.randn(7, warm_config.num_heads, warm_config.head_dim).astype(np.float32)
    warm_values_src = np.random.randn(7, warm_config.num_heads, warm_config.head_dim).astype(np.float32)
    warm_only_cache.update(
        _np_to_mx(warm_keys_src, dtype=_mx_dtype("float16")),
        _np_to_mx(warm_values_src, dtype=_mx_dtype("float16")),
        warm_quantizer,
    )
    warm_q = np.random.randn(2, warm_config.num_heads, warm_config.head_dim).astype(np.float32)
    warm_out = _mx_to_np(warm_only_cache.attention_forward(_np_to_mx(warm_q, dtype=_mx_dtype("float16")), warm_quantizer), np.float32)
    warm_ref_k = _mx_to_np(warm_only_cache.reconstruct_warm_keys(warm_quantizer), np.float32)
    warm_ref_v = _mx_to_np(warm_only_cache.reconstruct_warm_values(warm_quantizer), np.float32)
    warm_ref = dense_attention_reference_np(warm_q, warm_ref_k, warm_ref_v)
    _assert_close("warm_attention", warm_out, warm_ref, atol=2e-3, rtol=2e-3)

    logger.info("[9] Hot+warm combined attention matches dense reference over hot + reconstructed warm K/V")
    combo_q = np.random.randn(2, config.num_heads, config.head_dim).astype(np.float32)
    combo_out = _mx_to_np(cache.attention_forward(_np_to_mx(combo_q, dtype=_mx_dtype("float16")), quantizer), np.float32)
    combo_keys = np.concatenate([
        _mx_to_np(cache.hot_keys, np.float32),
        _mx_to_np(cache.reconstruct_warm_keys(quantizer), np.float32),
    ], axis=0)
    combo_values = np.concatenate([
        _mx_to_np(cache.hot_values, np.float32),
        _mx_to_np(cache.reconstruct_warm_values(quantizer), np.float32),
    ], axis=0)
    ref_combo = dense_attention_reference_np(combo_q, combo_keys, combo_values)
    _assert_close("hot_warm_attention", combo_out, ref_combo, atol=2e-3, rtol=2e-3)

    logger.info("[10] Profiled attention matches cache output and reports finite metrics")
    profiled_out_mx, profile_metrics = cache.attention_forward_profile(
        _np_to_mx(combo_q, dtype=_mx_dtype("float16")),
        quantizer,
    )
    profiled_out = _mx_to_np(profiled_out_mx, np.float32)
    _assert_close("profiled_attention", profiled_out, combo_out, atol=2e-3, rtol=2e-3)
    expected_metric_keys = {
        "warm_read_mode",
        "warm_block_size_tokens",
        "query_batch",
        "hot_tokens",
        "warm_tokens",
        "cold_tokens",
        "warm_active",
        "warm_blocks",
        "warm_decode_tokens",
        "warm_reconstruct_ms",
        "concat_ms",
        "attention_ms",
        "total_ms",
        "warm_reconstruction_fp16_bytes",
        "warm_reconstruction_fp32_bytes",
        "dense_kv_fp32_bytes",
        "stored_bytes",
    }
    assert expected_metric_keys.issubset(profile_metrics.keys())
    assert int(profile_metrics["warm_tokens"]) == cache.num_warm
    for key in ("warm_reconstruct_ms", "concat_ms", "attention_ms", "total_ms"):
        assert float(profile_metrics[key]) >= 0.0

    logger.info("[11] Blockwise warm attention matches full warm attention and lowers transient warm bytes")
    blockwise_out_mx, blockwise_metrics = cache.attention_forward_profile(
        _np_to_mx(combo_q, dtype=_mx_dtype("float16")),
        quantizer,
        warm_read_mode="blockwise",
        warm_block_size_tokens=2,
    )
    blockwise_out = _mx_to_np(blockwise_out_mx, np.float32)
    _assert_close("blockwise_attention", blockwise_out, combo_out, atol=2e-3, rtol=2e-3)
    assert blockwise_metrics["warm_read_mode"] == "blockwise"
    assert int(blockwise_metrics["warm_blocks"]) == 2
    assert int(blockwise_metrics["warm_decode_tokens"]) == cache.num_warm
    assert int(blockwise_metrics["warm_reconstruction_fp16_bytes"]) < int(profile_metrics["warm_reconstruction_fp16_bytes"])
    assert int(blockwise_metrics["warm_reconstruction_fp32_bytes"]) < int(profile_metrics["warm_reconstruction_fp32_bytes"])

    logger.info("[12] Partial warm block selection matches dense reference over hot plus selected warm ranges")
    selected_ranges = [(0, 1), (cache.num_warm - 1, cache.num_warm)]
    partial_out_mx, partial_metrics = cache.attention_forward_profile(
        _np_to_mx(combo_q, dtype=_mx_dtype("float16")),
        quantizer,
        warm_read_mode="blockwise",
        warm_block_size_tokens=2,
        warm_block_token_ranges=selected_ranges,
    )
    partial_out = _mx_to_np(partial_out_mx, np.float32)
    selected_warm_keys = np.concatenate([
        _mx_to_np(cache.reconstruct_warm_key_block(quantizer, start_token, end_token), np.float32)
        for start_token, end_token in selected_ranges
    ], axis=0)
    selected_warm_values = np.concatenate([
        _mx_to_np(cache.reconstruct_warm_value_block(quantizer, start_token, end_token), np.float32)
        for start_token, end_token in selected_ranges
    ], axis=0)
    partial_ref = dense_attention_reference_np(
        combo_q,
        np.concatenate([_mx_to_np(cache.hot_keys, np.float32), selected_warm_keys], axis=0),
        np.concatenate([_mx_to_np(cache.hot_values, np.float32), selected_warm_values], axis=0),
    )
    _assert_close("partial_blockwise_attention", partial_out, partial_ref, atol=2e-3, rtol=2e-3)
    assert int(partial_metrics["warm_blocks"]) == len(selected_ranges)
    assert int(partial_metrics["warm_decode_tokens"]) == sum(end - start for start, end in selected_ranges)

    try:
        cache.attention_forward_profile(
            _np_to_mx(combo_q, dtype=_mx_dtype("float16")),
            quantizer,
            warm_read_mode="blockwise",
            warm_block_size_tokens=2,
            warm_block_token_ranges=[(0, 2), (1, 3)],
        )
        raise AssertionError("overlapping warm block token ranges should raise ValueError")
    except ValueError:
        pass

    logger.info("[13] Cold spill writes actual .npz files with expected fields")
    with tempfile.TemporaryDirectory() as tmpdir:
        cold_cache = RFSNv10KVCacheMLX(config, layer_idx=0)
        large_keys = np.random.randn(20, config.num_heads, config.head_dim).astype(np.float32)
        large_values = np.random.randn(20, config.num_heads, config.head_dim).astype(np.float32)
        cold_cache.update(
            _np_to_mx(large_keys, dtype=_mx_dtype("float16")),
            _np_to_mx(large_values, dtype=_mx_dtype("float16")),
            quantizer,
            disk_dir=Path(tmpdir),
        )
        assert cold_cache.num_hot == config.hot_capacity
        assert cold_cache.num_warm == config.warm_capacity
        assert cold_cache.num_cold == 20 - config.hot_capacity - config.warm_capacity
        assert len(cold_cache.cold_chunk_paths) == 1
        assert cold_cache.cold_chunk_paths[0].suffix == ".npz"
        assert cold_cache.cold_chunk_paths[0].exists()

        chunk = cold_cache.load_cold_chunk(0)
        expected = {
            "seq_tokens",
            "num_heads",
            "head_dim",
            "key_pq_codes",
            "key_rvq_codes",
            "key_rvq_mask",
            "key_rvq_offsets",
            "value_pq_codes",
            "value_rvq_codes",
            "value_rvq_mask",
            "value_rvq_offsets",
        }
        assert expected.issubset(set(chunk.keys()))

        logger.info("[14] Router prefetch loads chunk files without crashing")
        router = AsyncHierarchicalRouterMLX(config, disk_dir=Path(tmpdir))
        loaded = asyncio.run(router.predict_and_prefetch(current_position=0, context_window=8192, top_k=2))
        assert isinstance(loaded, list)
        assert 0 in loaded or loaded == []

    logger.info("[15] Memory usage accounting is finite")
    usage = cache.memory_usage_bytes()
    for value in usage.values():
        assert isinstance(value, int)
        assert value >= 0

    logger.info("All tests passed")
    logger.info("Remaining limitations: cold read-path attention is not implemented; quantized codebook fast paths are absent.")
    return True


if __name__ == "__main__":
    run_tests()
