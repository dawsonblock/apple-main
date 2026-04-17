"""
RFSN v10.2 — MLX experimental prototype compatibility facade
=============================================================

This module preserves the original public surface while the implementation is
split across dedicated common, attention, codec, cache, and router modules.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

import numpy as np

import rfsn_v10_common as common
from rfsn_v10_attention import (
    RFSNHybridAttentionMLX,
    dense_attention_reference_mx,
    dense_attention_reference_np,
)
from rfsn_v10_cache import RFSNv10KVCacheMLX
from rfsn_v10_codec import (
    HybridQuantizerMLX,
    ProductQuantizerMLX,
    ResidualVQMLX,
    calibrate_quantizer,
)
from rfsn_v10_router import AsyncHierarchicalRouterMLX

HAS_MLX = common.HAS_MLX
mx = common.mx
logger = common.logger

RFSNConfig = common.RFSNConfig
_mx_dtype = common._mx_dtype
_mx_to_np = common._mx_to_np
_np_to_mx = common._np_to_mx
_force_eval = common._force_eval

__all__ = [
    "HAS_MLX",
    "mx",
    "RFSNConfig",
    "ProductQuantizerMLX",
    "ResidualVQMLX",
    "HybridQuantizerMLX",
    "RFSNHybridAttentionMLX",
    "RFSNv10KVCacheMLX",
    "AsyncHierarchicalRouterMLX",
    "dense_attention_reference_mx",
    "dense_attention_reference_np",
    "calibrate_quantizer",
    "_mx_dtype",
    "_mx_to_np",
    "_np_to_mx",
    "_force_eval",
    "run_tests",
]


def _assert_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name} mismatch; max_abs={max_abs}")


def run_tests() -> bool:
    common._require_mlx()
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
    rvq_codes, rvq_mask, rvq_offsets, rvq_entry_mask = quantizer.rvq.encode(residuals)
    offsets_np = _mx_to_np(rvq_offsets, np.int32)
    assert offsets_np.ndim == 1
    assert np.all(offsets_np >= 0)
    assert np.all(offsets_np < vectors.shape[0])
    assert rvq_codes.shape[1] == config.num_rvq_layers if rvq_codes.shape[0] > 0 else True
    assert rvq_entry_mask.shape[0] == rvq_codes.shape[0]

    logger.info("[3] Hybrid reconstruction improves or matches PQ-only")
    hybrid_recon = quantizer.decode(pq_codes, rvq_codes, rvq_mask, rvq_offsets, rvq_entry_mask)
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
    assert cache.hot_keys.shape == (config.hot_capacity, config.num_heads, config.head_dim)
    assert cache.hot_values.shape == (config.hot_capacity, config.num_heads, config.head_dim)
    _assert_close("preallocated_hot_keys", _mx_to_np(cache.hot_keys[:cache.num_hot], np.float32), hot_keys, atol=2e-3, rtol=2e-3)
    _assert_close("preallocated_hot_values", _mx_to_np(cache.hot_values[:cache.num_hot], np.float32), hot_values, atol=2e-3, rtol=2e-3)

    warm_keys = np.random.randn(5, config.num_heads, config.head_dim).astype(np.float32)
    warm_values = np.random.randn(5, config.num_heads, config.head_dim).astype(np.float32)
    cache.update(_np_to_mx(warm_keys, dtype=_mx_dtype("float16")), _np_to_mx(warm_values, dtype=_mx_dtype("float16")), quantizer)

    assert cache.warm_key_pq_codes.shape[0] == cache.num_warm
    assert cache.warm_value_pq_codes.shape[0] == cache.num_warm
    assert cache.warm_key_rvq_mask.shape == (cache.num_warm, config.num_heads)
    assert cache.warm_value_rvq_mask.shape == (cache.num_warm, config.num_heads)
    assert cache.warm_key_rvq_entry_mask.shape[0] == cache.warm_key_rvq_codes.shape[0]
    assert cache.warm_value_rvq_entry_mask.shape[0] == cache.warm_value_rvq_codes.shape[0]

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
        _mx_to_np(cache.hot_keys[:cache.num_hot], np.float32),
        _mx_to_np(cache.reconstruct_warm_keys(quantizer), np.float32),
    ], axis=0)
    combo_values = np.concatenate([
        _mx_to_np(cache.hot_values[:cache.num_hot], np.float32),
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
        np.concatenate([_mx_to_np(cache.hot_keys[:cache.num_hot], np.float32), selected_warm_keys], axis=0),
        np.concatenate([_mx_to_np(cache.hot_values[:cache.num_hot], np.float32), selected_warm_values], axis=0),
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
            "key_rvq_entry_mask",
            "value_pq_codes",
            "value_rvq_codes",
            "value_rvq_mask",
            "value_rvq_offsets",
            "value_rvq_entry_mask",
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