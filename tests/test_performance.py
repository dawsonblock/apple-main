from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rfsn_v10_mlx_ane_complete as rfsn


def _make_config() -> rfsn.RFSNConfig:
    return rfsn.RFSNConfig(
        hidden_dim=512,
        num_heads=4,
        head_dim=128,
        num_layers=2,
        hot_capacity=8,
        warm_capacity=8,
        cold_capacity=64,
        block_size_seq=2,
        rvq_codebook_size=128,
        rvq_max_active=16,
    )


def _fill_hot_only(cache: rfsn.RFSNv10KVCacheMLX, quantizer: rfsn.HybridQuantizerMLX, config: rfsn.RFSNConfig) -> None:
    token_keys = np.random.randn(1, config.num_heads, config.head_dim).astype(np.float32)
    token_values = np.random.randn(1, config.num_heads, config.head_dim).astype(np.float32)
    for _ in range(config.hot_capacity):
        cache.update(
            rfsn._np_to_mx(token_keys, dtype=rfsn._mx_dtype("float16")),
            rfsn._np_to_mx(token_values, dtype=rfsn._mx_dtype("float16")),
            quantizer,
        )


def _fill_hot_and_warm(cache: rfsn.RFSNv10KVCacheMLX, quantizer: rfsn.HybridQuantizerMLX, config: rfsn.RFSNConfig) -> None:
    total_tokens = config.hot_capacity + 4
    keys = np.random.randn(total_tokens, config.num_heads, config.head_dim).astype(np.float32)
    values = np.random.randn(total_tokens, config.num_heads, config.head_dim).astype(np.float32)
    cache.update(
        rfsn._np_to_mx(keys, dtype=rfsn._mx_dtype("float16")),
        rfsn._np_to_mx(values, dtype=rfsn._mx_dtype("float16")),
        quantizer,
    )


def verify_no_hot_concatenate() -> None:
    config = _make_config()
    quantizer = rfsn.HybridQuantizerMLX(config)
    cache = rfsn.RFSNv10KVCacheMLX(config, layer_idx=0)

    original_concatenate = rfsn.mx.concatenate
    concatenate_calls = 0

    def tracked_concatenate(*args, **kwargs):
        nonlocal concatenate_calls
        concatenate_calls += 1
        return original_concatenate(*args, **kwargs)

    rfsn.mx.concatenate = tracked_concatenate
    try:
        _fill_hot_only(cache, quantizer, config)
    finally:
        rfsn.mx.concatenate = original_concatenate

    assert concatenate_calls == 0, f"expected zero hot-path concatenations, saw {concatenate_calls}"
    assert cache.num_hot == config.hot_capacity
    assert cache.hot_keys.shape == (config.hot_capacity, config.num_heads, config.head_dim)


def verify_no_host_copies_in_attention() -> None:
    config = _make_config()
    quantizer = rfsn.HybridQuantizerMLX(config)
    cache = rfsn.RFSNv10KVCacheMLX(config, layer_idx=0)
    _fill_hot_and_warm(cache, quantizer, config)

    query = np.random.randn(2, config.num_heads, config.head_dim).astype(np.float32)
    query_mx = rfsn._np_to_mx(query, dtype=rfsn._mx_dtype("float16"))

    original_mx_to_np = rfsn._mx_to_np
    original_np_to_mx = rfsn._np_to_mx
    mx_to_np_calls = 0
    np_to_mx_calls = 0

    def tracked_mx_to_np(*args, **kwargs):
        nonlocal mx_to_np_calls
        mx_to_np_calls += 1
        return original_mx_to_np(*args, **kwargs)

    def tracked_np_to_mx(*args, **kwargs):
        nonlocal np_to_mx_calls
        np_to_mx_calls += 1
        return original_np_to_mx(*args, **kwargs)

    rfsn._mx_to_np = tracked_mx_to_np
    rfsn._np_to_mx = tracked_np_to_mx
    try:
        output = cache.attention_forward(
            query_mx,
            quantizer,
            warm_read_mode="blockwise",
            warm_block_size_tokens=2,
        )
        rfsn._force_eval(output)
    finally:
        rfsn._mx_to_np = original_mx_to_np
        rfsn._np_to_mx = original_np_to_mx

    assert mx_to_np_calls == 0, f"attention_forward touched host via _mx_to_np {mx_to_np_calls} time(s)"
    assert np_to_mx_calls == 0, f"attention_forward rebuilt device tensors via _np_to_mx {np_to_mx_calls} time(s)"


def maybe_run_profiler() -> None:
    profiler = getattr(rfsn.mx, "profiler", None)
    if profiler is None:
        print("mx.profiler unavailable in this MLX runtime; using structural checks only")
        return

    start = getattr(profiler, "start", None)
    stop = getattr(profiler, "stop", None)
    if not callable(start) or not callable(stop):
        print("mx.profiler present but does not expose start/stop; using structural checks only")
        return

    config = _make_config()
    quantizer = rfsn.HybridQuantizerMLX(config)
    cache = rfsn.RFSNv10KVCacheMLX(config, layer_idx=0)
    _fill_hot_and_warm(cache, quantizer, config)
    query = rfsn._np_to_mx(np.random.randn(1, config.num_heads, config.head_dim).astype(np.float32), dtype=rfsn._mx_dtype("float16"))

    start()
    output = cache.attention_forward(query, quantizer, warm_read_mode="blockwise", warm_block_size_tokens=2)
    rfsn._force_eval(output)
    stop()
    print("mx.profiler captured a decode step")


def main() -> None:
    if not rfsn.HAS_MLX:
        raise SystemExit("MLX is required to run tests/test_performance.py")

    np.random.seed(0)
    verify_no_hot_concatenate()
    verify_no_host_copies_in_attention()
    maybe_run_profiler()
    print("performance structural checks passed")


if __name__ == "__main__":
    main()