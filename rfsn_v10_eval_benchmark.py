"""
Sequence-level benchmark for dense attention vs. the current hot+warm cache path.

This harness keeps the prototype honest: it measures the cache path exactly as it
exists today and can compare full warm-tier reconstruction against exact blockwise
warm reads under the same workload.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Sequence

import numpy as np

import rfsn_v10_mlx_ane_complete as rfsn


CSV_FIELDS = [
    "mode",
    "warm_read_mode",
    "warm_block_size_tokens",
    "warm_selection_policy",
    "warm_selection_blocks",
    "trial_index",
    "seed",
    "seq_len",
    "step_tokens",
    "num_steps",
    "query_batch",
    "num_heads",
    "head_dim",
    "num_subspaces",
    "pq_bits",
    "num_rvq_layers",
    "hot_capacity",
    "warm_capacity",
    "cold_capacity",
    "warm_active_steps",
    "warm_active_ratio",
    "cold_spill_active",
    "avg_warm_blocks",
    "peak_warm_blocks",
    "avg_warm_decode_tokens",
    "avg_warm_coverage_ratio",
    "avg_warm_tokens_per_processed_block",
    "avg_update_ms",
    "avg_dense_ms",
    "avg_cache_total_ms",
    "avg_cache_concat_ms",
    "avg_cache_attention_ms",
    "avg_warm_reconstruct_ms",
    "mean_output_mse",
    "mean_output_l2",
    "max_output_linf",
    "mean_score_mse",
    "mean_score_l2",
    "max_score_linf",
    "final_visible_tokens",
    "final_hot_tokens",
    "final_warm_tokens",
    "final_cold_tokens",
    "final_hot_bytes",
    "final_warm_key_pq_bytes",
    "final_warm_value_pq_bytes",
    "final_warm_key_rvq_bytes",
    "final_warm_value_rvq_bytes",
    "final_stored_bytes",
    "peak_stored_bytes",
    "peak_warm_reconstruction_fp16_bytes",
    "peak_warm_reconstruction_fp32_bytes",
    "peak_dense_kv_fp32_bytes",
    "dense_visible_kv_bytes",
    "dense_full_kv_bytes",
    "visible_savings_bytes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense attention vs. the current RFSN hot+warm cache path")
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[2048, 4096])
    parser.add_argument("--modes", nargs="+", choices=["pq", "hybrid"], default=["pq", "hybrid"])
    parser.add_argument("--warm-read-modes", nargs="+", choices=["full", "blockwise"], default=["full"])
    parser.add_argument("--warm-selection-policies", nargs="+", choices=["all", "recent"], default=["all"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-tokens", type=int, default=256)
    parser.add_argument("--warm-block-size", type=int, default=256)
    parser.add_argument("--warm-selection-blocks", type=int, default=2)
    parser.add_argument("--query-batch", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-subspaces", type=int, default=8)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--num-rvq-layers", type=int, default=4)
    parser.add_argument("--rvq-codebook-size", type=int, default=128)
    parser.add_argument("--rvq-sparsity-threshold", type=float, default=0.005)
    parser.add_argument("--hot-capacity", type=int, default=1024)
    parser.add_argument("--warm-capacity", type=int, default=8192)
    parser.add_argument("--cold-capacity", type=int, default=2000000)
    parser.add_argument("--skip-score-drift", action="store_true")
    parser.add_argument("--allow-cold-spill", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("rfsn_v10_benchmark_results.csv"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.step_tokens <= 0:
        raise ValueError("step_tokens must be positive")
    if args.warm_block_size <= 0:
        raise ValueError("warm_block_size must be positive")
    if args.warm_selection_blocks <= 0:
        raise ValueError("warm_selection_blocks must be positive")
    if args.query_batch <= 0:
        raise ValueError("query_batch must be positive")
    if args.num_heads <= 0 or args.head_dim <= 0:
        raise ValueError("num_heads and head_dim must be positive")
    if args.num_subspaces <= 0:
        raise ValueError("num_subspaces must be positive")
    if args.head_dim % args.num_subspaces != 0:
        raise ValueError("head_dim must be divisible by num_subspaces")
    if any(seq_len <= 0 for seq_len in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive")

    visible_capacity = args.hot_capacity + args.warm_capacity
    if not args.allow_cold_spill and any(seq_len > visible_capacity for seq_len in args.sequence_lengths):
        raise ValueError(
            "sequence length exceeds hot_capacity + warm_capacity; either lower the sequence length "
            "or pass --allow-cold-spill to benchmark the current no-cold-read path honestly"
        )


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def max_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(max(values))


def sample_prefix_lengths(seq_len: int, step_tokens: int) -> List[int]:
    positions = list(range(step_tokens, seq_len + 1, step_tokens))
    if not positions or positions[-1] != seq_len:
        positions.append(seq_len)
    return positions


def build_warm_block_ranges(num_warm: int, block_size_tokens: int) -> List[tuple[int, int]]:
    return [
        (start_token, min(num_warm, start_token + block_size_tokens))
        for start_token in range(0, num_warm, block_size_tokens)
    ]


def select_warm_block_ranges(
    num_warm: int,
    block_size_tokens: int,
    policy: str,
    selection_blocks: int,
) -> List[tuple[int, int]]:
    all_ranges = build_warm_block_ranges(num_warm, block_size_tokens)
    if policy == "all":
        return all_ranges
    if policy == "recent":
        return all_ranges[-min(selection_blocks, len(all_ranges)) :]
    raise ValueError(f"Unsupported warm selection policy: {policy}")


def attention_scores_np(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    if k.shape[0] == 0:
        return np.zeros((q.shape[0], q.shape[1], 0), dtype=np.float32)
    scale = k.shape[-1] ** -0.5
    return np.einsum("bhd,shd->bhs", q.astype(np.float32), k.astype(np.float32)) * scale


def error_metrics(reference: np.ndarray, candidate: np.ndarray, prefix: str) -> Dict[str, float]:
    delta = candidate.astype(np.float32) - reference.astype(np.float32)
    return {
        f"{prefix}_mse": float(np.mean(delta ** 2)),
        f"{prefix}_l2": float(np.linalg.norm(delta.reshape(-1))),
        f"{prefix}_linf": float(np.max(np.abs(delta))),
    }


def build_visible_cache_keys(
    cache: rfsn.RFSNv10KVCacheMLX,
    quantizer: rfsn.HybridQuantizerMLX,
    warm_block_token_ranges: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    key_parts: List[np.ndarray] = []
    if cache.num_hot > 0:
        key_parts.append(rfsn._mx_to_np(cache.hot_keys, np.float32))
    if cache.num_warm > 0:
        if warm_block_token_ranges is None:
            key_parts.append(rfsn._mx_to_np(cache.reconstruct_warm_keys(quantizer), np.float32))
        else:
            for start_token, end_token in warm_block_token_ranges:
                key_parts.append(
                    rfsn._mx_to_np(cache.reconstruct_warm_key_block(quantizer, start_token, end_token), np.float32)
                )
    if not key_parts:
        return np.zeros((0, cache.config.num_heads, cache.config.head_dim), dtype=np.float32)
    return np.concatenate(key_parts, axis=0)


def build_selected_dense_keys(
    prefix_keys: np.ndarray,
    hot_tokens: int,
    warm_tokens: int,
    warm_block_token_ranges: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    key_parts: List[np.ndarray] = []
    if hot_tokens > 0:
        key_parts.append(prefix_keys[:hot_tokens].astype(np.float32))
    if warm_tokens > 0:
        warm_slice = prefix_keys[hot_tokens:hot_tokens + warm_tokens].astype(np.float32)
        if warm_block_token_ranges is None:
            key_parts.append(warm_slice)
        else:
            for start_token, end_token in warm_block_token_ranges:
                key_parts.append(warm_slice[start_token:end_token])
    if not key_parts:
        return np.zeros((0, prefix_keys.shape[1], prefix_keys.shape[2]), dtype=np.float32)
    return np.concatenate(key_parts, axis=0)


def build_config(args: argparse.Namespace, mode: str, disk_cache_dir: Path) -> rfsn.RFSNConfig:
    rvq_layers = 0 if mode == "pq" else args.num_rvq_layers
    return rfsn.RFSNConfig(
        hidden_dim=args.num_heads * args.head_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_layers=1,
        num_subspaces=args.num_subspaces,
        pq_bits=args.pq_bits,
        subspace_dim=args.head_dim // args.num_subspaces,
        num_rvq_layers=rvq_layers,
        rvq_codebook_size=args.rvq_codebook_size,
        rvq_sparsity_threshold=args.rvq_sparsity_threshold,
        hot_capacity=args.hot_capacity,
        warm_capacity=args.warm_capacity,
        cold_capacity=args.cold_capacity,
        block_size_seq=args.warm_block_size,
        disk_cache_dir=str(disk_cache_dir),
    )


def run_trial(
    args: argparse.Namespace,
    mode: str,
    warm_read_mode: str,
    warm_selection_policy: str,
    seq_len: int,
    trial_index: int,
) -> Dict[str, float | int | str]:
    seed = args.seed + trial_index
    rng = np.random.default_rng(seed)
    prefix_lengths = sample_prefix_lengths(seq_len, args.step_tokens)

    with tempfile.TemporaryDirectory(prefix="rfsn_bench_") as tmpdir:
        config = build_config(args, mode, Path(tmpdir))
        quantizer = rfsn.HybridQuantizerMLX(config)
        cache = rfsn.RFSNv10KVCacheMLX(config, layer_idx=0)

        keys = rng.standard_normal((seq_len, config.num_heads, config.head_dim)).astype(np.float32)
        values = rng.standard_normal((seq_len, config.num_heads, config.head_dim)).astype(np.float32)
        queries = rng.standard_normal((len(prefix_lengths), args.query_batch, config.num_heads, config.head_dim)).astype(np.float32)

        update_ms_values: List[float] = []
        dense_ms_values: List[float] = []
        cache_total_ms_values: List[float] = []
        cache_concat_ms_values: List[float] = []
        cache_attention_ms_values: List[float] = []
        warm_reconstruct_ms_values: List[float] = []
        output_mse_values: List[float] = []
        output_l2_values: List[float] = []
        output_linf_values: List[float] = []
        score_mse_values: List[float] = []
        score_l2_values: List[float] = []
        score_linf_values: List[float] = []

        peak_stored_bytes = 0
        peak_warm_reconstruction_fp16_bytes = 0
        peak_warm_reconstruction_fp32_bytes = 0
        peak_dense_kv_fp32_bytes = 0
        warm_active_steps = 0
        cold_spill_active = 0
        warm_blocks_values: List[float] = []
        warm_decode_tokens_values: List[float] = []
        warm_coverage_ratio_values: List[float] = []
        warm_tokens_per_block_values: List[float] = []
        cursor = 0

        for step_index, prefix_len in enumerate(prefix_lengths):
            if prefix_len > cursor:
                new_keys = keys[cursor:prefix_len]
                new_values = values[cursor:prefix_len]
                update_start = time.perf_counter()
                cache.update(
                    rfsn._np_to_mx(new_keys, dtype=rfsn._mx_dtype("float16")),
                    rfsn._np_to_mx(new_values, dtype=rfsn._mx_dtype("float16")),
                    quantizer,
                    disk_dir=Path(tmpdir),
                )
                update_ms_values.append((time.perf_counter() - update_start) * 1000.0)
                cursor = prefix_len

            query = queries[step_index]

            dense_start = time.perf_counter()
            dense_output = rfsn.dense_attention_reference_np(query, keys[:prefix_len], values[:prefix_len])
            dense_ms_values.append((time.perf_counter() - dense_start) * 1000.0)

            warm_block_token_ranges = None
            if warm_read_mode == "blockwise":
                warm_block_token_ranges = select_warm_block_ranges(
                    cache.num_warm,
                    args.warm_block_size,
                    warm_selection_policy,
                    args.warm_selection_blocks,
                )

            cache_query = rfsn._np_to_mx(query, dtype=rfsn._mx_dtype("float16"))
            cache_output_mx, cache_metrics = cache.attention_forward_profile(
                cache_query,
                quantizer,
                warm_read_mode=warm_read_mode,
                warm_block_size_tokens=args.warm_block_size,
                warm_block_token_ranges=warm_block_token_ranges,
            )
            cache_output = rfsn._mx_to_np(cache_output_mx, np.float32)

            cache_total_ms_values.append(float(cache_metrics["total_ms"]))
            cache_concat_ms_values.append(float(cache_metrics["concat_ms"]))
            cache_attention_ms_values.append(float(cache_metrics["attention_ms"]))
            warm_reconstruct_ms_values.append(float(cache_metrics["warm_reconstruct_ms"]))
            warm_blocks_values.append(float(cache_metrics["warm_blocks"]))
            warm_decode_tokens_values.append(float(cache_metrics["warm_decode_tokens"]))

            if cache.num_warm > 0:
                warm_coverage_ratio_values.append(float(cache_metrics["warm_decode_tokens"]) / cache.num_warm)
            if float(cache_metrics["warm_blocks"]) > 0.0:
                warm_tokens_per_block_values.append(
                    float(cache_metrics["warm_decode_tokens"]) / float(cache_metrics["warm_blocks"])
                )

            output_metrics = error_metrics(dense_output, cache_output, "output")
            output_mse_values.append(output_metrics["output_mse"])
            output_l2_values.append(output_metrics["output_l2"])
            output_linf_values.append(output_metrics["output_linf"])

            if args.skip_score_drift:
                score_mse_values.append(float("nan"))
                score_l2_values.append(float("nan"))
                score_linf_values.append(float("nan"))
            else:
                dense_score_keys = build_selected_dense_keys(
                    keys[:prefix_len],
                    cache.num_hot,
                    cache.num_warm,
                    warm_block_token_ranges,
                )
                dense_scores = attention_scores_np(query, dense_score_keys)
                cache_scores = attention_scores_np(
                    query,
                    build_visible_cache_keys(cache, quantizer, warm_block_token_ranges),
                )
                score_metrics = error_metrics(dense_scores, cache_scores, "score")
                score_mse_values.append(score_metrics["score_mse"])
                score_l2_values.append(score_metrics["score_l2"])
                score_linf_values.append(score_metrics["score_linf"])

            warm_active_steps += int(cache.num_warm > 0)
            cold_spill_active = max(cold_spill_active, int(cache.num_cold > 0))
            peak_stored_bytes = max(peak_stored_bytes, int(cache_metrics["stored_bytes"]))
            peak_warm_reconstruction_fp16_bytes = max(
                peak_warm_reconstruction_fp16_bytes,
                int(cache_metrics["warm_reconstruction_fp16_bytes"]),
            )
            peak_warm_reconstruction_fp32_bytes = max(
                peak_warm_reconstruction_fp32_bytes,
                int(cache_metrics["warm_reconstruction_fp32_bytes"]),
            )
            peak_dense_kv_fp32_bytes = max(
                peak_dense_kv_fp32_bytes,
                int(cache_metrics["dense_kv_fp32_bytes"]),
            )

        final_usage = cache.memory_usage_bytes()
        final_stored_bytes = int(
            final_usage["hot_bytes"]
            + final_usage["warm_key_pq_bytes"]
            + final_usage["warm_value_pq_bytes"]
            + final_usage["warm_key_rvq_bytes"]
            + final_usage["warm_value_rvq_bytes"]
        )
        visible_tokens = cache.num_hot + cache.num_warm
        dense_visible_kv_bytes = int(visible_tokens * config.num_heads * config.head_dim * 2 * 2)
        dense_full_kv_bytes = int(seq_len * config.num_heads * config.head_dim * 2 * 2)

        return {
            "mode": mode,
            "warm_read_mode": warm_read_mode,
            "warm_block_size_tokens": args.warm_block_size,
            "warm_selection_policy": warm_selection_policy,
            "warm_selection_blocks": 0 if warm_selection_policy == "all" else args.warm_selection_blocks,
            "trial_index": trial_index,
            "seed": seed,
            "seq_len": seq_len,
            "step_tokens": args.step_tokens,
            "num_steps": len(prefix_lengths),
            "query_batch": args.query_batch,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "num_subspaces": config.num_subspaces,
            "pq_bits": config.pq_bits,
            "num_rvq_layers": config.num_rvq_layers,
            "hot_capacity": config.hot_capacity,
            "warm_capacity": config.warm_capacity,
            "cold_capacity": config.cold_capacity,
            "warm_active_steps": warm_active_steps,
            "warm_active_ratio": float(warm_active_steps / len(prefix_lengths)),
            "cold_spill_active": cold_spill_active,
            "avg_warm_blocks": mean_or_zero(warm_blocks_values),
            "peak_warm_blocks": max_or_zero(warm_blocks_values),
            "avg_warm_decode_tokens": mean_or_zero(warm_decode_tokens_values),
            "avg_warm_coverage_ratio": mean_or_zero(warm_coverage_ratio_values),
            "avg_warm_tokens_per_processed_block": mean_or_zero(warm_tokens_per_block_values),
            "avg_update_ms": mean_or_zero(update_ms_values),
            "avg_dense_ms": mean_or_zero(dense_ms_values),
            "avg_cache_total_ms": mean_or_zero(cache_total_ms_values),
            "avg_cache_concat_ms": mean_or_zero(cache_concat_ms_values),
            "avg_cache_attention_ms": mean_or_zero(cache_attention_ms_values),
            "avg_warm_reconstruct_ms": mean_or_zero(warm_reconstruct_ms_values),
            "mean_output_mse": mean_or_zero(output_mse_values),
            "mean_output_l2": mean_or_zero(output_l2_values),
            "max_output_linf": max_or_zero(output_linf_values),
            "mean_score_mse": float(np.nanmean(score_mse_values)) if score_mse_values else float("nan"),
            "mean_score_l2": float(np.nanmean(score_l2_values)) if score_l2_values else float("nan"),
            "max_score_linf": float(np.nanmax(score_linf_values)) if score_linf_values else float("nan"),
            "final_visible_tokens": visible_tokens,
            "final_hot_tokens": cache.num_hot,
            "final_warm_tokens": cache.num_warm,
            "final_cold_tokens": cache.num_cold,
            "final_hot_bytes": int(final_usage["hot_bytes"]),
            "final_warm_key_pq_bytes": int(final_usage["warm_key_pq_bytes"]),
            "final_warm_value_pq_bytes": int(final_usage["warm_value_pq_bytes"]),
            "final_warm_key_rvq_bytes": int(final_usage["warm_key_rvq_bytes"]),
            "final_warm_value_rvq_bytes": int(final_usage["warm_value_rvq_bytes"]),
            "final_stored_bytes": final_stored_bytes,
            "peak_stored_bytes": peak_stored_bytes,
            "peak_warm_reconstruction_fp16_bytes": peak_warm_reconstruction_fp16_bytes,
            "peak_warm_reconstruction_fp32_bytes": peak_warm_reconstruction_fp32_bytes,
            "peak_dense_kv_fp32_bytes": peak_dense_kv_fp32_bytes,
            "dense_visible_kv_bytes": dense_visible_kv_bytes,
            "dense_full_kv_bytes": dense_full_kv_bytes,
            "visible_savings_bytes": dense_visible_kv_bytes - final_stored_bytes,
        }


def write_rows(output_path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_args(args)

    rows: List[Dict[str, float | int | str]] = []
    for seq_len in args.sequence_lengths:
        for mode in args.modes:
            for warm_read_mode in args.warm_read_modes:
                selection_policies = ["all"] if warm_read_mode == "full" else args.warm_selection_policies
                for warm_selection_policy in selection_policies:
                    for trial_index in range(args.trials):
                        row = run_trial(args, mode, warm_read_mode, warm_selection_policy, seq_len, trial_index)
                        rows.append(row)
                        print(
                            "mode={mode}/{warm_mode}/{policy} seq={seq} trial={trial} dense_ms={dense:.3f} cache_ms={cache:.3f} "
                            "warm_ms={warm:.3f} warm_blocks={blocks:.1f} coverage={coverage:.2f} output_mse={mse:.6e} cold={cold}".format(
                                mode=row["mode"],
                                warm_mode=row["warm_read_mode"],
                                policy=row["warm_selection_policy"],
                                seq=row["seq_len"],
                                trial=row["trial_index"],
                                dense=row["avg_dense_ms"],
                                cache=row["avg_cache_total_ms"],
                                warm=row["avg_warm_reconstruct_ms"],
                                blocks=row["avg_warm_blocks"],
                                coverage=row["avg_warm_coverage_ratio"],
                                mse=row["mean_output_mse"],
                                cold=row["cold_spill_active"],
                            )
                        )

    write_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()