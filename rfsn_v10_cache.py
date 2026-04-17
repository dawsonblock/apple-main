from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rfsn_v10_attention as attention
import rfsn_v10_codec as codec
import rfsn_v10_common as common


class RFSNv10KVCacheMLX:
    def __init__(self, config: common.RFSNConfig, layer_idx: int = 0):
        common._require_mlx()
        self.config = config
        self.layer_idx = layer_idx
        self.hot_capacity = config.hot_capacity
        self.warm_capacity = config.warm_capacity
        self.hot_cache_dtype_name = config.hot_cache_dtype
        self.hot_cache_dtype = common._resolve_cache_storage_dtype(self.hot_cache_dtype_name)

        self.hot_keys = common.mx.zeros(
            (self.hot_capacity, config.num_heads, config.head_dim),
            dtype=self.hot_cache_dtype,
        )
        self.hot_values = common.mx.zeros(
            (self.hot_capacity, config.num_heads, config.head_dim),
            dtype=self.hot_cache_dtype,
        )

        self.warm_key_pq_codes = common.mx.zeros(
            (0, config.num_heads, config.num_subspaces),
            dtype=common._mx_dtype("uint8"),
        )
        self.warm_key_rvq_codes = common.mx.zeros(
            (0, config.num_rvq_layers),
            dtype=common._mx_dtype("uint16"),
        )
        self.warm_key_rvq_mask = common.mx.zeros(
            (0, config.num_heads),
            dtype=common._mx_dtype("bool_"),
        )
        self.warm_key_rvq_offsets = common.mx.zeros((0,), dtype=common._mx_dtype("int32"))
        self.warm_key_rvq_entry_mask = common.mx.zeros((0,), dtype=common._mx_dtype("bool_"))

        self.warm_value_pq_codes = common.mx.zeros(
            (0, config.num_heads, config.num_subspaces),
            dtype=common._mx_dtype("uint8"),
        )
        self.warm_value_rvq_codes = common.mx.zeros(
            (0, config.num_rvq_layers),
            dtype=common._mx_dtype("uint16"),
        )
        self.warm_value_rvq_mask = common.mx.zeros(
            (0, config.num_heads),
            dtype=common._mx_dtype("bool_"),
        )
        self.warm_value_rvq_offsets = common.mx.zeros((0,), dtype=common._mx_dtype("int32"))
        self.warm_value_rvq_entry_mask = common.mx.zeros((0,), dtype=common._mx_dtype("bool_"))

        self.cold_chunk_paths: List[Path] = []
        self.num_hot = 0
        self.num_warm = 0
        self.num_cold = 0
        self.total_tokens = 0
        self.attention = attention.RFSNHybridAttentionMLX(config)

    @property
    def current_tier(self) -> str:
        if self.num_hot < self.hot_capacity:
            return "hot"
        if self.num_warm < self.warm_capacity:
            return "warm"
        return "cold"

    def _write_hot_rows(self, buffer, start: int, values):
        if values.shape[0] == 0:
            return buffer
        indices = common.mx.arange(start, start + values.shape[0], dtype=common._mx_dtype("int32"))
        cast_values = values.astype(buffer.dtype)
        return buffer.at[indices].add(cast_values - buffer[indices])

    def update(
        self,
        new_keys,
        new_values,
        quantizer: codec.HybridQuantizerMLX,
        disk_dir: Optional[Path] = None,
    ) -> None:
        if new_keys.shape != new_values.shape:
            raise ValueError("Keys and values must have identical shapes")

        start = 0
        total = new_keys.shape[0]

        hot_space = max(0, self.hot_capacity - self.num_hot)
        hot_take = min(total - start, hot_space)
        if hot_take > 0:
            hot_keys = new_keys[start:start + hot_take]
            hot_values = new_values[start:start + hot_take]
            self.hot_keys = self._write_hot_rows(self.hot_keys, self.num_hot, hot_keys)
            self.hot_values = self._write_hot_rows(self.hot_values, self.num_hot, hot_values)
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

    def _encode_warm_batch(
        self,
        batch,
        quantizer: codec.HybridQuantizerMLX,
        base_token_offset: int,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        s, h, d = batch.shape
        flat = batch.reshape(s * h, d)
        pq_codes, rvq_codes, rvq_mask, rvq_offsets, rvq_entry_mask = quantizer.encode(flat)
        pq_codes_reshaped = pq_codes.reshape(s, h, self.config.num_subspaces)
        rvq_mask_reshaped = rvq_mask.reshape(s, h)
        shifted_offsets = rvq_offsets + int(base_token_offset * h)
        return pq_codes_reshaped, rvq_codes, rvq_mask_reshaped, shifted_offsets, rvq_entry_mask

    def _add_to_warm(self, keys, values, quantizer: codec.HybridQuantizerMLX) -> None:
        s = keys.shape[0]
        base_offset = self.num_warm

        key_pq, key_rvq, key_mask, key_offsets, key_entry_mask = self._encode_warm_batch(keys, quantizer, base_offset)
        value_pq, value_rvq, value_mask, value_offsets, value_entry_mask = self._encode_warm_batch(values, quantizer, base_offset)

        self.warm_key_pq_codes = common.mx.concatenate([self.warm_key_pq_codes, key_pq], axis=0)
        self.warm_key_rvq_codes = common.mx.concatenate([self.warm_key_rvq_codes, key_rvq], axis=0)
        self.warm_key_rvq_mask = common.mx.concatenate([self.warm_key_rvq_mask, key_mask], axis=0)
        self.warm_key_rvq_offsets = common.mx.concatenate([self.warm_key_rvq_offsets, key_offsets], axis=0)
        self.warm_key_rvq_entry_mask = common.mx.concatenate([self.warm_key_rvq_entry_mask, key_entry_mask], axis=0)

        self.warm_value_pq_codes = common.mx.concatenate([self.warm_value_pq_codes, value_pq], axis=0)
        self.warm_value_rvq_codes = common.mx.concatenate([self.warm_value_rvq_codes, value_rvq], axis=0)
        self.warm_value_rvq_mask = common.mx.concatenate([self.warm_value_rvq_mask, value_mask], axis=0)
        self.warm_value_rvq_offsets = common.mx.concatenate([self.warm_value_rvq_offsets, value_offsets], axis=0)
        self.warm_value_rvq_entry_mask = common.mx.concatenate([self.warm_value_rvq_entry_mask, value_entry_mask], axis=0)

        self.num_warm += s

    def _add_to_cold(
        self,
        keys,
        values,
        quantizer: codec.HybridQuantizerMLX,
        disk_dir: Optional[Path],
    ) -> None:
        disk_dir = Path(self.config.disk_cache_dir) if disk_dir is None else Path(disk_dir)
        disk_dir.mkdir(parents=True, exist_ok=True)

        s, h, d = keys.shape
        key_flat = keys.reshape(s * h, d)
        value_flat = values.reshape(s * h, d)

        key_pq, key_rvq, key_mask, key_offsets, key_entry_mask = quantizer.encode(key_flat)
        value_pq, value_rvq, value_mask, value_offsets, value_entry_mask = quantizer.encode(value_flat)

        chunk_id = len(self.cold_chunk_paths)
        chunk_path = disk_dir / f"layer{self.layer_idx}_chunk{chunk_id}.npz"
        np.savez_compressed(
            chunk_path,
            seq_tokens=np.array([s], dtype=np.int32),
            num_heads=np.array([h], dtype=np.int32),
            head_dim=np.array([d], dtype=np.int32),
            key_pq_codes=common._mx_to_np(key_pq, np.uint8),
            key_rvq_codes=common._mx_to_np(key_rvq, np.uint16),
            key_rvq_mask=common._mx_to_np(key_mask, np.bool_),
            key_rvq_offsets=common._mx_to_np(key_offsets, np.int32),
            key_rvq_entry_mask=common._mx_to_np(key_entry_mask, np.bool_),
            value_pq_codes=common._mx_to_np(value_pq, np.uint8),
            value_rvq_codes=common._mx_to_np(value_rvq, np.uint16),
            value_rvq_mask=common._mx_to_np(value_mask, np.bool_),
            value_rvq_offsets=common._mx_to_np(value_offsets, np.int32),
            value_rvq_entry_mask=common._mx_to_np(value_entry_mask, np.bool_),
        )
        self.cold_chunk_paths.append(chunk_path)
        self.num_cold += s

    def load_cold_chunk(self, chunk_id: int) -> Dict[str, Any]:
        path = self.cold_chunk_paths[chunk_id]
        data = np.load(path)
        return {key: common._np_to_mx(np.array(value)) for key, value in data.items()}

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

    def _build_blockwise_numpy_decode_state(self, quantizer: codec.HybridQuantizerMLX) -> Dict[str, Any]:
        return {
            "pq_codebooks": common._mx_to_np(quantizer.pq.codebooks, np.float32),
            "rvq_codebooks": common._mx_to_np(quantizer.rvq.codebooks, np.float32),
            "key": {
                "pq_codes": common._mx_to_np(self.warm_key_pq_codes, np.uint8),
                "rvq_codes": common._mx_to_np(self.warm_key_rvq_codes, np.int32),
                "rvq_offsets": common._mx_to_np(self.warm_key_rvq_offsets, np.int32),
            },
            "value": {
                "pq_codes": common._mx_to_np(self.warm_value_pq_codes, np.uint8),
                "rvq_codes": common._mx_to_np(self.warm_value_rvq_codes, np.int32),
                "rvq_offsets": common._mx_to_np(self.warm_value_rvq_offsets, np.int32),
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
        decoded = codec._hybrid_decode_np(
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

    def reconstruct_warm_keys(self, quantizer: codec.HybridQuantizerMLX):
        if self.num_warm == 0:
            return common.mx.zeros(
                (0, self.config.num_heads, self.config.head_dim),
                dtype=common._mx_dtype("float16"),
            )
        flat_codes = self.warm_key_pq_codes.reshape(
            self.num_warm * self.config.num_heads,
            self.config.num_subspaces,
        )
        flat_mask = self.warm_key_rvq_mask.reshape(self.num_warm * self.config.num_heads)
        recon = quantizer.decode(
            flat_codes,
            self.warm_key_rvq_codes,
            flat_mask,
            self.warm_key_rvq_offsets,
            self.warm_key_rvq_entry_mask,
        )
        return recon.reshape(self.num_warm, self.config.num_heads, self.config.head_dim)

    def reconstruct_warm_values(self, quantizer: codec.HybridQuantizerMLX):
        if self.num_warm == 0:
            return common.mx.zeros(
                (0, self.config.num_heads, self.config.head_dim),
                dtype=common._mx_dtype("float16"),
            )
        flat_codes = self.warm_value_pq_codes.reshape(
            self.num_warm * self.config.num_heads,
            self.config.num_subspaces,
        )
        flat_mask = self.warm_value_rvq_mask.reshape(self.num_warm * self.config.num_heads)
        recon = quantizer.decode(
            flat_codes,
            self.warm_value_rvq_codes,
            flat_mask,
            self.warm_value_rvq_offsets,
            self.warm_value_rvq_entry_mask,
        )
        return recon.reshape(self.num_warm, self.config.num_heads, self.config.head_dim)

    def _reconstruct_warm_component_block(
        self,
        pq_codes,
        rvq_codes,
        rvq_mask,
        rvq_offsets,
        rvq_entry_mask,
        quantizer: codec.HybridQuantizerMLX,
        start_token: int,
        end_token: int,
    ):
        if start_token < 0 or end_token < start_token or end_token > int(pq_codes.shape[0]):
            raise ValueError("Warm block token range is invalid")

        token_count = end_token - start_token
        if token_count == 0:
            return common.mx.zeros(
                (0, self.config.num_heads, self.config.head_dim),
                dtype=common._mx_dtype("float16"),
            )

        row_start = start_token * self.config.num_heads
        row_end = end_token * self.config.num_heads
        block_pq_codes = pq_codes[start_token:end_token].reshape(
            token_count * self.config.num_heads,
            self.config.num_subspaces,
        )
        block_mask = rvq_mask[start_token:end_token].reshape(token_count * self.config.num_heads)
        local_entry_mask = rvq_entry_mask & (rvq_offsets >= row_start) & (rvq_offsets < row_end)
        local_offsets = common.mx.where(
            local_entry_mask,
            rvq_offsets - row_start,
            common.mx.zeros_like(rvq_offsets),
        )
        decoded = quantizer.decode(
            block_pq_codes,
            rvq_codes,
            block_mask,
            local_offsets.astype(common._mx_dtype("int32")),
            local_entry_mask,
        )
        return decoded.reshape(token_count, self.config.num_heads, self.config.head_dim)

    def reconstruct_warm_key_block(
        self,
        quantizer: codec.HybridQuantizerMLX,
        start_token: int,
        end_token: int,
    ):
        return self._reconstruct_warm_component_block(
            self.warm_key_pq_codes,
            self.warm_key_rvq_codes,
            self.warm_key_rvq_mask,
            self.warm_key_rvq_offsets,
            self.warm_key_rvq_entry_mask,
            quantizer,
            start_token,
            end_token,
        )

    def reconstruct_warm_value_block(
        self,
        quantizer: codec.HybridQuantizerMLX,
        start_token: int,
        end_token: int,
    ):
        return self._reconstruct_warm_component_block(
            self.warm_value_pq_codes,
            self.warm_value_rvq_codes,
            self.warm_value_rvq_mask,
            self.warm_value_rvq_offsets,
            self.warm_value_rvq_entry_mask,
            quantizer,
            start_token,
            end_token,
        )

    def _attention_forward_impl(
        self,
        q,
        quantizer: codec.HybridQuantizerMLX,
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

        q_mx = q.astype(common._mx_dtype("float32"))
        metrics: Dict[str, float | int | str] = {}

        if collect_metrics:
            stored = self.memory_usage_bytes()
            metrics = {
                "warm_read_mode": warm_read_mode,
                "warm_block_size_tokens": int(block_size_tokens),
                "query_batch": int(q_mx.shape[0]),
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
                "query_fp32_bytes": int(np.prod(q_mx.shape) * 4),
                "stored_bytes": int(
                    stored["hot_bytes"]
                    + stored["warm_key_pq_bytes"]
                    + stored["warm_value_pq_bytes"]
                    + stored["warm_key_rvq_bytes"]
                    + stored["warm_value_rvq_bytes"]
                ),
            }
            total_start = time.perf_counter()

        hot_keys_mx = None
        hot_values_mx = None

        if self.num_hot > 0:
            hot_keys_mx = self.hot_keys[:self.num_hot].astype(common._mx_dtype("float32"))
            hot_values_mx = self.hot_values[:self.num_hot].astype(common._mx_dtype("float32"))

        if warm_read_mode == "full":
            key_parts = []
            value_parts = []
            if hot_keys_mx is not None and hot_values_mx is not None:
                key_parts.append(hot_keys_mx)
                value_parts.append(hot_values_mx)

            if self.num_warm > 0:
                if collect_metrics:
                    reconstruct_start = time.perf_counter()

                warm_keys = self.reconstruct_warm_keys(quantizer).astype(common._mx_dtype("float32"))
                warm_values = self.reconstruct_warm_values(quantizer).astype(common._mx_dtype("float32"))
                if collect_metrics:
                    common._force_eval(warm_keys, warm_values)

                key_parts.append(warm_keys)
                value_parts.append(warm_values)

                if collect_metrics:
                    metrics["warm_reconstruct_ms"] = (time.perf_counter() - reconstruct_start) * 1000.0
                    metrics["warm_blocks"] = int(self.num_warm > 0)
                    metrics["warm_decode_tokens"] = int(self.num_warm)
                    metrics["warm_reconstruction_fp16_bytes"] = int(
                        self.num_warm * self.config.num_heads * self.config.head_dim * 2 * 2
                    )
                    metrics["warm_reconstruction_fp32_bytes"] = int(
                        self.num_warm * self.config.num_heads * self.config.head_dim * 4 * 2
                    )

            if not key_parts:
                out = common.mx.zeros(q.shape, dtype=common._mx_dtype("float16"))
                if collect_metrics:
                    metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
                return out, metrics

            if collect_metrics:
                concat_start = time.perf_counter()

            all_keys = key_parts[0] if len(key_parts) == 1 else common.mx.concatenate(key_parts, axis=0)
            all_values = value_parts[0] if len(value_parts) == 1 else common.mx.concatenate(value_parts, axis=0)
            if collect_metrics:
                common._force_eval(all_keys, all_values)
                metrics["concat_ms"] = (time.perf_counter() - concat_start) * 1000.0
                metrics["dense_kv_fp32_bytes"] = int(
                    all_keys.shape[0] * self.config.num_heads * self.config.head_dim * 4 * 2
                )
                attention_start = time.perf_counter()

            out = attention.dense_attention_reference_mx(q_mx, all_keys, all_values)

            if collect_metrics:
                common._force_eval(out)
                metrics["attention_ms"] = (time.perf_counter() - attention_start) * 1000.0
                metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0

            return out.astype(common._mx_dtype("float16")), metrics

        if self.num_hot == 0 and self.num_warm == 0:
            out = common.mx.zeros(q.shape, dtype=common._mx_dtype("float16"))
            if collect_metrics:
                metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            return out, metrics

        running_max = common.mx.full(
            (q_mx.shape[0], q_mx.shape[1]),
            -float("inf"),
            dtype=common._mx_dtype("float32"),
        )
        running_sum = common.mx.zeros((q_mx.shape[0], q_mx.shape[1]), dtype=common._mx_dtype("float32"))
        running_out = common.mx.zeros(q_mx.shape, dtype=common._mx_dtype("float32"))

        hot_fp32_bytes = 0
        if hot_keys_mx is not None and hot_values_mx is not None:
            hot_fp32_bytes = int(self.num_hot * self.config.num_heads * self.config.head_dim * 4 * 2)
            if collect_metrics:
                metrics["dense_kv_fp32_bytes"] = hot_fp32_bytes
                attention_start = time.perf_counter()
            running_max, running_sum, running_out = attention._streaming_attention_update_mx(
                q_mx,
                hot_keys_mx,
                hot_values_mx,
                running_max,
                running_sum,
                running_out,
            )
            if collect_metrics:
                common._force_eval(running_max, running_sum, running_out)
                metrics["attention_ms"] = float(metrics["attention_ms"]) + (time.perf_counter() - attention_start) * 1000.0

        warm_ranges = (
            self._default_warm_block_token_ranges(block_size_tokens)
            if warm_block_token_ranges is None
            else list(warm_block_token_ranges)
        )

        for start_token, end_token in warm_ranges:
            token_count = end_token - start_token

            if collect_metrics:
                reconstruct_start = time.perf_counter()

            warm_keys = self.reconstruct_warm_key_block(
                quantizer,
                start_token,
                end_token,
            ).astype(common._mx_dtype("float32"))
            warm_values = self.reconstruct_warm_value_block(
                quantizer,
                start_token,
                end_token,
            ).astype(common._mx_dtype("float32"))

            if collect_metrics:
                common._force_eval(warm_keys, warm_values)
                metrics["warm_reconstruct_ms"] = float(metrics["warm_reconstruct_ms"]) + (time.perf_counter() - reconstruct_start) * 1000.0
                metrics["warm_blocks"] = int(metrics["warm_blocks"]) + 1
                metrics["warm_decode_tokens"] = int(metrics["warm_decode_tokens"]) + token_count
                metrics["warm_reconstruction_fp16_bytes"] = max(
                    int(metrics["warm_reconstruction_fp16_bytes"]),
                    token_count * self.config.num_heads * self.config.head_dim * 2 * 2,
                )
                block_fp32_bytes = int(token_count * self.config.num_heads * self.config.head_dim * 4 * 2)
                metrics["warm_reconstruction_fp32_bytes"] = max(
                    int(metrics["warm_reconstruction_fp32_bytes"]),
                    block_fp32_bytes,
                )
                metrics["dense_kv_fp32_bytes"] = max(
                    int(metrics["dense_kv_fp32_bytes"]),
                    hot_fp32_bytes + block_fp32_bytes,
                )
                attention_start = time.perf_counter()

            running_max, running_sum, running_out = attention._streaming_attention_update_mx(
                q_mx,
                warm_keys,
                warm_values,
                running_max,
                running_sum,
                running_out,
            )

            if collect_metrics:
                common._force_eval(running_max, running_sum, running_out)
                metrics["attention_ms"] = float(metrics["attention_ms"]) + (time.perf_counter() - attention_start) * 1000.0

        denom = common.mx.where(running_sum == 0.0, common.mx.ones_like(running_sum), running_sum)
        out = (running_out / denom[:, :, None]).astype(common._mx_dtype("float32"))

        if collect_metrics:
            common._force_eval(out)
            metrics["total_ms"] = (time.perf_counter() - total_start) * 1000.0

        return out.astype(common._mx_dtype("float16")), metrics

    def attention_forward(
        self,
        q,
        quantizer: codec.HybridQuantizerMLX,
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
        quantizer: codec.HybridQuantizerMLX,
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
        rvq_code_bytes = self.config.num_rvq_layers * common._dtype_nbytes("uint16")
        rvq_offset_bytes = common._dtype_nbytes("int32")
        rvq_entry_bytes = common._dtype_nbytes("bool_")
        rvq_mask_bytes = self.config.num_heads * common._dtype_nbytes("bool_")
        return {
            "hot_bytes": self.num_hot * self.config.num_heads * self.config.head_dim * common._dtype_nbytes(self.hot_cache_dtype_name) * 2,
            "warm_key_pq_bytes": self.num_warm * self.config.num_heads * self.config.num_subspaces,
            "warm_value_pq_bytes": self.num_warm * self.config.num_heads * self.config.num_subspaces,
            "warm_key_rvq_bytes": warm_key_rvq_rows * (rvq_code_bytes + rvq_offset_bytes + rvq_entry_bytes) + self.num_warm * rvq_mask_bytes,
            "warm_value_rvq_bytes": warm_value_rvq_rows * (rvq_code_bytes + rvq_offset_bytes + rvq_entry_bytes) + self.num_warm * rvq_mask_bytes,
            "cold_chunks": len(self.cold_chunk_paths),
            "cold_tokens": self.num_cold,
        }