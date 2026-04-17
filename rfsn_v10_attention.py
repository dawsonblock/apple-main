from __future__ import annotations

from typing import Tuple

import numpy as np

import rfsn_v10_common as common


def dense_attention_reference_mx(q, k, v):
    if k.shape[0] == 0:
        return common.mx.zeros(q.shape, dtype=common._mx_dtype("float32"))
    q_f32 = q.astype(common._mx_dtype("float32"))
    k_f32 = k.astype(common._mx_dtype("float32"))
    v_f32 = v.astype(common._mx_dtype("float32"))
    scale = k.shape[-1] ** -0.5
    scores = common.mx.einsum("bhd,shd->bhs", q_f32, k_f32) * scale
    weights = common._stable_softmax_mx(scores, axis=-1)
    return common.mx.einsum("bhs,shd->bhd", weights, v_f32).astype(common._mx_dtype("float32"))


def _streaming_attention_update_mx(
    q,
    k,
    v,
    running_max,
    running_sum,
    running_out,
):
    if k.shape[0] == 0:
        return running_max, running_sum, running_out

    q_f32 = q.astype(common._mx_dtype("float32"))
    k_f32 = k.astype(common._mx_dtype("float32"))
    v_f32 = v.astype(common._mx_dtype("float32"))
    scale = k.shape[-1] ** -0.5
    scores = common.mx.einsum("bhd,shd->bhs", q_f32, k_f32) * scale
    chunk_max = common.mx.max(scores, axis=-1)
    new_max = common.mx.where(running_max > chunk_max, running_max, chunk_max)

    prev_rescale = common.mx.exp(running_max - new_max)
    chunk_weights = common.mx.exp(scores - new_max[:, :, None])

    running_sum = running_sum * prev_rescale + common.mx.sum(chunk_weights, axis=-1)
    running_out = (
        running_out * prev_rescale[:, :, None]
        + common.mx.einsum("bhs,shd->bhd", chunk_weights, v_f32)
    )
    return new_max, running_sum, running_out


def dense_attention_reference_np(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    if k.shape[0] == 0:
        return np.zeros_like(q, dtype=np.float32)
    scale = k.shape[-1] ** -0.5
    scores = np.einsum("bhd,shd->bhs", q.astype(np.float32), k.astype(np.float32)) * scale
    weights = common._stable_softmax_np(scores, axis=-1)
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


class RFSNHybridAttentionMLX(common.nn.Module):
    def __init__(self, config: common.RFSNConfig):
        common._require_mlx()
        super().__init__()
        self.config = config

    def __call__(self, q, keys, values):
        out = dense_attention_reference_mx(q, keys, values)
        return out.astype(common._mx_dtype("float16"))
