from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

import rfsn_v10_common as common


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


class ProductQuantizerMLX(common.nn.Module):
    def __init__(self, config: common.RFSNConfig):
        common._require_mlx()
        super().__init__()
        self.config = config
        self.num_subspaces = config.num_subspaces
        self.subspace_dim = config.subspace_dim
        self.codebook_size = 1 << config.pq_bits

        scale = (2.0 / self.subspace_dim) ** 0.5
        codebooks = np.random.randn(
            self.num_subspaces, self.codebook_size, self.subspace_dim
        ).astype(np.float32) * scale
        self.codebooks = common._np_to_mx(codebooks, dtype=common._mx_dtype("float16"))

    def quantize(self, vectors) -> Tuple[Any, Any]:
        vectors_f32 = vectors.astype(common._mx_dtype("float32"))
        n, d = vectors_f32.shape
        if d != self.num_subspaces * self.subspace_dim:
            raise ValueError(f"Expected dim {self.num_subspaces * self.subspace_dim}, got {d}")

        vectors_sub = vectors_f32.reshape(n, self.num_subspaces, self.subspace_dim)
        codebooks_f32 = self.codebooks.astype(common._mx_dtype("float32"))
        dists = common.mx.sum(
            (vectors_sub[:, :, None, :] - codebooks_f32[None, :, :, :]) ** 2,
            axis=-1,
        )
        codes = common.mx.argmin(dists, axis=-1).astype(common._mx_dtype("uint8"))

        gathered = common.mx.take_along_axis(
            codebooks_f32,
            common.mx.transpose(codes.astype(common._mx_dtype("int32")), (1, 0))[:, :, None],
            axis=1,
        )
        reconstructed = common.mx.transpose(gathered, (1, 0, 2)).reshape(n, d)
        residuals = vectors_f32 - reconstructed
        return codes, residuals.astype(vectors.dtype)

    def decode(self, codes):
        n = codes.shape[0]
        codebooks_f32 = self.codebooks.astype(common._mx_dtype("float32"))
        gathered = common.mx.take_along_axis(
            codebooks_f32,
            common.mx.transpose(codes.astype(common._mx_dtype("int32")), (1, 0))[:, :, None],
            axis=1,
        )
        out = common.mx.transpose(gathered, (1, 0, 2)).reshape(
            n,
            self.num_subspaces * self.subspace_dim,
        )
        return out.astype(common._mx_dtype("float16"))


class ResidualVQMLX(common.nn.Module):
    def __init__(self, config: common.RFSNConfig):
        common._require_mlx()
        super().__init__()
        self.config = config
        self.num_layers = config.num_rvq_layers
        self.codebook_size = config.rvq_codebook_size
        self.head_dim = config.head_dim
        self.sparsity_threshold = config.rvq_sparsity_threshold
        self.max_active = config.rvq_max_active

        scale = (2.0 / self.head_dim) ** 0.5
        codebooks = np.random.randn(
            self.num_layers, self.codebook_size, self.head_dim
        ).astype(np.float32) * scale
        self.codebooks = common._np_to_mx(codebooks, dtype=common._mx_dtype("float16"))

    def encode(self, residuals) -> Tuple[Any, Any, Any, Any]:
        residuals_f32 = residuals.astype(common._mx_dtype("float32"))
        total_rows = residuals_f32.shape[0]
        norms = common.mx.sqrt(common.mx.sum(residuals_f32 * residuals_f32, axis=1))
        row_mask = norms > self.sparsity_threshold
        scores = common.mx.where(row_mask, norms, common.mx.zeros_like(norms))

        empty_codes = common.mx.zeros((0, self.num_layers), dtype=common._mx_dtype("uint16"))
        empty_offsets = common.mx.zeros((0,), dtype=common._mx_dtype("int32"))
        empty_mask = common.mx.zeros((0,), dtype=common._mx_dtype("bool_"))
        row_mask_bool = row_mask.astype(common._mx_dtype("bool_"))

        if total_rows == 0 or self.num_layers == 0:
            return empty_codes, row_mask_bool, empty_offsets, empty_mask

        if self.max_active == 0:
            active_count = int(np.sum(common._mx_to_np(row_mask, np.int32)))
            if active_count == 0:
                return empty_codes, row_mask_bool, empty_offsets, empty_mask
            offsets = common.mx.argsort(-scores)[:active_count].astype(common._mx_dtype("int32"))
            entry_mask = common.mx.ones((active_count,), dtype=common._mx_dtype("bool_"))
        else:
            active_width = total_rows if self.max_active < 0 else min(self.max_active, total_rows)
            offsets = common.mx.argsort(-scores)[:active_width].astype(common._mx_dtype("int32"))
            entry_mask = scores[offsets] > 0.0

        safe_offsets = common.mx.where(entry_mask, offsets, common.mx.zeros_like(offsets))
        active = residuals_f32[safe_offsets]
        active = common.mx.where(entry_mask[:, None], active, common.mx.zeros_like(active))

        code_layers = []
        codebooks_f32 = self.codebooks.astype(common._mx_dtype("float32"))
        for layer in range(self.num_layers):
            cb = codebooks_f32[layer]
            dists = common.mx.sum((active[:, None, :] - cb[None, :, :]) ** 2, axis=-1)
            indices = common.mx.argmin(dists, axis=1).astype(common._mx_dtype("uint16"))
            code_layers.append(indices)
            chosen = cb[indices.astype(common._mx_dtype("int32"))]
            active = common.mx.where(entry_mask[:, None], active - chosen, active)

        codes = common.mx.stack(code_layers, axis=1)
        return codes, row_mask_bool, safe_offsets, entry_mask.astype(common._mx_dtype("bool_"))

    def decode_correction(self, total_rows: int, rvq_codes, rvq_offsets, rvq_entry_mask=None):
        out = common.mx.zeros((total_rows, self.head_dim), dtype=common._mx_dtype("float32"))
        if rvq_codes.shape[0] == 0:
            return out.astype(common._mx_dtype("float16"))

        entry_mask = (
            common.mx.ones((rvq_codes.shape[0],), dtype=common._mx_dtype("bool_"))
            if rvq_entry_mask is None
            else rvq_entry_mask.astype(common._mx_dtype("bool_"))
        )
        safe_offsets = common.mx.where(
            entry_mask,
            rvq_offsets.astype(common._mx_dtype("int32")),
            common.mx.zeros_like(rvq_offsets.astype(common._mx_dtype("int32"))),
        )

        codebooks_f32 = self.codebooks.astype(common._mx_dtype("float32"))
        gathered = common.mx.take_along_axis(
            codebooks_f32,
            common.mx.transpose(rvq_codes.astype(common._mx_dtype("int32")), (1, 0))[:, :, None],
            axis=1,
        )
        corrections = common.mx.sum(common.mx.transpose(gathered, (1, 0, 2)), axis=1)
        corrections = common.mx.where(entry_mask[:, None], corrections, common.mx.zeros_like(corrections))
        out = out.at[safe_offsets].add(corrections)
        return out.astype(common._mx_dtype("float16"))


class HybridQuantizerMLX(common.nn.Module):
    def __init__(self, config: common.RFSNConfig):
        common._require_mlx()
        super().__init__()
        self.config = config
        self.pq = ProductQuantizerMLX(config)
        self.rvq = ResidualVQMLX(config)

    def encode(self, vectors) -> Tuple[Any, Any, Any, Any, Any]:
        pq_codes, residuals = self.pq.quantize(vectors)
        rvq_codes, rvq_mask, rvq_offsets, rvq_entry_mask = self.rvq.encode(residuals)
        return pq_codes, rvq_codes, rvq_mask, rvq_offsets, rvq_entry_mask

    def decode(self, pq_codes, rvq_codes, rvq_mask, rvq_offsets, rvq_entry_mask=None):
        del rvq_mask
        pq_recon = self.pq.decode(pq_codes).astype(common._mx_dtype("float32"))
        correction = self.rvq.decode_correction(
            pq_codes.shape[0],
            rvq_codes,
            rvq_offsets,
            rvq_entry_mask,
        ).astype(common._mx_dtype("float32"))
        return (pq_recon + correction).astype(common._mx_dtype("float16"))


def calibrate_quantizer(
    quantizer: HybridQuantizerMLX,
    calibration_vectors,
    num_iterations: int = 10,
) -> Dict[str, list[float]]:
    common.logger.info(
        "Calibrating PQ codebooks with %d vectors for %d iterations",
        calibration_vectors.shape[0],
        num_iterations,
    )
    vectors = common._mx_to_np(calibration_vectors, np.float32)
    n, _ = vectors.shape
    codebook_size = 1 << quantizer.config.pq_bits
    sub_dim = quantizer.config.subspace_dim

    metrics = {"avg_distortion": []}
    codebooks = common._mx_to_np(quantizer.pq.codebooks, np.float32)

    for sub in range(quantizer.config.num_subspaces):
        s0 = sub * sub_dim
        s1 = s0 + sub_dim
        idx = np.random.choice(n, size=codebook_size, replace=n < codebook_size)
        codebooks[sub] = vectors[idx, s0:s1]

    quantizer.pq.codebooks = common._np_to_mx(codebooks, dtype=common._mx_dtype("float16"))

    for _ in range(num_iterations):
        codes, _ = quantizer.pq.quantize(common._np_to_mx(vectors, dtype=common._mx_dtype("float16")))
        codes_np = common._mx_to_np(codes, np.int32)
        recon = np.zeros_like(vectors, dtype=np.float32)
        codebooks = common._mx_to_np(quantizer.pq.codebooks, np.float32)

        for sub in range(quantizer.config.num_subspaces):
            s0 = sub * sub_dim
            s1 = s0 + sub_dim
            sub_vectors = vectors[:, s0:s1]
            for code_idx in range(codebook_size):
                mask = codes_np[:, sub] == code_idx
                if np.any(mask):
                    codebooks[sub, code_idx] = sub_vectors[mask].mean(axis=0)
            recon[:, s0:s1] = codebooks[sub][codes_np[:, sub]]

        quantizer.pq.codebooks = common._np_to_mx(codebooks, dtype=common._mx_dtype("float16"))
        mse = float(np.mean((vectors - recon) ** 2))
        metrics["avg_distortion"].append(mse)

    return metrics
