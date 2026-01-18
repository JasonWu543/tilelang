# ruff: noqa
from __future__ import annotations

from typing import Optional

import torch
import tilelang as tl
import tilelang.language as T

from index import prepare_token_indices


PASS_CONFIGS = {
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def _pick_coalesced_width(dim: int) -> int:
    for width in (8, 4, 2, 1):
        if dim % width == 0:
            return width
    return 1


@tl.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def dense_warmup_kl_kernel(
    heads: int,
    dim_mla: int,
    dim_idx: int,
    kv_group: int = 1,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    coalesced_width_mla: Optional[int] = None,
    coalesced_width_idx: Optional[int] = None,
):
    assert heads % kv_group == 0
    if coalesced_width_mla is None:
        coalesced_width_mla = _pick_coalesced_width(dim_mla)
    if coalesced_width_idx is None:
        coalesced_width_idx = _pick_coalesced_width(dim_idx)

    scale_mla = dim_mla**-0.5
    scale_idx = dim_idx**-0.5

    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    q_mla_shape = [seq_len, heads, dim_mla]
    kv_mla_shape = [seq_len, heads // kv_group, dim_mla]
    q_idx_shape = [seq_len, heads, dim_idx]
    k_idx_shape = [seq_len, heads, dim_idx]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    @T.prim_func
    def main(
        Q_mla: T.Tensor(q_mla_shape, dtype),
        KV_mla: T.Tensor(kv_mla_shape, dtype),
        Q_idx: T.Tensor(q_idx_shape, dtype),
        K_idx: T.Tensor(k_idx_shape, dtype),
        Offsets: T.Tensor(offsets_shape, T.int32),
        TokenIndices: T.Tensor(token_indices_shape, T.int32),
        Output_KL: T.Tensor([seq_len, heads], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), threads=threads) as (by, bx):
            Q_mla_s = T.alloc_shared([block_M, dim_mla], dtype)
            Q_idx_s = T.alloc_shared([block_M, dim_idx], dtype)
            K_mla_s = T.alloc_shared([block_N, dim_mla], dtype)
            K_idx_s = T.alloc_shared([block_N, dim_idx], dtype)

            S_tile = T.alloc_fragment([block_M, block_N], accum_dtype)
            T_tile = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_tile = T.alloc_fragment([block_M, block_N], accum_dtype)
            Tmp_tile = T.alloc_fragment([block_M, block_N], accum_dtype)

            m_p = T.alloc_fragment([block_M], accum_dtype)
            m_p_prev = T.alloc_fragment([block_M], accum_dtype)
            m_block_p = T.alloc_fragment([block_M], accum_dtype)
            alpha_p = T.alloc_fragment([block_M], accum_dtype)
            z_p = T.alloc_fragment([block_M], accum_dtype)
            acc_S = T.alloc_fragment([block_M], accum_dtype)
            acc_T = T.alloc_fragment([block_M], accum_dtype)

            m_q = T.alloc_fragment([block_M], accum_dtype)
            m_q_prev = T.alloc_fragment([block_M], accum_dtype)
            m_block_q = T.alloc_fragment([block_M], accum_dtype)
            alpha_q = T.alloc_fragment([block_M], accum_dtype)
            z_q = T.alloc_fragment([block_M], accum_dtype)

            sum_buf = T.alloc_fragment([block_M], accum_dtype)
            kl = T.alloc_fragment([block_M], accum_dtype)

            neg_inf = -T.infinity(accum_dtype)

            T.fill(m_p, neg_inf)
            T.fill(m_q, neg_inf)
            T.fill(z_p, 0)
            T.fill(z_q, 0)
            T.fill(acc_S, 0)
            T.fill(acc_T, 0)

            T.copy(
                Q_mla[bx * block_M : (bx + 1) * block_M, by, :],
                Q_mla_s,
                coalesced_width=coalesced_width_mla,
            )
            T.copy(
                Q_idx[bx * block_M : (bx + 1) * block_M, by, :],
                Q_idx_s,
                coalesced_width=coalesced_width_idx,
            )

            loop_range = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    KV_mla[k * block_N : (k + 1) * block_N, by // kv_group, :],
                    K_mla_s,
                    coalesced_width=coalesced_width_mla,
                )
                T.copy(
                    K_idx[k * block_N : (k + 1) * block_N, by, :],
                    K_idx_s,
                    coalesced_width=coalesced_width_idx,
                )

                T.gemm(
                    Q_mla_s,
                    K_mla_s,
                    S_tile,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.gemm(
                    Q_idx_s,
                    K_idx_s,
                    T_tile,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )

                for i, j in T.Parallel(block_M, block_N):
                    row = bx * block_M + i
                    col = k * block_N + j
                    if row < seq_len and col < seq_len:
                        b_row = TokenIndices[row, 0]
                        b_col = TokenIndices[col, 0]
                        p_row = TokenIndices[row, 1]
                        p_col = TokenIndices[col, 1]
                        if (b_row != b_col) or (p_col > p_row):
                            S_tile[i, j] = neg_inf
                            T_tile[i, j] = neg_inf
                        else:
                            S_tile[i, j] = S_tile[i, j] * scale_mla
                            T_tile[i, j] = T_tile[i, j] * scale_idx
                    else:
                        S_tile[i, j] = neg_inf
                        T_tile[i, j] = neg_inf

                T.copy(m_p, m_p_prev)
                T.reduce_max(S_tile, m_block_p, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    m_p[i] = T.max(m_p_prev[i], m_block_p[i])
                    alpha_p[i] = T.exp(m_p_prev[i] - m_p[i])

                T.copy(m_q, m_q_prev)
                T.reduce_max(T_tile, m_block_q, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    m_q[i] = T.max(m_q_prev[i], m_block_q[i])
                    alpha_q[i] = T.exp(m_q_prev[i] - m_q[i])

                for i, j in T.Parallel(block_M, block_N):
                    Tmp_tile[i, j] = T.exp(T_tile[i, j] - m_q[i])
                T.reduce_sum(Tmp_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    z_q[i] = z_q[i] * alpha_q[i] + sum_buf[i]

                for i in T.Parallel(block_M):
                    z_p[i] = z_p[i] * alpha_p[i]
                    acc_S[i] = acc_S[i] * alpha_p[i]
                    acc_T[i] = acc_T[i] * alpha_p[i]

                for i, j in T.Parallel(block_M, block_N):
                    P_tile[i, j] = T.exp(S_tile[i, j] - m_p[i])

                T.reduce_sum(P_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    z_p[i] = z_p[i] + sum_buf[i]

                for i, j in T.Parallel(block_M, block_N):
                    Tmp_tile[i, j] = S_tile[i, j] * P_tile[i, j]
                T.reduce_sum(Tmp_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    acc_S[i] = acc_S[i] + sum_buf[i]

                for i, j in T.Parallel(block_M, block_N):
                    Tmp_tile[i, j] = T_tile[i, j] * P_tile[i, j]
                T.reduce_sum(Tmp_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    acc_T[i] = acc_T[i] + sum_buf[i]

            for i in T.Parallel(block_M):
                kl[i] = (acc_S[i] / z_p[i]) - m_p[i] - T.log(z_p[i]) - (
                    (acc_T[i] / z_p[i]) - m_q[i] - T.log(z_q[i])
                )

            for i in T.Parallel(block_M):
                row = bx * block_M + i
                if row < seq_len:
                    Output_KL[row, by] = kl[i]

    return main


def dense_warmup_kl_interface(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    q_mla: torch.Tensor,
    kv_mla: torch.Tensor,
    offsets: torch.Tensor,
    kv_group: int = 1,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
) -> torch.Tensor:
    assert q_idx.is_cuda and q_mla.is_cuda, "inputs must be on CUDA"
    assert q_idx.dtype == q_mla.dtype == k_idx.dtype == kv_mla.dtype
    assert q_idx.is_contiguous() and k_idx.is_contiguous()
    assert q_mla.is_contiguous() and kv_mla.is_contiguous()

    seq_len, heads, dim_idx = q_idx.shape
    seq_len_mla, heads_mla, dim_mla = q_mla.shape
    assert seq_len_mla == seq_len
    assert heads_mla == heads
    assert k_idx.shape == (seq_len, heads, dim_idx)
    assert kv_mla.shape == (seq_len, heads // kv_group, dim_mla)

    if offsets.dtype != torch.int32:
        offsets = offsets.to(torch.int32)
    if offsets.device != q_idx.device:
        offsets = offsets.to(q_idx.device)
    token_indices = prepare_token_indices(offsets)

    kernel = dense_warmup_kl_kernel(
        heads,
        dim_mla,
        dim_idx,
        kv_group=kv_group,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
        dtype=q_idx.dtype,
        accum_dtype=T.float32,
    )
    return kernel(q_mla, kv_mla, q_idx, k_idx, offsets, token_indices)


# Backward-compatible alias for callers using the legacy typo.
dense_warmup_kl_kernal = dense_warmup_kl_kernel


def _torch_ref_dense_warmup_kl(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    q_mla: torch.Tensor,
    kv_mla: torch.Tensor,
    offsets: torch.Tensor,
    kv_group: int = 1,
) -> torch.Tensor:
    seq_len, heads, dim_idx = q_idx.shape
    dim_mla = q_mla.shape[-1]
    token_indices = prepare_token_indices(offsets)
    seq_ids = token_indices[:, 0]
    pos_ids = token_indices[:, 1]

    same_seq = seq_ids[:, None] == seq_ids[None, :]
    causal = pos_ids[:, None] >= pos_ids[None, :]
    mask = same_seq & causal

    scale_mla = dim_mla**-0.5
    scale_idx = dim_idx**-0.5

    out = torch.empty((seq_len, heads), device=q_idx.device, dtype=torch.float32)
    mask_f = mask.unsqueeze(0)
    for h in range(heads):
        kv_head = h // kv_group
        s = torch.matmul(q_mla[:, h, :], kv_mla[:, kv_head, :].transpose(0, 1)) * scale_mla
        t = torch.matmul(q_idx[:, h, :], k_idx[:, h, :].transpose(0, 1)) * scale_idx
        s = s.masked_fill(~mask_f[0], float("-inf"))
        t = t.masked_fill(~mask_f[0], float("-inf"))
        p = torch.softmax(s, dim=-1)
        logsum_s = torch.logsumexp(s, dim=-1)
        logsum_t = torch.logsumexp(t, dim=-1)
        out[:, h] = (p * (s - t)).sum(dim=-1) - logsum_s + logsum_t
    return out


def test_dense_warmup_kl() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    offsets = torch.tensor([0, 7, 13], device=device, dtype=torch.int32)
    seq_len = int(offsets[-1].item())
    heads = 4
    kv_group = 2
    dim_mla = 64
    dim_idx = 64

    q_mla = torch.randn(seq_len, heads, dim_mla, device=device, dtype=dtype)
    kv_mla = torch.randn(seq_len, heads // kv_group, dim_mla, device=device, dtype=dtype)
    q_idx = torch.randn(seq_len, heads, dim_idx, device=device, dtype=dtype)
    k_idx = torch.randn(seq_len, heads, dim_idx, device=device, dtype=dtype)

    tl_out = dense_warmup_kl_interface(
        q_idx,
        k_idx,
        q_mla,
        kv_mla,
        offsets,
        kv_group=kv_group,
        block_M=32,
        block_N=32,
        num_stages=2,
        threads=128,
    )
    ref_out = _torch_ref_dense_warmup_kl(q_idx, k_idx, q_mla, kv_mla, offsets, kv_group=kv_group)
    torch.testing.assert_close(tl_out, ref_out, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    test_dense_warmup_kl()
