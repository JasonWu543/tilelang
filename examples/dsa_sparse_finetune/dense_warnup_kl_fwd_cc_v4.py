# ruff: noqa

from typing import Optional

import torch
import tilelang as tl
import tilelang.language as T
from index import prepare_token_indices


PASS_CONFIGS = {
    tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}

LOG2E = 1.44269504


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
    seq_len: int,   # 显式传递 seq_len
    batch_plus_one: int,  # 显式传递 batch_plus_one
    kv_group: int = 1,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 256,
    coalesced_width_mla: Optional[int] = None,
    coalesced_width_idx: Optional[int] = None,
):
    assert heads % kv_group == 0
    if coalesced_width_mla is None:
        coalesced_width_mla = _pick_coalesced_width(dim_mla)
    if coalesced_width_idx is None:
        coalesced_width_idx = _pick_coalesced_width(dim_idx)

    inv_sqrt_mla = dim_mla**-0.5
    inv_sqrt_idx = dim_idx**-0.5
    scale_mla = inv_sqrt_mla * LOG2E
    scale_idx = inv_sqrt_idx * LOG2E

    q_mla_shape = [seq_len, heads, dim_mla]
    kv_mla_shape = [seq_len, heads // kv_group, dim_mla]
    q_idx_shape = [seq_len, heads, dim_idx]
    k_idx_shape = [seq_len, heads, dim_idx]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    dtype = T.bfloat16
    accum_dtype = T.float32

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
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, threads=threads) as (bx, by):
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
            alpha_p = T.alloc_fragment([block_M], accum_dtype)
            z_p = T.alloc_fragment([block_M], accum_dtype)
            acc_entropy = T.alloc_fragment([block_M], accum_dtype)
            acc_cross = T.alloc_fragment([block_M], accum_dtype)

            m_q = T.alloc_fragment([block_M], accum_dtype)
            m_q_prev = T.alloc_fragment([block_M], accum_dtype)
            alpha_q = T.alloc_fragment([block_M], accum_dtype)
            z_q = T.alloc_fragment([block_M], accum_dtype)

            sum_buf = T.alloc_fragment([block_M], accum_dtype)
            kl = T.alloc_fragment([block_M], accum_dtype)

            neg_inf = -T.infinity(accum_dtype)

            T.fill(m_p, neg_inf)
            T.fill(m_q, neg_inf)
            T.fill(z_p, 0)
            T.fill(z_q, 0)
            T.fill(acc_entropy, 0)
            T.fill(acc_cross, 0)

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

                for i, j in T.Parallel(block_M, block_N):
                    row = bx * block_M + i
                    col = k * block_N + j
                    if row < seq_len and col < seq_len:
                        b_row = TokenIndices[row, 0]
                        b_col = TokenIndices[col, 0]
                        p_row = TokenIndices[row, 1]
                        p_col = TokenIndices[col, 1]
                        valid = (b_row == b_col) & (p_col <= p_row)
                        S_tile[i, j] = T.if_then_else(valid, 0, neg_inf)
                        T_tile[i, j] = T.if_then_else(valid, 0, neg_inf)
                    else:
                        S_tile[i, j] = neg_inf
                        T_tile[i, j] = neg_inf

                T.gemm(Q_mla_s, K_mla_s, S_tile, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Q_idx_s, K_idx_s, T_tile, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(m_p, m_p_prev)
                T.fill(m_p, neg_inf)
                T.reduce_max(S_tile, m_p, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    m_p[i] = T.max(m_p[i], m_p_prev[i])
                    alpha_p[i] = T.exp2(m_p_prev[i] * scale_mla - m_p[i] * scale_mla)

                T.copy(m_q, m_q_prev)
                T.fill(m_q, neg_inf)
                T.reduce_max(T_tile, m_q, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    m_q[i] = T.max(m_q[i], m_q_prev[i])
                    alpha_q[i] = T.exp2(m_q_prev[i] * scale_idx - m_q[i] * scale_idx)

                for i, j in T.Parallel(block_M, block_N):
                    Tmp_tile[i, j] = T.exp2(T_tile[i, j] * scale_idx - m_q[i] * scale_idx)
                T.reduce_sum(Tmp_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    z_q[i] = z_q[i] * alpha_q[i] + sum_buf[i]

                for i in T.Parallel(block_M):
                    z_p[i] = z_p[i] * alpha_p[i]
                    acc_entropy[i] = acc_entropy[i] * alpha_p[i]
                    acc_cross[i] = acc_cross[i] * alpha_p[i]

                for i, j in T.Parallel(block_M, block_N):
                    P_tile[i, j] = T.exp2(S_tile[i, j] * scale_mla - m_p[i] * scale_mla)
                T.reduce_sum(P_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    z_p[i] = z_p[i] + sum_buf[i]

                for i, j in T.Parallel(block_M, block_N):
                    Tmp_tile[i, j] = P_tile[i, j] * (S_tile[i, j] * inv_sqrt_mla)
                T.reduce_sum(Tmp_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    acc_entropy[i] = acc_entropy[i] + sum_buf[i]

                for i, j in T.Parallel(block_M, block_N):
                    Tmp_tile[i, j] = P_tile[i, j] * (T_tile[i, j] * inv_sqrt_idx)
                T.reduce_sum(Tmp_tile, sum_buf, dim=1)
                for i in T.Parallel(block_M):
                    acc_cross[i] = acc_cross[i] + sum_buf[i]

            for i in T.Parallel(block_M):
                logsum_p = m_p[i] * inv_sqrt_mla + T.log(z_p[i])
                logsum_q = m_q[i] * inv_sqrt_idx + T.log(z_q[i])
                kl[i] = acc_entropy[i] / z_p[i] - acc_cross[i] / z_p[i] - logsum_p + logsum_q

            for i in T.Parallel(block_M):
                row = bx * block_M + i
                if row < seq_len:
                    Output_KL[row, by] = kl[i]

    return main


def dense_kl_div_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim_mla: int,
    dim_idx: int,
    kv_group_mla: int,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    return dense_warmup_kl_kernel(
        heads,
        dim_mla,
        dim_idx,
        kv_group=kv_group_mla,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
        seq_len=seq_len,
        batch_plus_one=batch + 1,
    )


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
    threads: int = 256,
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
        seq_len=seq_len,
        batch_plus_one=offsets.shape[0],
    )
    return kernel(q_mla, kv_mla, q_idx, k_idx, offsets, token_indices)


def torch_ref_dense_kl(
    Q_indexer, K_indexer, Q_mla, KV_mla, offsets, kv_group=1,
    sm_scale_indexer=None, sm_scale_mla=None
):
    """PyTorch reference implementation."""
    S, H, Didx = Q_indexer.shape
    _, G, Dmla = KV_mla.shape
    assert G == H // kv_group
    dim_v = 512
    tail_dim = Dmla - dim_v
    if sm_scale_indexer is None:
        sm_scale_indexer = Didx ** -0.5
    if sm_scale_mla is None:
        sm_scale_mla = Dmla ** -0.5

    # Build causal mask
    token_indices = prepare_token_indices(offsets)
    b = token_indices[:, 0]
    s = token_indices[:, 1]
    bos = offsets[b]
    eos = offsets[b + 1]
    maxk = bos + s

    k = torch.arange(S, device=Q_indexer.device)
    mask = (k[None, :] >= bos[:, None]) & (k[None, :] < eos[:, None]) & (k[None, :] <= maxk[:, None])

    # Teacher logits (indexer): [S, H, S]
    t = torch.einsum("shd,Shd->shS", Q_indexer.float(), K_indexer.float()) * sm_scale_indexer
    t = t.masked_fill(~mask[:, None, :], float("-inf"))

    # Student logits (MLA): [S, H, S]
    K_mla = KV_mla.float()  # [S, G, Dmla]
    K_mla_expanded = K_mla.repeat_interleave(kv_group, dim=1)  # [S, H, Dmla]
    
    s_main = torch.einsum("shd,Shd->shS", Q_mla[..., :dim_v].float(), K_mla_expanded[..., :dim_v].float()) * sm_scale_mla
    if tail_dim > 0:
        s_tail = torch.einsum("shd,Shd->shS", Q_mla[..., dim_v:].float(), K_mla_expanded[..., dim_v:].float()) * sm_scale_mla
        s_logits = s_main + s_tail
    else:
        s_logits = s_main
    
    s_logits = s_logits.masked_fill(~mask[:, None, :], float("-inf"))

    # KL divergence: KL(teacher || student)
    # MLA 是老师 (Teacher)
    p_teacher = torch.softmax(s_logits, dim=-1)  # MLA 输出的注意力分布
    log_p_teacher = torch.log_softmax(s_logits, dim=-1)  # MLA 对数概率

    # Indexer 是学生 (Student)
    q_student_logits = t  # Indexer 的输出
    log_q_student = torch.log_softmax(q_student_logits, dim=-1)  # Indexer 对数概率

    # 2. 计算 KL 散度：KL(Teacher || Student)
    kl = (p_teacher * (log_p_teacher - log_q_student)).sum(dim=-1)
    # pt = torch.softmax(t, dim=-1)  # [S, H, S]
    # log_pt = torch.log_softmax(t, dim=-1)
    # log_ps = torch.log_softmax(s_logits, dim=-1)
    
    # kl = (pt * (log_pt - log_ps)).sum(dim=-1)  # [S, H]
    
    return kl


def test_dense_warmup_kl() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

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
        threads=256,
    )

    ref_out = torch_ref_dense_kl(q_idx, k_idx, q_mla, kv_mla, offsets, kv_group=kv_group)
    torch.testing.assert_close(tl_out, ref_out, rtol=2e-2, atol=2e-2)


dense_warmup_kl_kernal = dense_warmup_kl_kernel


if __name__ == "__main__":
    test_dense_warmup_kl()
