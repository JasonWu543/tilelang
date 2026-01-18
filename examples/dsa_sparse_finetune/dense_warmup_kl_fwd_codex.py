# ruff: noqa
import torch
import tilelang
from tilelang import language as T

from index import prepare_token_indices


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dense_warmup_kl_fwd(
    heads: int,
    dim_indexer: int,
    dim_mla: int,       # = dim + tail_dim
    dim_v: int,         # 利用这个计算tail_dim
    kv_group: int = 1,
    sm_scale_indexer: float | None = None,
    sm_scale_mla: float | None = None,
    block_M: int = 64,  # query block size
    block_N: int = 64,  # key block size
    num_stages: int = 2,
    threads: int = 256,
):
    assert dim_indexer == tilelang.math.next_power_of_2(dim_indexer)
    assert dim_mla == tilelang.math.next_power_of_2(dim_mla)
    assert dim_v == tilelang.math.next_power_of_2(dim_v)
    tail_dim = dim_mla - dim_v
    assert tail_dim >= 0

    if sm_scale_indexer is None:
        sm_scale_indexer = (1.0 / dim_indexer) ** 0.5
    if sm_scale_mla is None:
        sm_scale_mla = (1.0 / dim_mla) ** 0.5

    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    # shapes
    Q_idx_shape = [seq_len, heads, dim_indexer]
    K_idx_shape = [seq_len, heads, dim_indexer]

    Q_mla_shape = [seq_len, heads, dim_mla]
    KV_mla_shape = [seq_len, kv_group, dim_mla]

    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    Loss_shape = [seq_len, heads]

    dtype = T.bfloat16
    acc = T.float32
    i32 = T.int32

    head_kv = heads // kv_group

    BM = block_M
    BN = block_N
    NK = tilelang.cdiv(1, 1)  # placeholder; 我们用 for 循环 + runtime mask（见下）

    @T.prim_func
    def main(
        Q_indexer: T.Tensor(Q_idx_shape, dtype),   # [S,H,Didx]
        K_indexer: T.Tensor(K_idx_shape, dtype),   # [S,H,Didx]
        Q_mla: T.Tensor(Q_mla_shape, dtype),       # [S,H,Dmla]
        KV_mla: T.Tensor(KV_mla_shape, dtype),     # [S,G,Dmla]
        Offsets: T.Tensor(offsets_shape, i32),     # [B+1]
        TokenIndices: T.Tensor(token_indices_shape, i32),  # [S,2] -> (b_i, s_i)
        LossPerTokenHead: T.Tensor(Loss_shape, acc),       # [S,H]
    ):
        # grid: (q_block, head)
        with T.Kernel(tilelang.cdiv(seq_len, BM), heads, threads=threads) as (bx, by):
            h = by
            q0 = bx * BM

            # ---- shared buffers ----
            Qidx_s = T.alloc_shared([BM, dim_indexer], dtype)
            Kidx_s = T.alloc_shared([BN, dim_indexer], dtype)

            Qmla_s = T.alloc_shared([BM, dim_v], dtype)
            Qmla_tail_s = T.alloc_shared([BM, tail_dim], dtype)

            Kmla_s = T.alloc_shared([BN, dim_v], dtype)
            Kmla_tail_s = T.alloc_shared([BN, tail_dim], dtype)

            # ---- per-row meta (fragment) ----
            bos = T.alloc_fragment([BM], i32)
            eos = T.alloc_fragment([BM], i32)
            max_k_global = T.alloc_fragment([BM], i32)
            valid_q = T.alloc_fragment([BM], "bool")

            # ---- streaming accumulators (teacher & student) ----
            mt = T.alloc_fragment([BM], acc)
            lt = T.alloc_fragment([BM], acc)
            att = T.alloc_fragment([BM], acc)  # sum exp(t-mt)*t
            ats = T.alloc_fragment([BM], acc)  # sum exp(t-mt)*s

            ms = T.alloc_fragment([BM], acc)
            ls = T.alloc_fragment([BM], acc)

            alpha_t = T.alloc_fragment([BM], acc)
            alpha_s = T.alloc_fragment([BM], acc)

            # ---- score tiles ----
            t_tile = T.alloc_fragment([BM, BN], acc)
            s_tile = T.alloc_fragment([BM, BN], acc)

            # init
            for mi in T.Parallel(BM):
                qg = q0 + mi
                valid_q[mi] = qg < seq_len
                mt[mi] = -(2**30)
                lt[mi] = 0.0
                att[mi] = 0.0
                ats[mi] = 0.0
                ms[mi] = -(2**30)
                ls[mi] = 0.0

                # default
                bos[mi] = 0
                eos[mi] = 0
                max_k_global[mi] = -1

                if valid_q[mi]:
                    b_i = TokenIndices[qg, 0]
                    s_i = TokenIndices[qg, 1]
                    bos_i = Offsets[b_i]
                    eos_i = Offsets[b_i + 1]
                    bos[mi] = bos_i
                    eos[mi] = eos_i
                    max_k_global[mi] = bos_i + s_i  # causal: key_global <= bos+s_i

            # load Q blocks to shared
            # indexer Q
            for mi, di in T.Parallel(BM, dim_indexer):
                qg = q0 + mi
                Qidx_s[mi, di] = T.if_then_else(valid_q[mi], Q_indexer[qg, h, di], T.cast(0, dtype))

            # mla Q (split main/tail)
            for mi, di in T.Parallel(BM, dim_v):
                qg = q0 + mi
                Qmla_s[mi, di] = T.if_then_else(valid_q[mi], Q_mla[qg, h, di], T.cast(0, dtype))
            for mi, di in T.Parallel(BM, tail_dim):
                qg = q0 + mi
                Qmla_tail_s[mi, di] = T.if_then_else(valid_q[mi], Q_mla[qg, h, dim_v + di], T.cast(0, dtype))

            # iterate over key blocks globally: k0 in [0, seq_len)
            # 这里不用 T.Pipelined（避免对动态 seq_len 的限制），用普通 for + mask
            for kb in range(0, 1):  # placeholder; 下面用 while 动态
                pass

            k0 = 0
            while k0 < seq_len:
                # load K blocks to shared (with global bounds check)
                for nj, di in T.Parallel(BN, dim_indexer):
                    kg = k0 + nj
                    Kidx_s[nj, di] = T.if_then_else(kg < seq_len, K_indexer[kg, h, di], T.cast(0, dtype))

                # mla K/V are grouped by kv_group
                g = h // head_kv
                for nj, di in T.Parallel(BN, dim_v):
                    kg = k0 + nj
                    Kmla_s[nj, di] = T.if_then_else(kg < seq_len, KV_mla[kg, g, di], T.cast(0, dtype))
                for nj, di in T.Parallel(BN, tail_dim):
                    kg = k0 + nj
                    Kmla_tail_s[nj, di] = T.if_then_else(kg < seq_len, KV_mla[kg, g, dim_v + di], T.cast(0, dtype))

                # compute logits tiles
                # t_tile = Qidx_s @ Kidx_s^T
                for mi, nj in T.Parallel(BM, BN):
                    t_tile[mi, nj] = 0.0
                    s_tile[mi, nj] = 0.0

                T.gemm(Qidx_s, Kidx_s, t_tile, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # s_tile = Qmla_main @ Kmla_main^T + Qmla_tail @ Kmla_tail^T
                T.gemm(Qmla_s, Kmla_s, s_tile, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if tail_dim > 0:
                    T.gemm(Qmla_tail_s, Kmla_tail_s, s_tile, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # apply scaling + causal/packed mask: invalid -> -inf
                for mi, nj in T.Parallel(BM, BN):
                    kg = k0 + nj
                    # valid key for this row?
                    in_range = (kg < seq_len) & valid_q[mi]
                    same_seq = in_range & (kg >= bos[mi]) & (kg < eos[mi])
                    causal_ok = same_seq & (kg <= max_k_global[mi])
                    t = t_tile[mi, nj] * sm_scale_indexer
                    s = s_tile[mi, nj] * sm_scale_mla
                    t_tile[mi, nj] = T.if_then_else(causal_ok, t, -T.infinity(acc))
                    s_tile[mi, nj] = T.if_then_else(causal_ok, s, -T.infinity(acc))

                # ---- update teacher streaming (mt, lt, att, ats) ----
                # mt = max(mt, rowmax(t_tile))
                mt_prev = T.alloc_fragment([BM], acc)
                for mi in T.Parallel(BM):
                    mt_prev[mi] = mt[mi]

                T.reduce_max(t_tile, mt, dim=1, clear=False)  # mt now is max(old, blockmax)

                for mi in T.Parallel(BM):
                    alpha_t[mi] = T.exp(mt_prev[mi] - mt[mi])
                    lt[mi] = lt[mi] * alpha_t[mi]
                    att[mi] = att[mi] * alpha_t[mi]
                    ats[mi] = ats[mi] * alpha_t[mi]

                # w = exp(t_tile - mt[:,None]); accumulate
                sum_w = T.alloc_fragment([BM], acc)
                sum_wt = T.alloc_fragment([BM], acc)
                sum_ws = T.alloc_fragment([BM], acc)
                for mi in T.Parallel(BM):
                    sum_w[mi] = 0.0
                    sum_wt[mi] = 0.0
                    sum_ws[mi] = 0.0

                for mi, nj in T.Parallel(BM, BN):
                    w = T.exp(t_tile[mi, nj] - mt[mi])
                    sum_w[mi] += w
                    sum_wt[mi] += w * t_tile[mi, nj]
                    sum_ws[mi] += w * s_tile[mi, nj]

                for mi in T.Parallel(BM):
                    lt[mi] += sum_w[mi]
                    att[mi] += sum_wt[mi]
                    ats[mi] += sum_ws[mi]

                # ---- update student streaming (ms, ls) only for lse_s ----
                ms_prev = T.alloc_fragment([BM], acc)
                for mi in T.Parallel(BM):
                    ms_prev[mi] = ms[mi]

                T.reduce_max(s_tile, ms, dim=1, clear=False)

                for mi in T.Parallel(BM):
                    alpha_s[mi] = T.exp(ms_prev[mi] - ms[mi])
                    ls[mi] = ls[mi] * alpha_s[mi]

                sum_ws_only = T.alloc_fragment([BM], acc)
                for mi in T.Parallel(BM):
                    sum_ws_only[mi] = 0.0

                for mi, nj in T.Parallel(BM, BN):
                    w2 = T.exp(s_tile[mi, nj] - ms[mi])
                    sum_ws_only[mi] += w2

                for mi in T.Parallel(BM):
                    ls[mi] += sum_ws_only[mi]

                k0 += BN

            # finalize KL per row
            for mi in T.Parallel(BM):
                qg = q0 + mi
                if valid_q[mi]:
                    lse_t = T.log(lt[mi]) + mt[mi]
                    lse_s = T.log(ls[mi]) + ms[mi]
                    Et_t = att[mi] / lt[mi]
                    Et_s = ats[mi] / lt[mi]
                    kl = (Et_t - lse_t) - (Et_s - lse_s)
                    LossPerTokenHead[qg, h] = kl
                else:
                    # out of range rows
                    pass

    return main


def dense_warmup_kl_fwd_interface(
    Q_indexer: torch.Tensor,   # [S,H,Didx] bf16
    K_indexer: torch.Tensor,   # [S,H,Didx] bf16
    Q_mla: torch.Tensor,       # [S,H,Dmla] bf16
    KV_mla: torch.Tensor,      # [S,G,Dmla] bf16
    offsets: torch.Tensor,     # [B+1] int32
    kv_group: int = 1,
    sm_scale_indexer: float | None = None,
    sm_scale_mla: float | None = None,
    block_M: int = 64,
    block_N: int = 64,
    threads: int = 256,
):
    assert Q_indexer.is_cuda and Q_indexer.dtype == torch.bfloat16
    assert K_indexer.is_cuda and K_indexer.dtype == torch.bfloat16
    assert Q_mla.is_cuda and Q_mla.dtype == torch.bfloat16
    assert KV_mla.is_cuda and KV_mla.dtype == torch.bfloat16
    assert offsets.dtype == torch.int32 and offsets.is_cuda

    S, H, Didx = Q_indexer.shape
    S2, H2, Didx2 = K_indexer.shape
    assert S == S2 and H == H2 and Didx == Didx2

    S3, H3, Dmla = Q_mla.shape
    assert S3 == S and H3 == H

    S4, G, Dmla2 = KV_mla.shape
    assert S4 == S and Dmla2 == Dmla
    assert G == kv_group

    token_indices = prepare_token_indices(offsets)

    kernel = dense_warmup_kl_fwd(
        heads=H,
        dim_indexer=Didx,
        dim_mla=Dmla,
        dim_v=512,  # 你这里 MLA 的 V dim 固定 512；若不同自己传参改掉
        kv_group=kv_group,
        sm_scale_indexer=sm_scale_indexer,
        sm_scale_mla=sm_scale_mla,
        block_M=block_M,
        block_N=block_N,
        threads=threads,
    )
    (loss_per_token_head,) = kernel(Q_indexer, K_indexer, Q_mla, KV_mla, offsets, token_indices)
    # scalar loss（你可以选择 mean/sum）
    return loss_per_token_head

def torch_ref_dense_kl(
    Q_indexer, K_indexer, Q_mla, KV_mla, offsets, kv_group=1,
    sm_scale_indexer=None, sm_scale_mla=None
):
    S, H, Didx = Q_indexer.shape
    _, G, Dmla = KV_mla.shape
    head_kv = H // kv_group
    dim_v = 512
    tail_dim = Dmla - dim_v

    if sm_scale_indexer is None:
        sm_scale_indexer = Didx ** -0.5
    if sm_scale_mla is None:
        sm_scale_mla = Dmla ** -0.5

    # build per-token (bos,eos,max_k_global) from offsets
    token_indices = prepare_token_indices(offsets)
    b = token_indices[:, 0]
    s = token_indices[:, 1]
    bos = offsets[b]
    eos = offsets[b + 1]
    maxk = bos + s

    # causal + same-seq mask: [S,S]
    k = torch.arange(S, device=Q_indexer.device)
    mask = (k[None, :] >= bos[:, None]) & (k[None, :] < eos[:, None]) & (k[None, :] <= maxk[:, None])

    # teacher logits: [S,H,S]
    t = torch.einsum("shd,Skd->shS", Q_indexer.float(), K_indexer.float()).mul(sm_scale_indexer)
    t = t.masked_fill(~mask[:, None, :], float("-inf"))

    # student logits: [S,H,S]
    g = (torch.arange(H, device=Q_indexer.device) // head_kv)  # [H]
    K_mla = KV_mla.float()  # [S,G,Dmla]
    s_main = torch.einsum("shd,Sgd->shS", Q_mla[..., :dim_v].float(), K_mla[..., :dim_v]).mul(sm_scale_mla)
    if tail_dim > 0:
        s_tail = torch.einsum("shd,Sgd->shS", Q_mla[..., dim_v:].float(), K_mla[..., dim_v:]).mul(sm_scale_mla)
        s_logits = s_main + s_tail
    else:
        s_logits = s_main
    # 让 K 的 group 对齐 head：用 gather
    s_logits = s_logits.view(S, H, S)  # 这里只在 kv_group=1 最简单；kv_group>1 你需要按 head 的 group 选择 KV
    s_logits = s_logits.masked_fill(~mask[:, None, :], float("-inf"))

    pt = torch.softmax(t, dim=-1)
    kl = (pt * (torch.log_softmax(t, dim=-1) - torch.log_softmax(s_logits, dim=-1))).sum(dim=-1)  # [S,H]
    return kl

from tilelang.profiler import do_bench

def test_dense_warmup_kl():
    torch.manual_seed(0)
    S = 1024
    H = 128
    Didx = 512
    Dmla = 576
    G = 1

    Qidx = torch.randn(S, H, Didx, device="cuda", dtype=torch.bfloat16)
    Kidx = torch.randn(S, H, Didx, device="cuda", dtype=torch.bfloat16)
    Qmla = torch.randn(S, H, Dmla, device="cuda", dtype=torch.bfloat16)
    KV  = torch.randn(S, G, Dmla, device="cuda", dtype=torch.bfloat16)

    offsets = torch.tensor([0, S], device="cuda", dtype=torch.int32)

    tl_loss = dense_warmup_kl_fwd_interface(Qidx, Kidx, Qmla, KV, offsets, kv_group=G, block_M=64, block_N=64, threads=256)
    ref = torch_ref_dense_kl(Qidx, Kidx, Qmla, KV, offsets, kv_group=G)

    max_err = (tl_loss - ref).abs().max().item()
    mean_err = (tl_loss - ref).abs().mean().item()
    print("max_err:", max_err, "mean_err:", mean_err)

    def fn():
        out = dense_warmup_kl_fwd_interface(Qidx, Kidx, Qmla, KV, offsets, kv_group=G, block_M=64, block_N=64, threads=256)
        return out

    ms = do_bench(fn, warmup=50, rep=100)
    print(f"tilelang kernel: {ms:.3f} ms")

if __name__ == "__main__":
    test_dense_warmup_kl()