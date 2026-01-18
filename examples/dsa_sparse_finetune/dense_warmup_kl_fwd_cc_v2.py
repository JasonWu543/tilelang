# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from index import prepare_token_indices
from utils import assert_tensors_similar


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def dense_warmup_kl_kernel(
    heads,
    dim_indexer,
    dim_mla,
    tail_dim,
    kv_group=1,
    sm_scale_indexer=None,
    sm_scale_mla=None,
    block_K=64,
    num_stages=2,
    threads=128,
):
    """
    Dense warmup KL divergence kernel.
    Computes KL(indexer || mla) in blocks to avoid O(N^2) memory.
    """
    assert dim_indexer == tilelang.math.next_power_of_2(dim_indexer)
    assert dim_mla == tilelang.math.next_power_of_2(dim_mla)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    
    if sm_scale_indexer is None:
        sm_scale_indexer = dim_indexer ** -0.5
    if sm_scale_mla is None:
        sm_scale_mla = (dim_mla + tail_dim) ** -0.5

    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    head_kv = heads // kv_group
    
    # Input shapes
    q_indexer_shape = [seq_len, heads, dim_indexer]
    k_indexer_shape = [seq_len, heads, dim_indexer]
    q_mla_shape = [seq_len, heads, dim_mla + tail_dim]
    kv_mla_shape = [seq_len, head_kv, dim_mla + tail_dim]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    kl_shape = [seq_len, heads]
    
    dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    H = heads
    G = head_kv
    D_idx = dim_indexer
    D_mla = dim_mla
    D_tail = tail_dim
    BK = block_K

    @T.prim_func
    def main(
        Q_indexer: T.Tensor(q_indexer_shape, dtype),
        K_indexer: T.Tensor(k_indexer_shape, dtype),
        Q_mla: T.Tensor(q_mla_shape, dtype),
        KV_mla: T.Tensor(kv_mla_shape, dtype),
        Offsets: T.Tensor(offsets_shape, indices_dtype),
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
        KL_loss: T.Tensor(kl_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, heads, threads=threads) as (bx, by):
            # Shared memory for keys
            K_idx_shared = T.alloc_shared([BK, D_idx], dtype)
            K_mla_shared = T.alloc_shared([BK, D_mla], dtype)
            K_tail_shared = T.alloc_shared([BK, D_tail], dtype)
            
            # Attention score fragments for current block
            score_idx = T.alloc_fragment([BK], accum_dtype)
            score_mla = T.alloc_fragment([BK], accum_dtype)
            
            # Softmax accumulators
            max_score_idx = T.alloc_fragment([1], accum_dtype)
            max_score_mla = T.alloc_fragment([1], accum_dtype)
            sum_exp_idx = T.alloc_fragment([1], accum_dtype)
            sum_exp_mla = T.alloc_fragment([1], accum_dtype)
            kl_acc = T.alloc_fragment([1], accum_dtype)
            max_score_idx_ro = T.alloc_fragment([1], accum_dtype)
            max_score_mla_ro = T.alloc_fragment([1], accum_dtype)
            sum_exp_idx_ro = T.alloc_fragment([1], accum_dtype)
            sum_exp_mla_ro = T.alloc_fragment([1], accum_dtype)
            
            # Initialize accumulators
            T.fill(max_score_idx, -T.infinity(accum_dtype))
            T.fill(max_score_mla, -T.infinity(accum_dtype))
            T.fill(sum_exp_idx, 0)
            T.fill(sum_exp_mla, 0)
            T.fill(kl_acc, 0)
            
            # Get sequence boundaries
            s_i = bx
            h_i = by
            b_i, q_pos = TokenIndices[s_i, 0], TokenIndices[s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]
            max_kv_i = q_pos  # causal: can only attend to positions <= q_pos
            
            # Determine KV group for this head
            g_i = h_i // kv_group
            
            seq_len_cur = eos - bos
            num_blocks = T.ceildiv(seq_len_cur, BK)
            
            # ====== Pass 1: Compute max scores for numerical stability ======
            q_idx_p1 = T.alloc_fragment([D_idx], dtype)
            q_mla_p1 = T.alloc_fragment([D_mla], dtype)
            q_tail_p1 = T.alloc_fragment([D_tail], dtype)
            for d in T.Parallel(D_idx):
                q_idx_p1[d] = Q_indexer[bos + s_i, h_i, d]
            for d in T.Parallel(D_mla):
                q_mla_p1[d] = Q_mla[bos + s_i, h_i, d]
            for d in T.Parallel(D_tail):
                q_tail_p1[d] = Q_mla[bos + s_i, h_i, D_mla + d]
            for block_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_start = block_i * BK
                k_end = T.min(k_start + BK, seq_len_cur)
                
                # Load K blocks - avoid nested T.Parallel
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_idx):
                        if k_idx < k_end:
                            K_idx_shared[bk_i, d_i] = K_indexer[bos + k_idx, h_i, d_i]
                        else:
                            K_idx_shared[bk_i, d_i] = 0
                
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_mla):
                        if k_idx < k_end:
                            K_mla_shared[bk_i, d_i] = KV_mla[bos + k_idx, g_i, d_i]
                        else:
                            K_mla_shared[bk_i, d_i] = 0
                
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_tail):
                        if k_idx < k_end:
                            K_tail_shared[bk_i, d_i] = KV_mla[bos + k_idx, g_i, D_mla + d_i]
                        else:
                            K_tail_shared[bk_i, d_i] = 0
                
                # Compute attention scores
                for bk_i in T.Parallel(BK):
                    score_idx[bk_i] = 0
                    score_mla[bk_i] = 0
                
                # Indexer attention: Q @ K^T
                for bk_i in T.Parallel(BK):
                    for d in T.Serial(D_idx):
                        score_idx[bk_i] += q_idx_p1[d] * K_idx_shared[bk_i, d]
                
                # MLA attention: Q @ K^T (main + tail)
                for bk_i in T.Parallel(BK):
                    for d in T.Serial(D_mla):
                        score_mla[bk_i] += q_mla_p1[d] * K_mla_shared[bk_i, d]
                    for d in T.Serial(D_tail):
                        score_mla[bk_i] += q_tail_p1[d] * K_tail_shared[bk_i, d]
                
                # Apply causal mask and find max
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    is_valid = (k_idx < k_end) & (k_idx <= max_kv_i)
                    
                    if is_valid:
                        score_idx[bk_i] = score_idx[bk_i] * sm_scale_indexer
                        score_mla[bk_i] = score_mla[bk_i] * sm_scale_mla
                        max_score_idx[0] = T.max(max_score_idx[0], score_idx[bk_i])
                        max_score_mla[0] = T.max(max_score_mla[0], score_mla[bk_i])
                    else:
                        score_idx[bk_i] = -T.infinity(accum_dtype)
                        score_mla[bk_i] = -T.infinity(accum_dtype)

            max_score_idx_ro[0] = max_score_idx[0]
            max_score_mla_ro[0] = max_score_mla[0]
            
            # ====== Pass 2: Compute softmax denominators ======
            q_idx_p2 = T.alloc_fragment([D_idx], dtype)
            q_mla_p2 = T.alloc_fragment([D_mla], dtype)
            q_tail_p2 = T.alloc_fragment([D_tail], dtype)
            for d in T.Parallel(D_idx):
                q_idx_p2[d] = Q_indexer[bos + s_i, h_i, d]
            for d in T.Parallel(D_mla):
                q_mla_p2[d] = Q_mla[bos + s_i, h_i, d]
            for d in T.Parallel(D_tail):
                q_tail_p2[d] = Q_mla[bos + s_i, h_i, D_mla + d]
            for block_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_start = block_i * BK
                k_end = T.min(k_start + BK, seq_len_cur)
                
                # Reload K blocks
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_idx):
                        if k_idx < k_end:
                            K_idx_shared[bk_i, d_i] = K_indexer[bos + k_idx, h_i, d_i]
                        else:
                            K_idx_shared[bk_i, d_i] = 0
                
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_mla):
                        if k_idx < k_end:
                            K_mla_shared[bk_i, d_i] = KV_mla[bos + k_idx, g_i, d_i]
                        else:
                            K_mla_shared[bk_i, d_i] = 0
                
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_tail):
                        if k_idx < k_end:
                            K_tail_shared[bk_i, d_i] = KV_mla[bos + k_idx, g_i, D_mla + d_i]
                        else:
                            K_tail_shared[bk_i, d_i] = 0
                
                # Recompute scores
                for bk_i in T.Parallel(BK):
                    score_idx[bk_i] = 0
                    score_mla[bk_i] = 0
                
                for bk_i in T.Parallel(BK):
                    for d in T.Serial(D_idx):
                        score_idx[bk_i] += q_idx_p2[d] * K_idx_shared[bk_i, d]
                
                for bk_i in T.Parallel(BK):
                    for d in T.Serial(D_mla):
                        score_mla[bk_i] += q_mla_p2[d] * K_mla_shared[bk_i, d]
                    for d in T.Serial(D_tail):
                        score_mla[bk_i] += q_tail_p2[d] * K_tail_shared[bk_i, d]
                
                # Apply scaling, mask, and accumulate exp
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    is_valid = (k_idx < k_end) & (k_idx <= max_kv_i)
                    
                    if is_valid:
                        score_idx[bk_i] = score_idx[bk_i] * sm_scale_indexer
                        score_mla[bk_i] = score_mla[bk_i] * sm_scale_mla
                        sum_exp_idx[0] += T.exp(score_idx[bk_i] - max_score_idx_ro[0])
                        sum_exp_mla[0] += T.exp(score_mla[bk_i] - max_score_mla_ro[0])

            sum_exp_idx_ro[0] = sum_exp_idx[0]
            sum_exp_mla_ro[0] = sum_exp_mla[0]
            
            # ====== Pass 3: Compute KL divergence ======
            q_idx_p3 = T.alloc_fragment([D_idx], dtype)
            q_mla_p3 = T.alloc_fragment([D_mla], dtype)
            q_tail_p3 = T.alloc_fragment([D_tail], dtype)
            for d in T.Parallel(D_idx):
                q_idx_p3[d] = Q_indexer[bos + s_i, h_i, d]
            for d in T.Parallel(D_mla):
                q_mla_p3[d] = Q_mla[bos + s_i, h_i, d]
            for d in T.Parallel(D_tail):
                q_tail_p3[d] = Q_mla[bos + s_i, h_i, D_mla + d]
            for block_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_start = block_i * BK
                k_end = T.min(k_start + BK, seq_len_cur)
                
                # Reload K blocks again
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_idx):
                        if k_idx < k_end:
                            K_idx_shared[bk_i, d_i] = K_indexer[bos + k_idx, h_i, d_i]
                        else:
                            K_idx_shared[bk_i, d_i] = 0
                
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_mla):
                        if k_idx < k_end:
                            K_mla_shared[bk_i, d_i] = KV_mla[bos + k_idx, g_i, d_i]
                        else:
                            K_mla_shared[bk_i, d_i] = 0
                
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    for d_i in T.Serial(D_tail):
                        if k_idx < k_end:
                            K_tail_shared[bk_i, d_i] = KV_mla[bos + k_idx, g_i, D_mla + d_i]
                        else:
                            K_tail_shared[bk_i, d_i] = 0
                
                # Recompute scores one more time
                for bk_i in T.Parallel(BK):
                    score_idx[bk_i] = 0
                    score_mla[bk_i] = 0
                
                for bk_i in T.Parallel(BK):
                    for d in T.Serial(D_idx):
                        score_idx[bk_i] += q_idx_p3[d] * K_idx_shared[bk_i, d]
                
                for bk_i in T.Parallel(BK):
                    for d in T.Serial(D_mla):
                        score_mla[bk_i] += q_mla_p3[d] * K_mla_shared[bk_i, d]
                    for d in T.Serial(D_tail):
                        score_mla[bk_i] += q_tail_p3[d] * K_tail_shared[bk_i, d]
                
                # Compute KL: p * (log_p - log_q)
                for bk_i in T.Parallel(BK):
                    k_idx = k_start + bk_i
                    is_valid = (k_idx < k_end) & (k_idx <= max_kv_i)
                    
                    if is_valid:
                        score_idx[bk_i] = score_idx[bk_i] * sm_scale_indexer
                        score_mla[bk_i] = score_mla[bk_i] * sm_scale_mla
                        
                        # p = exp(score - max) / sum_exp
                        p_idx = T.exp(score_idx[bk_i] - max_score_idx_ro[0]) / sum_exp_idx_ro[0]
                        
                        # log_p = score - max - log(sum_exp)
                        log_p_idx = score_idx[bk_i] - max_score_idx_ro[0] - T.log(sum_exp_idx_ro[0])
                        log_p_mla = score_mla[bk_i] - max_score_mla_ro[0] - T.log(sum_exp_mla_ro[0])
                        
                        # KL = p * (log_p - log_q)
                        kl_acc[0] += p_idx * (log_p_idx - log_p_mla)
            
            # Write result
            KL_loss[bos + s_i, h_i] = kl_acc[0]

    return main


def dense_warmup_kl_interface(
    q_indexer, k_indexer, q_mla, kv_mla, offsets,
    kv_group=1, sm_scale_indexer=None, sm_scale_mla=None,
    block_K=64, num_stages=2, threads=128
):
    """Interface function for dense warmup KL kernel."""
    assert q_indexer.is_contiguous() and k_indexer.is_contiguous()
    assert q_mla.is_contiguous() and kv_mla.is_contiguous()
    
    seq_len, heads, dim_indexer = q_indexer.shape
    _, _, dim_mla_total = q_mla.shape
    _, head_kv, _ = kv_mla.shape
    
    assert heads % kv_group == 0
    assert head_kv == heads // kv_group
    
    dim_mla = 512  # Fixed as per your config
    tail_dim = dim_mla_total - dim_mla
    
    token_indices = prepare_token_indices(offsets)
    
    kl_loss = torch.zeros((seq_len, heads), dtype=torch.float32, device=q_indexer.device)
    
    kernel = dense_warmup_kl_kernel(
        heads, dim_indexer, dim_mla, tail_dim, kv_group,
        sm_scale_indexer, sm_scale_mla,
        block_K, num_stages, threads
    )
    
    kernel(q_indexer, k_indexer, q_mla, kv_mla, offsets, token_indices, kl_loss)
    
    return kl_loss


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


def test_dense_warmup_kl(
    B=2,
    S=1024,
    H=32,
    HKV=1,
    D_indexer=128,
    D_mla=576,
    dtype=torch.bfloat16,
    block_K=64,
    num_stages=2,
    threads=128,
):
    torch.manual_seed(0)
    
    kv_group = H // HKV
    
    q_indexer = torch.randn((S, H, D_indexer), dtype=dtype, device="cuda")
    k_indexer = torch.randn((S, H, D_indexer), dtype=dtype, device="cuda")
    q_mla = torch.randn((S, H, D_mla), dtype=dtype, device="cuda")
    kv_mla = torch.randn((S, HKV, D_mla), dtype=dtype, device="cuda")
    
    # Create offsets for B sequences
    offsets = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    seq_len_per_batch = S // B
    for i in range(B):
        offsets[i + 1] = offsets[i] + seq_len_per_batch
    offsets[-1] = S  # Handle remainder
    
    print(f"Testing with B={B}, S={S}, H={H}, HKV={HKV}, D_indexer={D_indexer}, D_mla={D_mla}")
    print(f"Offsets: {offsets.tolist()}")
    
    # TileLang kernel
    tl_kl = dense_warmup_kl_interface(
        q_indexer, k_indexer, q_mla, kv_mla, offsets,
        kv_group=kv_group, block_K=block_K, num_stages=num_stages, threads=threads
    )
    
    # PyTorch reference
    ref_kl = torch_ref_dense_kl(
        q_indexer, k_indexer, q_mla, kv_mla, offsets, kv_group=kv_group
    )
    
    print(f"TileLang KL shape: {tl_kl.shape}, mean: {tl_kl.mean().item():.6f}")
    print(f"Reference KL shape: {ref_kl.shape}, mean: {ref_kl.mean().item():.6f}")
    
    assert_tensors_similar(tl_kl, ref_kl, eps=1e-2, name="KL divergence")
    print("✓ Correctness test passed!")
    
    # Benchmark
    def fn():
        return dense_warmup_kl_interface(
            q_indexer, k_indexer, q_mla, kv_mla, offsets,
            kv_group=kv_group, block_K=block_K, num_stages=num_stages, threads=threads
        )
    
    from tilelang.profiler import do_bench
    ms = do_bench(fn, rep=100, warmup=50)
    
    # FLOPs: 2 * S^2 * (D_indexer + D_mla) * H for both attention computations
    flops = 2 * S * S * (D_indexer + D_mla) * H * 2  # 2 for multiply-add
    tflops = flops / (ms * 1e-3) / 1e12
    
    print(f"Average time: {ms:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")
    print(f"Memory saved vs full materialization: {S*S*H*4/(1024**3):.2f} GB")


if __name__ == "__main__":
    test_dense_warmup_kl(
        B=2,
        S=2048,
        H=32,
        HKV=1,
        D_indexer=128,
        D_mla=576,
        dtype=torch.bfloat16,
        block_K=64,
        num_stages=2,
        threads=128,
    )
