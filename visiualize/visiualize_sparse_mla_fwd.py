import torch


def tinfo(name, x, sample_elems=6):
    """打印张量的关键摘要信息，避免输出太大。"""
    if torch.is_tensor(x):
        flat = x.detach().reshape(-1)
        sample = flat[:sample_elems].to("cpu")
        sample_list = sample.tolist()
        print(f"[INFO] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, sample={sample_list}")
    else:
        print(f"[INFO] {name}: {x}")


def ref_sparse_mla_fwd_interface_debug(Q, KV, Indices, offsets, sm_scale=None, is_casual=True):
    print("\n===== Enter ref_sparse_mla_fwd_interface_debug =====")
    tinfo("Q (input)", Q)  # 注释：输入 Q
    tinfo("KV (input)", KV)  # 注释：输入 KV
    tinfo("Indices (input)", Indices)  # 注释：输入稀疏索引
    tinfo("offsets (input)", offsets)  # 注释：offsets 用来切 ragged 序列

    Q = Q.float()
    print("[STEP] Q = Q.float()  # 注释：转 float 方便计算")
    tinfo("Q (float)", Q)

    KV = KV.float()
    print("[STEP] KV = KV.float()  # 注释：转 float 方便计算")
    tinfo("KV (float)", KV)

    all_o = []
    print("[STEP] all_o = []  # 注释：存每段 offset 的输出")
    print(f"[INFO] offsets.shape={tuple(offsets.shape)}  # 注释：有 offsets.shape[0]-1 段")

    # 逐段处理 ragged 序列
    for i in range(offsets.shape[0] - 1):
        print("\n----------------------------------------------")
        print(f"[LOOP] segment i={i}  # 注释：处理第 {i} 段序列")
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        print(f"[INFO] start={start}, end={end}, length={end-start}  # 注释：这一段的 token 数")

        q = Q[None, start:end]
        print("[STEP] q = Q[None, start:end]  # 注释：取出该段 Q，并加 batch 维")
        tinfo("q", q)

        kv = KV[None, start:end]
        print("[STEP] kv = KV[None, start:end]  # 注释：取出该段 KV，并加 batch 维")
        tinfo("kv", kv)

        indices = Indices[None, start:end].clone()
        print("[STEP] indices = Indices[None, start:end].clone()  # 注释：取出该段 indices，并 clone 避免原地影响")
        tinfo("indices (before transpose)", indices)

        indices = indices.transpose(1, 2)
        print("[STEP] indices = indices.transpose(1, 2)  # 注释：把 [b,sq,g,topk] -> [b,g,sq,topk]，用于 scatter")
        tinfo("indices (after transpose)", indices)

        b, sq, h, dim_q = q.shape
        print("[STEP] b, sq, h, dim_q = q.shape  # 注释：读取 Q 的维度含义")
        print(f"[INFO] b={b}, sq={sq}, h={h}, dim_q={dim_q}")

        b2, sk, g, kv_dim = kv.shape
        print("[STEP] b, sk, g, _ = kv.shape  # 注释：读取 KV 的维度含义（sk=key长度，g=group数）")
        print(f"[INFO] b2={b2}, sk={sk}, g={g}, kv_dim={kv_dim}")

        assert kv.shape[-1] == 576, "you should assign dim otherwise"
        print("[ASSERT] kv.shape[-1] == 576  # 注释：这里假设 KV 的最后维是 576 = (512 latent + 64 rope)")

        dim = 512
        print("[STEP] dim = 512  # 注释：把前 512 维当成 V（latent）")

        k = kv
        print("[STEP] k = kv  # 注释：K 使用全部 576 维")
        tinfo("k", k)

        v = kv[..., :dim]
        print("[STEP] v = kv[..., :dim]  # 注释：V 只取前 512 维")
        tinfo("v", v)

        b3, _, _, dim_v = v.shape
        print("[STEP] b, _, _, dim_v = v.shape  # 注释：dim_v 应该是 512")
        print(f"[INFO] b3={b3}, dim_v={dim_v}")

        g_index = g
        print("[STEP] g_index = g  # 注释：g_index 就是 group 数")
        print(f"[INFO] g_index={g_index}")

        h_index = h // g
        print("[STEP] h_index = h // g  # 注释：每个 group 对应的 head 数")
        print(f"[INFO] h_index={h_index}")

        # causal mask：形状 [sq, sk]，True 表示允许 attend
        device = q.device
        print(f"[INFO] device={device}  # 注释：后续 arange 全放在同一设备")

        q_pos = torch.arange(0, sq, dtype=torch.int32, device=device).view(-1, 1)
        print("[STEP] q_pos = arange(0,sq).view(-1,1)  # 注释：query 位置索引，形状 [sq,1]")
        tinfo("q_pos", q_pos)

        k_pos = torch.arange(0, sk, dtype=torch.int32, device=device).view(1, -1)
        print("[STEP] k_pos = arange(0,sk).view(1,-1)  # 注释：key 位置索引，形状 [1,sk]")
        tinfo("k_pos", k_pos)

        compressed_casual_mask = (q_pos >= k_pos)
        print("[STEP] compressed_casual_mask = (q_pos >= k_pos)  # 注释：causal mask，形状 [sq,sk]")
        tinfo("compressed_casual_mask", compressed_casual_mask)

        # indices 越界处理：把 >sk 的索引放到 sk（哨兵位）
        num_bad = (indices > sk).sum().item()
        print(f"[INFO] num_bad_indices={(int(num_bad))}  # 注释：indices 中 > sk 的数量（会被送到哨兵位）")

        indices[indices > sk] = sk
        print("[STEP] indices[indices > sk] = sk  # 注释：越界索引统一写成 sk（哨兵）")
        tinfo("indices (clipped)", indices)

        # scatter 构造稀疏 mask：先开 sk+1（哨兵位），再切掉最后一位
        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool)
        print("[STEP] mask = zeros(b,g,sq,sk+1)  # 注释：先开 sk+1（最后一格当哨兵位）")
        tinfo("mask (init)", mask)

        mask = mask.scatter(3, indices.long(), 1)
        print("[STEP] mask = mask.scatter(dim=3, index=indices, value=1)  # 注释：把 topk 索引位置置 True")
        tinfo("mask (after scatter)", mask)

        mask = mask[..., :-1]
        print("[STEP] mask = mask[..., :-1]  # 注释：去掉哨兵位，回到真实 sk 长度")
        tinfo("mask (after drop sentinel)", mask)

        mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
        print("[STEP] mask &= causal_mask  # 注释：稀疏 topk mask 与 causal mask 取交集")
        tinfo("mask (after causal)", mask)

        # 这行在你原代码里基本没效果，因为 1-1=0 => :0 为空切片
        mask[:, :, : 1 - 1, 0] = True
        print("[STEP] mask[:, :, :1-1, 0] = True  # 注释：这里 1-1=0，等价于 mask[:,:, :0,0]，不做任何事")
        tinfo("mask (after no-op line)", mask)

        mask = mask.view(b, g_index, 1, sq, sk)
        print("[STEP] mask = mask.view(b,g,1,sq,sk)  # 注释：加一个 head 维用于 broadcast 到 score 的 head 维")
        tinfo("mask (final)", mask)

        q = q.view(b, sq, g, -1, dim_q)
        print("[STEP] q = q.view(b,sq,g,h_index,dim_q)  # 注释：把 head 拆成 group 维和组内 head 维")
        tinfo("q (grouped)", q)

        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        print("[STEP] score = einsum(q,k)  # 注释：计算注意力打分，得到 [b,g,h_index,sq,sk]")
        tinfo("score", score)

        if sm_scale is None:
            sm_scale_eff = dim_q ** -0.5
            print("[STEP] sm_scale = dim_q**-0.5  # 注释：默认缩放因子 1/sqrt(dim_q)")
        else:
            sm_scale_eff = sm_scale
            print("[STEP] sm_scale = user provided  # 注释：使用用户传入的缩放因子")
        print(f"[INFO] sm_scale_eff={sm_scale_eff}")

        score = score.masked_fill(~mask, float("-inf"))
        print("[STEP] score = score.masked_fill(~mask, -inf)  # 注释：把不允许 attend 的位置置为 -inf")
        # 注意：这里 score 里会有 -inf，sample 可能打印出 inf
        tinfo("score (masked)", score)

        score = score.mul(sm_scale_eff)
        print("[STEP] score = score * sm_scale  # 注释：缩放 score")
        tinfo("score (scaled)", score)

        p = score.softmax(dim=-1)
        print("[STEP] p = softmax(score, dim=-1)  # 注释：对 key 维做 softmax 得到注意力权重")
        tinfo("p", p)

        p = p.view(b, g_index, h_index, -1, sq, sk)
        print("[STEP] p = p.view(b,g,h_index,?,sq,sk)  # 注释：这里 ? 通常是 1（因为维度刚好匹配）")
        tinfo("p (view1)", p)

        p = p.view(b, g, -1, sq, sk)
        print("[STEP] p = p.view(b,g,heads_per_group,sq,sk)  # 注释：把多余维度并回组内 head 维")
        tinfo("p (view2)", p)

        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        print("[STEP] o = einsum(p,v)  # 注释：注意力加权求和，得到 [b,sq,g,h_index,dim_v]")
        tinfo("o (before reshape)", o)

        o = o.reshape(b, sq, h, dim_v)
        print("[STEP] o = o.reshape(b,sq,h,dim_v)  # 注释：合并 (g,h_index) 回原始 head 数 h")
        tinfo("o (after reshape)", o)

        all_o.append(o.squeeze(0))
        print("[STEP] all_o.append(o.squeeze(0))  # 注释：去掉 batch 维，把该段输出保存起来")
        tinfo("o saved (squeezed)", all_o[-1])

    o = torch.cat(all_o, dim=0)
    print("\n[STEP] o = torch.cat(all_o, dim=0)  # 注释：把各段输出沿 token 维拼回去")
    tinfo("o (concatenated)", o)

    o = o.to(torch.bfloat16)
    print("[STEP] return o.to(bfloat16)  # 注释：输出转 bfloat16")
    tinfo("o (bfloat16)", o)

    print("===== Exit ref_sparse_mla_fwd_interface_debug =====\n")
    return o


def main():
    # 设备选择：优先 CUDA；没 CUDA 也能在 CPU 跑（只是更慢）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Using device: {device}")

    # 构造一个 ragged batch：两段长度分别 4 和 5，总长 9
    offsets = torch.tensor([0, 4, 9], dtype=torch.int64, device=device)
    print("[MAIN] offsets = [0,4,9]  # 注释：两段序列：len=4 和 len=5")
    tinfo("offsets", offsets)

    # 设定维度
    total_tokens = int(offsets[-1].item())
    h = 4        # 总 head 数
    g = 2        # group 数（GQA），要求 h % g == 0
    dim_q = 576  # Q 的 head_dim（要和 K 的最后维一致）
    kv_dim = 576 # KV 最后维（你函数里 assert 576）
    topk = 3     # 每个 query 选 topk 个 key 位置

    # 构造 Q: [T, h, dim_q]
    Q = torch.randn(total_tokens, h, dim_q, device=device, dtype=torch.float16)
    print("[MAIN] Q = randn(T,h,dim_q)  # 注释：随机 Q")
    tinfo("Q", Q)

    # 构造 KV: [T, g, 576]
    KV = torch.randn(total_tokens, g, kv_dim, device=device, dtype=torch.float16)
    print("[MAIN] KV = randn(T,g,576)  # 注释：随机 KV（按 group 存）")
    tinfo("KV", KV)

    # 构造 Indices: [T, g, topk]，索引范围 [0, sk-1]（每段 sk 可能不同，这里先全局随机，后面会 clip）
    # 为了让输出更稳定，我们让 indices 在 [0, total_tokens-1]，函数里会 clip 到 sk
    Indices = torch.randint(low=0, high=total_tokens, size=(total_tokens, g, topk), device=device, dtype=torch.int32)
    print("[MAIN] Indices = randint(T,g,topk)  # 注释：随机 topk key 索引（会在函数内按 sk clip）")
    tinfo("Indices", Indices)

    # 调用 debug 版本
    out = ref_sparse_mla_fwd_interface_debug(Q, KV, Indices, offsets, sm_scale=None, is_casual=True)
    print("[MAIN] out computed  # 注释：完成前向计算")
    tinfo("out", out)


if __name__ == "__main__":
    main()
