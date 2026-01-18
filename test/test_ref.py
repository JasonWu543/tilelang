# ruff: noqa
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(ROOT, "examples", "dsa_sparse_finetune")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from dense_warmup_kl_fwd_cc import dense_warmup_kl_interface
from index import prepare_token_indices


def torch_ref_dense_kl(
    Q_indexer, K_indexer, Q_mla, KV_mla, offsets, kv_group=1,
    sm_scale_indexer=None, sm_scale_mla=None
):
    S, H, Didx = Q_indexer.shape
    _, G, Dmla = KV_mla.shape
    assert G == H // kv_group

    dim_v = 512
    tail_dim = Dmla - dim_v

    if sm_scale_indexer is None:
        sm_scale_indexer = Didx ** -0.5
    if sm_scale_mla is None:
        sm_scale_mla = Dmla ** -0.5

    token_indices = prepare_token_indices(offsets)
    b = token_indices[:, 0]
    s = token_indices[:, 1]
    bos = offsets[b]
    eos = offsets[b + 1]
    maxk = bos + s

    k = torch.arange(S, device=Q_indexer.device)
    mask = (k[None, :] >= bos[:, None]) & (k[None, :] < eos[:, None]) & (k[None, :] <= maxk[:, None])

    t = torch.einsum("shd,Shd->shS", Q_indexer.float(), K_indexer.float()).mul(sm_scale_indexer)
    t = t.masked_fill(~mask[:, None, :], float("-inf"))

    K_mla = KV_mla.float()
    K_mla_expanded = K_mla.repeat_interleave(kv_group, dim=1)

    s_main = torch.einsum(
        "shd,Shd->shS",
        Q_mla[..., :dim_v].float(),
        K_mla_expanded[..., :dim_v].float(),
    ).mul(sm_scale_mla)
    if tail_dim > 0:
        s_tail = torch.einsum(
            "shd,Shd->shS",
            Q_mla[..., dim_v:].float(),
            K_mla_expanded[..., dim_v:].float(),
        ).mul(sm_scale_mla)
        s_logits = s_main + s_tail
    else:
        s_logits = s_main

    s_logits = s_logits.masked_fill(~mask[:, None, :], float("-inf"))

    pt = torch.softmax(t, dim=-1)
    log_pt = torch.log_softmax(t, dim=-1)
    log_ps = torch.log_softmax(s_logits, dim=-1)
    kl = (pt * (log_pt - log_ps)).sum(dim=-1)
    return kl


def make_offsets(batch_size, seq_len, device):
    lens = [seq_len // batch_size] * batch_size
    lens[-1] += seq_len - sum(lens)
    offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(lens):
        offsets[i + 1] = offsets[i] + l
    return offsets


def main():
    torch.manual_seed(0)

    B = 2
    S = 128
    H = 8
    HKV = 1
    D_indexer = 128
    D_mla = 576
    dtype = torch.bfloat16

    kv_group = H // HKV
    device = "cuda"

    q_indexer = torch.randn((S, H, D_indexer), dtype=dtype, device=device)
    k_indexer = torch.randn((S, H, D_indexer), dtype=dtype, device=device)
    q_mla = torch.randn((S, H, D_mla), dtype=dtype, device=device)
    kv_mla = torch.randn((S, HKV, D_mla), dtype=dtype, device=device)
    offsets = make_offsets(B, S, device)

    tl = dense_warmup_kl_interface(
        q_indexer, k_indexer, q_mla, kv_mla, offsets,
        kv_group=kv_group, block_K=32, num_stages=2, threads=128
    )
    ref = torch_ref_dense_kl(
        q_indexer, k_indexer, q_mla, kv_mla, offsets, kv_group=kv_group
    )

    diff = (tl - ref).float()
    print("tl", tl.shape, "ref", ref.shape)
    print("max", diff.abs().max().item(), "mean", diff.abs().mean().item())

    s = 3
    h = 0
    print("tl[3,0]", tl[s, h].item(), "ref[3,0]", ref[s, h].item())

    tol = 1e-2
    if torch.isnan(diff).any():
        print("NaN detected in diff")
    else:
        print("allclose", torch.allclose(tl, ref, atol=tol, rtol=0.0))


if __name__ == "__main__":
    main()
