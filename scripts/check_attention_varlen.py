"""Check that the sparse varlen attention matches a per-batch reference.

`full_attn.sparse_scaled_dot_product_attention` needs flash-attn's cumulative sequence
lengths on the device. It used to build them from the layout on every call (cumsum, cat,
cast, pageable upload); it now reads the layout's cached `cum_seqlen` and only casts. A wrong
or stale cache would silently mix batches, so this runs multi-batch sparse tensors and
compares against a direct softmax attention per batch, for both call shapes the transformer
uses: self-attention (`qkvpacked`) and the cross-attention that packs keys and values
(`kvpacked`), the latter with query and key lengths that differ.

Features are fp16 on the device and the reference is fp32, so the comparison is a tolerance
rather than bit-exactness; a misplaced batch boundary shows up as a mismatch of order 1.

Usage: python scripts\\check_attention_varlen.py
"""

import sys
from typing import List

import torch

import uniforge3d.modules.sparse as sp
from uniforge3d.modules.sparse import config as sparse_config
from uniforge3d.modules.sparse.attention import full_attn

HEADS = 4
CHANNELS = 32
TOLERANCE = 2e-2
SCALE = 1 / CHANNELS ** 0.5


def sparse_tensor(lengths: List[int], trailing: tuple, seed: int) -> sp.SparseTensor:
    """A sparse tensor holding batches of the given lengths, coordinates in a 32^3 grid."""
    torch.manual_seed(seed)
    coords = []
    for batch, voxels in enumerate(lengths):
        cells = torch.unique(torch.randint(0, 32, (4 * voxels, 3), dtype=torch.int32), dim=0)[:voxels]
        coords.append(torch.cat([torch.full((voxels, 1), batch, dtype=torch.int32), cells], dim=1))
    coords = torch.cat(coords, dim=0).cuda()
    return sp.SparseTensor(torch.randn(coords.shape[0], *trailing, device="cuda").half(), coords)


def cum_seqlen(tensor: sp.SparseTensor, lengths: List[int]) -> torch.Tensor:
    """The cumulative lengths as the replaced inline expression built them."""
    return torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(lengths), dim=0)]
    ).int().to(tensor.device)


def align(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, result: torch.Tensor) -> None:
    """Write per-head softmax attention of one batch into `result`, on the query rows."""
    weights = (query @ keys.transpose(-1, -2)) * SCALE
    result.copy_((weights.softmax(-1) @ values).transpose(0, 1))


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    print(f"attention module: {sparse_config.ATTN}")

    self_lengths = [64, 96, 128]
    query_lengths = [48, 80, 120]
    key_lengths = [64, 96, 128]
    worst = 0.0

    qkv = sparse_tensor(self_lengths, (3, HEADS, CHANNELS), seed=0)
    with torch.no_grad():
        out = full_attn.sparse_scaled_dot_product_attention(qkv)
    assert torch.equal(
        full_attn._cum_seqlen(qkv, self_lengths, qkv.device), cum_seqlen(qkv, self_lengths)
    ), "cached cu_seqlens differ from the built ones"
    reference = torch.empty_like(out.feats)
    for layout_slice in qkv.layout:
        q, k, v = (
            qkv.feats[layout_slice, index].float().transpose(0, 1) for index in range(3)
        )
        align(q, k, v, reference[layout_slice])
    delta = (out.feats.float() - reference).abs().max().item()
    print(f"self: lengths {self_lengths}, max abs diff {delta:.3e}")
    worst = max(worst, delta)

    q = sparse_tensor(query_lengths, (HEADS, CHANNELS), seed=1)
    kv = sparse_tensor(key_lengths, (2, HEADS, CHANNELS), seed=2)
    with torch.no_grad():
        out = full_attn.sparse_scaled_dot_product_attention(q, kv)
    assert torch.equal(
        full_attn._cum_seqlen(q, query_lengths, q.device), cum_seqlen(q, query_lengths)
    ), "cached query cu_seqlens differ from the built ones"
    assert torch.equal(
        full_attn._cum_seqlen(kv, key_lengths, kv.device), cum_seqlen(kv, key_lengths)
    ), "cached key cu_seqlens differ from the built ones"
    reference = torch.empty_like(out.feats)
    for batch in range(len(query_lengths)):
        q_rows = q.layout[batch]
        kv_rows = kv.layout[batch]
        align(
            q.feats[q_rows].float().transpose(0, 1),
            kv.feats[kv_rows, 0].float().transpose(0, 1),
            kv.feats[kv_rows, 1].float().transpose(0, 1),
            reference[q_rows],
        )
    delta = (out.feats.float() - reference).abs().max().item()
    print(f"cross: queries {query_lengths}, keys {key_lengths}, max abs diff {delta:.3e}")
    worst = max(worst, delta)

    if worst > TOLERANCE:
        print("FAIL: attention output does not match the per-batch reference")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
