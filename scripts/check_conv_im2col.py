"""Check the explicit-GEMM im2col against a direct reference.

`conv_flex_gemm.sparse_conv3d_forward` builds im2col one chunk at a time and used to resolve
the no-neighbour sentinel with boolean indexing, which runs a `nonzero` per chunk and thus a
device synchronisation. It now gathers with the sentinel clamped to row 0 and zeroes the
missing taps with a multiply, so the chunk needs no data-dependent shape. Both spellings have
to agree exactly, including on the taps that have no neighbour, so this compares the
convolution against a float64 reference built straight from the neighbour map.

Features and weights are small integers: every product and sum is then exact in fp32 and the
result is exact in fp16, so the two accumulation orders (GEMM vs the reference sum) must be
bit-identical rather than merely close.

Usage: python scripts\\check_conv_im2col.py
"""

import sys

import torch

import uniforge3d.modules.sparse as sp
from uniforge3d.modules.sparse import config as sparse_config
from uniforge3d.modules.sparse.conv import conv_flex_gemm

CHANNELS = 8
VOXELS = 400
GRID = 8


def reference(
    rows: torch.Tensor, feats: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Sum `feats[rows[:, tap]] @ weight[:, tap]` per output voxel, skipping missing taps."""
    out_channels, kd, kh, kw, in_channels = weight.shape
    out = torch.zeros(feats.shape[0], out_channels, dtype=torch.float64, device=feats.device)
    for tap in range(kd * kh * kw):
        flat = rows[:, tap].int().long()
        gathered = torch.zeros(feats.shape[0], in_channels, dtype=torch.float64, device=feats.device)
        found = flat >= 0
        gathered[found] = feats[flat[found]].double()
        kernel = weight[:, tap // (kh * kw), (tap // kw) % kh, tap % kw, :].double()
        out += gathered @ kernel.t()
    return out + bias.double()


def run(offload_bytes: int) -> float:
    torch.manual_seed(0)
    coords = torch.unique(torch.randint(0, GRID, (3 * VOXELS, 3), device="cuda", dtype=torch.int32), dim=0)[:VOXELS]
    coords = torch.cat([torch.zeros(VOXELS, 1, dtype=torch.int32, device="cuda"), coords], dim=1)
    feats = torch.randint(1, 4, (VOXELS, CHANNELS), device="cuda").half()

    conv = sp.SparseConv3d(CHANNELS, CHANNELS, 3).cuda().half()
    conv.weight.data = torch.randint(-2, 3, conv.weight.shape, device="cuda").half()
    conv.bias.data = torch.randint(-2, 3, conv.bias.shape, device="cuda").half()
    conv.eval()

    x = sp.SparseTensor(feats.clone(), coords)
    with torch.no_grad():
        sparse_config.OFFLOAD_BYTES = offload_bytes
        out = conv(x).feats.double()

    rows, _ = conv_flex_gemm._submanifold_neighbor_map(
        sp.SparseTensor(feats.clone(), coords), (3, 3, 3), (1, 1, 1)
    )
    missing = (rows.int() < 0).sum().item()
    expected = reference(rows, feats, conv.weight, conv.bias)
    delta = (out - expected).abs().max().item()
    print(
        f"chunk {conv_flex_gemm._IM2COL_CHUNK_BYTES // 1024} KB,"
        f" offload {offload_bytes // (1024 * 1024)} MB:"
        f" {missing} missing taps of {rows.numel()}, max abs diff {delta:.3e}"
    )
    return delta


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1

    # Small chunks force many iterations, including chunks that end on a sentinel row.
    default_chunk = conv_flex_gemm._IM2COL_CHUNK_BYTES
    conv_flex_gemm._IM2COL_CHUNK_BYTES = 2 * 1024
    try:
        worst = max(run(1 << 60), run(256 * 1024 * 1024))
    finally:
        conv_flex_gemm._IM2COL_CHUNK_BYTES = default_chunk
    if worst != 0.0:
        print("FAIL: im2col does not match the reference")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
