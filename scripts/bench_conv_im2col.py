"""Compare the explicit-GEMM im2col chunk loop against the boolean-indexing version.

`conv_flex_gemm.sparse_conv3d_forward` used to resolve the no-neighbour sentinel with boolean
indexing, which runs a `nonzero` per chunk and therefore reads a size back to the host. The
end-to-end pipeline cannot show the difference: its dual-contouring remesh varies by nearly an
order of magnitude in vertex count between runs of identical code, and the decode stage's cost
tracks that, not this loop. So this times the loop itself on a fixed sparse input of a size the
decode stage actually sees, against a copy of the replaced body, and counts the `aten::nonzero`
calls each version makes under the CPU profiler.

Usage: python scripts\\bench_conv_im2col.py [voxels] [channels]
"""

import sys
import time

import torch

import uniforge3d.modules.sparse as sp
from uniforge3d.modules.sparse import config as sparse_config
from uniforge3d.modules.sparse.conv import conv_flex_gemm

VOXELS = 150_000
CHANNELS = 512
GRID = 64
ITERATIONS = 3


def build(channels: int, voxels: int) -> tuple[sp.SparseTensor, sp.SparseConv3d]:
    torch.manual_seed(0)
    cells = torch.unique(torch.randint(0, GRID, (2 * voxels, 3), dtype=torch.int32), dim=0)[:voxels]
    coords = torch.cat(
        [torch.zeros(cells.shape[0], 1, dtype=torch.int32), cells], dim=1
    ).cuda()
    feats = torch.randn(cells.shape[0], channels, device="cuda").half()
    conv = sp.SparseConv3d(channels, channels, 3).cuda().half()
    conv.eval()
    return sp.SparseTensor(feats, coords), conv


def boolean_indexing_forward(conv: sp.SparseConv3d, x: sp.SparseTensor) -> sp.SparseTensor:
    """The chunk loop as it was before: boolean indexing to resolve the sentinel."""
    conv_flex_gemm.flex_gemm.ops.spconv.set_algorithm(conv_flex_gemm.conv_config.FLEX_GEMM_ALGO)
    conv_flex_gemm.flex_gemm.ops.spconv.set_hashmap_ratio(conv_flex_gemm.conv_config.FLEX_GEMM_HASHMAP_RATIO)

    Co, Kd, Kh, Kw, Ci = conv.weight.shape
    taps = Kd * Kh * Kw
    neighbor_map, stream_from_cpu = conv_flex_gemm._submanifold_neighbor_map(
        x, conv.weight.shape[1:4], conv.dilation
    )

    feats = x.feats
    voxels = feats.shape[0]
    weight_matrix = conv.weight.reshape(Co, taps * Ci).t()
    chunk = max(1024, conv_flex_gemm._IM2COL_CHUNK_BYTES // (taps * Ci * feats.element_size()))
    output = torch.empty((voxels, Co), dtype=feats.dtype, device=feats.device)
    for start in range(0, voxels, chunk):
        end = min(start + chunk, voxels)
        rows = neighbor_map[start:end]
        if stream_from_cpu:
            rows = rows.to(feats.device, non_blocking=True)
        im2col = torch.zeros(((end - start) * taps, Ci), device=feats.device, dtype=feats.dtype)
        flat = rows.reshape(-1).int()
        found = flat != -1
        im2col[found] = feats[flat[found].long()]
        im2col = im2col.view(end - start, taps * Ci)
        gemm = torch.addmm(conv.bias, im2col, weight_matrix)
        output[start:end] = gemm
    return x.replace(output)


def count_nonzero(call) -> int:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        call()
    return sum(
        event.count
        for event in profile.key_averages()
        if event.key.startswith("aten::nonzero")
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    voxels = int(sys.argv[1]) if len(sys.argv) > 1 else VOXELS
    channels = int(sys.argv[2]) if len(sys.argv) > 2 else CHANNELS

    x, conv = build(channels, voxels)
    taps = 27
    chunk = max(1024, conv_flex_gemm._IM2COL_CHUNK_BYTES // (taps * channels * 2))
    chunks = -(-voxels // chunk)
    print(
        f"{x.feats.shape[0]} voxels, {channels} channels, fp16, chunk {chunk} rows"
        f" -> {chunks} chunks, offload budget {sparse_config.OFFLOAD_BYTES // (1024 * 1024)} MB"
    )

    with torch.no_grad():
        results = {}
        for label, call in (
            ("boolean indexing", lambda: boolean_indexing_forward(conv, x)),
            ("clamped gather", lambda: conv_flex_gemm.sparse_conv3d_forward(conv, x)),
            ("boolean indexing", lambda: boolean_indexing_forward(conv, x)),
        ):
            call()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(ITERATIONS):
                output = call()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / ITERATIONS
            results[label] = output.feats.clone()
            print(f"{label:>18}: {elapsed * 1000:8.1f} ms/call, {count_nonzero(call)} aten::nonzero per call")

    delta = (results["boolean indexing"] - results["clamped gather"]).abs().max().item()
    print(f"max abs diff {delta:.3e}")
    return 0 if delta == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
