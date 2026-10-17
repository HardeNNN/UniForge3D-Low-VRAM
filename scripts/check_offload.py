"""Check that the flex_gemm output offload cannot corrupt a caller's input.

`conv_flex_gemm.sparse_conv3d_forward` releases the caller's input features when it has to
accumulate an oversized output in RAM. That is only legal when the caller is finished with
that tensor, so any block that reads its input after the convolution keeps a copy: above
`config.OFFLOAD_BYTES` it parks that copy in host RAM and adds it back in chunks, below it
it reads the device tensor directly. This builds a sparse tensor above the offload
threshold and asserts the residual block's output matches the same block with the offload
disabled, which is the same as comparing the parked residual against a resident one.

Usage: python scripts\\check_offload.py
"""

import sys

import torch

import uniforge3d.modules.sparse as sp
from uniforge3d.modules.sparse import config as sparse_config
from uniforge3d.models.vaes.sparse_unet import SparseConvNeXtBlock3d

CHANNELS = 128
VOXELS = 1_200_000  # x128 channels x 2 B is above the 256 MB offload threshold


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"device free {free / 2**30:.2f}/{total / 2**30:.2f} GB")

    torch.manual_seed(0)
    coords = torch.randint(0, 128, (2 * VOXELS, 3), device="cuda", dtype=torch.int32)
    coords = torch.unique(coords, dim=0)[:VOXELS]
    coords = torch.cat([torch.zeros(VOXELS, 1, dtype=torch.int32, device="cuda"), coords], dim=1)
    feats = torch.randn(VOXELS, CHANNELS, device="cuda", dtype=torch.float16)

    # The release is the documented behaviour of the offload: it makes room for the output.
    x = sp.SparseTensor(feats.clone(), coords)
    sp.SparseConv3d(CHANNELS, CHANNELS, 3).cuda().half()(x)
    print(f"conv released its input features: x.feats = {tuple(x.feats.shape)} {x.feats.device}")

    # A block whose residual reads the input must be unaffected by that release.
    block = SparseConvNeXtBlock3d(CHANNELS).cuda().float().eval()
    with torch.no_grad():
        sparse_config.OFFLOAD_BYTES = 1 << 60
        reference = block(sp.SparseTensor(feats.float(), coords))
        sparse_config.OFFLOAD_BYTES = 256 * 1024 * 1024
        offloaded = block(sp.SparseTensor(feats.float(), coords))
    assert reference.feats.shape == offloaded.feats.shape, (reference.feats.shape, offloaded.feats.shape)
    delta = (reference.feats - offloaded.feats).abs().max().item()
    print(f"SparseConvNeXtBlock3d offload vs no offload: max abs diff {delta:.3e} shape {tuple(offloaded.feats.shape)}")
    if delta != 0.0:
        print("FAIL: the offload changed the result")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
