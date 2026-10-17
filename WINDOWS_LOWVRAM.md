# Windows + 8 GB VRAM

Upstream UniForge3D targets a Linux box with a large GPU: its setup builds CUDA
extensions from source, and it keeps a 1.7B stage-1 DiT, a 1.2B VGGT backbone, three
1.7B cascade denoisers and two VAE decoders in fp32. This fork runs the same model on a
consumer 8 GB card with prebuilt Windows wheels and a staged residency strategy.

Tested configuration: Windows 11, Python 3.11, RTX 3070 Ti (8 GB), 64 GB RAM.

## 1. Requirements

| Item | Notes |
| --- | --- |
| Python 3.11 | every prebuilt CUDA wheel here is `cp311 win_amd64`. |
| Disk | ~30 GB of weights + ~4 GB of wheels + outputs. |
| NVIDIA driver | recent enough for CUDA 12.8. |
| git + `curl.exe` | `git` only to clone; `curl.exe` ships with Windows. |

```powershell
winget install Python.Python.3.11
git clone https://github.com/HardeNNN/UniForge3D-Low-VRAM
cd UniForge3D-Low-VRAM
```

## 2. Install

`scripts/fetch.py` downloads every large file over many parallel HTTP range requests,
resuming each chunk it has to retry. Plain `pip` and `huggingface_hub` transfers stalled
on long downloads over this link, so keep the fetch separate from the install.

```powershell
python scripts\fetch.py wheels          # torch, torchvision, flash-attn, the 5 CUDA extension wheels -> whl/
scripts\install_windows.ps1             # venv + local wheels + pinned dependencies
python scripts\fetch.py weights         # the 17 model files -> weights/  (~30 GB)
python scripts\fetch.py extras          # dinov3 + RMBG-2.0 -> models/   (unzips on completion)
python scripts\make_local_config.py     # weights/pipeline.local.json
```

The wheels are all `cp311`, and `fetch.py wheels` refuses to run under another Python rather
than assemble an environment the extensions cannot load into.

`fetch.py` is resumable: re-run it after an interruption and it continues from the parts
already on disk. `--jobs N` fetches N files at once, `--connections N` sets how many
range chunks are in flight.

Gated models (`facebook/dinov3-*`, `briaai/RMBG-2.0`) come from the ungated mirror
release used by the `TRELLIS.2-stableprojectorz` project, so no Hugging Face token is
needed.

Nothing here uses `huggingface_hub` to transfer weights: the client in the pinned
version can stall or misfire on very large files, which is why `fetch.py` talks plain
HTTP instead. The installer therefore pins `huggingface-hub>=0.36,<1` up front, matching
the client that `transformers==4.57.3` expects.

## 3. Run

```powershell
.\.venv\Scripts\python.exe infer.py --config weights\pipeline.local.json --vram-debug
```

`weights\pipeline.local.json` is generated from the upstream `pipeline.json` with every
Hub reference rewritten to a local file, so inference never contacts the Hub.
`--vram-debug` prints allocator and device statistics plus the elapsed time of each stage.

| Flag | Effect |
| --- | --- |
| `--low-vram` / `--no-low-vram` | staged residency + fp16 weights (default on). |
| `--weights-dtype {float16,float32,auto}` | weight precision for the denoisers; `auto` takes whatever the residency mode does anyway, which is fp16 under the default `--low-vram` and each checkpoint's published precision with `--no-low-vram`. |
| `--device N` | CUDA device index. |
| `--limit N` | process at most N cases that do not have a result yet. |
| `--overwrite` | recompute cases that already have a `prediction.glb`. |
| `--seed N` | override the config seed. |
| `--save-textures` | also write the baked base colour and metallic-roughness maps as PNG beside each `prediction.glb`. |
| `--low-ram` | keep only the stage being run in memory: one group of weights instead of the whole stack, at the cost of reloading them for every case. |
| `--prefetch` / `--no-prefetch` | load later stages' weights on a background thread while the GPU runs the current stage (default on, and never used by `--low-ram`). |
| `--vram-debug` | per-stage VRAM and timing report (or set `UNIFORGE_VRAM_DEBUG=1`). |
| `--watch SECONDS` | stay resident and re-scan `--input` for new cases every SECONDS. |

Cases are directories of images under `--input` (default `assets\`); `--input` may also be
a single case directory. A case whose `prediction.glb` already exists is skipped, so a batch
is resumable. Weights are loaded one stage group at a time, so the first case does not wait
for the whole stack before it starts (section 5).

`--watch` turns the process into a resident server: every stage group is resident after the
first case, then every case that appears under `--input` costs ~225 s. Weights stay in RAM
(~20 GB) and are staged onto the GPU per step, so nothing is re-read from disk. A case that
fails is reported and skipped instead of ending the process; if the failure poisoned the
CUDA context the process exits, because nothing can recover from that. `--watch` refuses
`--overwrite`, which would recompute every case on every rescan:

```powershell
.\.venv\Scripts\python.exe infer.py --config weights\pipeline.local.json --watch 30 --vram-debug
```

### Web UI

```powershell
.\.venv\Scripts\python.exe webui.py            # http://127.0.0.1:7860
```

`start_webui.bat` in the repository root runs exactly that, and `start_webui_lowram.bat` runs
it with `--low-ram`: double-click launchers for the two residency modes, extra flags passed
through. `convert_weights_fp16.bat` is the third launcher and is about disk, not residency:
it halves the fp32 checkpoints once and points `weights\pipeline.local.json` at the copies
(section 5).

`webui.py` loads the same pipeline and keeps it resident. Load 2-4 images, press Generate,
and the page streams the log the command line prints; when the case finishes it shows the
GLB in a viewer and offers it for download. Jobs are serialised, because one case at a time
is all the card holds. Each job writes `outputs/webui/<case>/`, holding a copy of the images
it ran on, `prediction.glb` and — with the texture box ticked, as it is by default —
`textures/basecolor.png` and `textures/metallic_roughness.png`. `--port`, `--host`, `--open`,
`--config`, `--device`, `--low-vram`, `--low-ram`, `--prefetch` and `--weights-dtype` mirror
the flags above; the page deliberately exposes no quality knobs, so the step counts stay
where the config puts them. It binds to the loopback interface and offers no public share
link.

`gradio` comes from `install_windows.ps1`. The page is a thin wrapper around
`pipeline.run`, so it cannot diverge from the command line. The interactive viewer is
Gradio's Babylon.js component, which asks for WebGPU: with WebGPU switched off in the
browser — a common way to free VRAM — the viewer stays empty, while the GLB download still
works.

## 4. How the 8 GB budget is met

1. **fp16 weights for the denoisers.** Stage 1 holds a 1.7B DiT *and* a 1.2B VGGT
   backbone at once; in fp32 that is ~12 GB of parameters alone, in fp16 ~5.9 GB. The
   cascade denoisers drop from ~7 GB to ~3.4 GB each. Decoders keep the fp16 torso their
   checkpoints were published with, and DINO/VGGT-conditioner heads keep their declared
   precision. Int8 weight quantisation is deliberately not used: the stages that exceed the
   card are activation-bound (decoded meshes of 6.5-7.8M vertices, the export remesh), where
   weights are a negligible share, and none of the hot kernels — flex_gemm sparse
   convolution, flash-attn varlen, cumesh dual contouring, xatlas — take int8 weights, so
   dequantisation would be added to a workload the card is not saturating. It would buy
   roughly 3 GB in the one stage that already has headroom (the DiT+VGGT pair).
2. **Autocast follows the weights.** Flow sampling runs under `autocast(dtype=fp16)` so
   ops stay in fp16 instead of being upcast per layer, and initial noise starts in the
   model dtype rather than fp32.
3. **Staged residency.** Only the models a step needs are on the GPU: DINO → stage 1 →
   Shape-512 → Shape-1024 → texture sampling → decoders → bake. Each group is offloaded
   to RAM and the allocator released before the next one loads.
4. **Texture stage split.** The texture flow model and the VGGT conditioner (~5.5 GB in
   fp16) are released after sampling, before the two VAE decoders run.
5. **Decoders run outside autocast with an fp32 boundary.** A decoder keeps its input and
   output layers in fp32 and casts only its torso to `self.dtype`, so the three decode
   helpers (`decode_sparse_structure`, `decode_shape_latent`, `decode_shape_and_texture`)
   cast incoming latents to fp32. The sparse-structure decoder has no normalization to
   promote them, so an fp16 latent from the fp16 sampler would otherwise reach fp32
   convolution weights, which upstream only avoided because autocast covered the decoder.
6. **Allocator settings.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
   before CUDA initialises on platforms that implement it, which keeps sparse 3D decoding
   from reserving far more VRAM than it holds. Windows builds do not implement it, so the
   figures in section 5 were reached without it.
7. **Sparse convolution without Triton.** The bundled flex_gemm wheel ships its CUDA
   hashmap kernels but none of its Triton kernels, so `FLEX_GEMM_ALGO` is
   `explicit_gemm` and `conv_flex_gemm` runs the im2col + GEMM itself in chunks: the
   im2col buffer is capped at 64 MB, an oversized neighbor map is built with the
   features parked in RAM, and an output above `config.OFFLOAD_BYTES` (256 MB) is
   accumulated in RAM. A whole
   im2col of a million voxels at 512 channels would be 27 GB in fp16. Releasing the input
   before that output lands is what keeps a dense decode inside the budget, so a block that
   reads its input *after* a convolution cannot also hold it on the device: above the same
   budget `SparseConvNeXtBlock3d` parks its residual in host RAM and adds it back 64 MB at a
   time. That is what makes the release legal, and it is also what keeps a full input-sized
   buffer from sitting on the device beside the output that replaced it.
   `scripts\check_offload.py` asserts the offload cannot change a result. Texture sampling
   uses the wheel's `grid_sample_3d_torch`, which drives the same CUDA hashmaps with the
   same weights-normalised trilinear maths as the Triton kernel it replaces.
8. **Every export remesh is capped at the official 512 grid.** Both dual-contouring sites
   build a shell only to decimate it straight back to ~1M faces, so the shell density
   `grid_size` asked for was pure waste: at 1024 one case produced 60.5M faces whose
   simplification alone cost 192 s and pushed the process to 10.6 GB. The baked texture does
   not depend on it — it samples the full-resolution attribute volume and projects onto the
   original mesh through a BVH built before remeshing — so `to_glb` uses
   `min(grid_size, 512)`, exactly as the `o_voxel` exporter in the venv does. The Shape-512
   stage's `official_shape_only_postprocess` takes the same cap from
   `utils.mesh.MAX_DC_RESOLUTION`: its only consumer is `surface_voxelize` at 64^3, which is
   far coarser than either grid. Uncapped, that stage asked for ~8.5 GB on an 8 GB card.

## 5. Measured budget and timings on an RTX 3070 Ti (8 GB)

`infer.py --vram-debug` on the heaviest sampled 4-image case, fp16 weights, staged
residency. "Peak used" is the lowest free VRAM seen during the stage, so it is what the
stage actually needed; the log's `peak` field is instead the largest allocation the case
reached, printed cumulatively and reset per case.

| Stage | Time | Reserved after | Peak used |
| --- | --- | --- | --- |
| DINO features | 3.5 s | 0.04 GB | 1.14 GB |
| Stage 1 DiT | 20.3 s | 0.04 GB | 1.15 GB |
| Stage 1 decode | 0.3 s | 0.05 GB | 1.15 GB |
| Shape-512 denoiser | 13.6 s | 0.07 GB | 1.18 GB |
| Shape-512 decode | 5.4 s | 2.25 GB | 3.37 GB |
| Shape-1024 denoiser | 74.6 s | 0.17 GB | 1.29 GB |
| Texture denoiser | 70.4 s | 0.12 GB | 1.24 GB |
| Texture decoders | 8.4 s | 1.43 GB | 2.57 GB |
| Texture export | 24.7 s | 3.05 GB | 4.20 GB |

All 25 GiB of weights take **101.9 s** to load in sequence, but they no longer have to be in
memory before a case starts: they arrive a stage at a time, so the run above began its first
denoiser step 21 s in (see *Host memory* below). Each case still costs **224.4 s** of work
plus that Stage-1 wait, and produces a 36 MB GLB (789k vertices, 937k faces, baked PBR
texture) that verifies as glTF 2.0 with matching declared size, both textures, and geometry
inside the unit cube. A four-case batch on the same build ran its second case in 232.7 s with
the same stage profile and a 6.14 GB GPU peak. `--vram-debug` also reports how much of the card is left before
the first allocation: on a desktop with a browser and an Electron app open it reads
`free at start 6.94/8.00 GB (1.06 GB held by other processes or the driver)`, and that — not
8.00 GB — is the budget every stage has to fit.

The same case with the 1024 grid of section 4.8 exported in **289 s** at a 10.63 GB process
peak with **0 MB free** — the driver was paging the whole stage through system RAM, which is
both why it took 289 s and why a batch run died there with a transient illegal memory
access. The Shape-512 stage failed the same way before section 4.8's cap: **62.5 s** and
`0.00/8.00 GB free` at a 6.32 GB peak on the following case, against **5.4 s** and 3.37 GB
peak once capped. Sampling accounts for the rest: each cascade is 20 steps of a
1.37B-parameter denoiser at ~2 s per forward with the conditional and unconditional branches
run separately, which measures as compute-bound at about half the card's fp16 tensor
throughput (the dense GEMMs land on 44 of the 3070 Ti's 87 TFLOPS).
`nvidia-smi dmon` during a dense case reports `sm` at 100% but only 116-177 W of the card's
290 W budget, with PCIe traffic in both directions — that is the state that accompanies the
paging above, so the per-forward timing stays the meaningful cost measure.

Cost is set by the shape's token count, not by VRAM, and the range across cases is large:

| Case | Decoded mesh | Shape-1024 denoiser | Texture denoiser | Decoders | Export | Total | Case peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case04 | 4.08M verts | 81.2 s | 77.7 s | 8.6 s | 47.9 s | 255.7 s | 6.17 GB |
| case06 | 5.88M verts | 134.1 s | 131.1 s | 19.6 s | 80.6 s | 425.7 s | 7.33 GB |
| case03 | 6.55M verts | 148.0 s | 143.2 s | 29.5 s | 62.7 s | 437.0 s | 9.19 GB |
| case05 | 7.81M verts | 505.0 s | 388.2 s | 78.2 s | 56.3 s | 1090.3 s | 11.24 GB |

Two things follow. A dense case spends most of its time in the denoisers — 893 of case05's
1090 s — and that stage peaks at 6.26 GB, inside the budget, so its cost is token-count
compute rather than paging. The decode and export peaks instead grow with the decoded mesh,
by roughly 0.6-1.4 GB per million vertices above the ~6.2 GB baseline: that is the official
`o_voxel` dual-grid conversion, which materialises per-voxel neighbour, hashmap and quad
arrays of its own, plus the subdivision state the texture decoder is guided by. Both are
third-party or architectural, so no chunk size or offload threshold moves them — parking the
residual per section 4.7 changed case06's decode stage by 0.2%. Capping the decoded
resolution would change geometry, so the peaks stand: a case decoding above ~7M vertices
pages and takes 2-4x longer than one that fits.

The batch is not what makes a dense case slow. Case05 costs **1001 s** run alone against
1090 s inside the four-case batch — 424.0 s of shape-1024 denoiser against the batch's
505.0 s, 381.5 s of texture denoiser against 388.2 s, 65.7 s of decoders, 67.7 s of export.
The 805 s of denoiser time is real and repeats; a resident batch process and its allocator
state add about 9%, not multiples.

`scripts\profile_sampling.py` on the same case (3 steps per stage, CUDA profiler) says where
those seconds sit. Kernel durations are trustworthy; wall time is not, because the profiler
serialises CPU-side work — the xatlas chart pass that takes 46 s alone took 217 s under the
trace and is what stretched that run to 378 s. Shares of GPU time, from
`scripts\analyze_trace.py`:

| Item | Share of GPU time |
| --- | --- |
| FlashAttention (`flash_fwd_kernel`, 1260 launches, 22.4 ms each) | 31.5% |
| flex_gemm sparse conv (`cutlass ... gemm_relu`) | 15.4% |
| pageable host-device memcpy (14,565 HtoD + 18,123 DtoH) | 10.7% |
| cumesh BVH (decode and bake) | 7.8% |
| dense GEMM (cublas/xmma) | 6.6% |
| ~400 other kernel names (elementwise, index, norm, copies) | 28% |

The shape of it matters more than the totals: all four sampling stages ran **91-99%
GPU-busy**, the decode 24-66%, and 322,079 kernels were launched. A dense forward is a
saturated GPU fed by about 2,400 launches per second, which rules out batching the
conditional and unconditional branches: stacking them leaves the FLOPs unchanged and there is
no idle GPU time in those stages to recover. The branches also cannot share one packed layout
as they stand — the unconditional branch zeroes `selected_view`, which collapses
`routed_full_only_flash` to one group of 4,101 keys instead of one per view.

Do not read absolute stage costs out of the reduced-step run: its dual-contouring remesh
collapsed to 133,651 vertices where the full run produced 4,704,292, so it is not the same
work scaled down. And the profiler's own report counts each kernel twice, once on its
operator and once as a kernel record, so its "summed device time" is about double the real
busy time — use the trace, not the printed total.

Where the host time goes, in the same trace, as counts of CUDA API calls per phase. Pinned
device-to-host copies are the scalar readbacks — a `nonzero` or `.item()` reading a length or a
value back to the host — and there are 14,316 of them, which is the 13,554 `nonzero` and 759
`.item()` below, each one a sync point of its own:

| Window | HtoD pageable | DtoH pageable | DtoH pinned | `cudaStreamSynchronize` | `cudaMalloc` |
| --- | --- | --- | --- | --- | --- |
| 0-30 s: conditioner, stage 1, shape-512 | 9,070 | 6,254 | 1,563 | 16,641 | 1,723 |
| 30-75 s: the two dense denoisers | 3,102 | 2,356 | 166 | 5,624 | 288 |
| 75-120 s: decode | 1,610 | 6,405 | 12,570 | 19,671 | 301 |
| 120-135 s: mesh post-process | 778 | 3,097 | 12 | 18 | 14,697 |
| 135 s onward: xatlas and bake | 5 | 11 | 5 | 21 | 32 |

The dense denoisers are not host-bound: 5,082 of their syncs are the texture stage's models
being staged into VRAM in one burst at 45-60 s, and the rest of the window is quiet. Every
sync in the run belongs to one of three operations — 27,659 to `aten::copy_` (`.to`, `.cpu`),
13,554 to `aten::nonzero`, 759 to `.item()` — and 15,300 of the 17,041 `cudaMalloc` calls come
from outside any torch operator, that is, from the third-party remeshing, BVH and
dual-contouring code, which allocates and frees per call. That accounts for about 6.3 s of
host API time in the post-process window alone. The pageable copies cost real device time
(5.12 s of HtoD, 4.44 s of DtoH); the 14,316 pinned readbacks cost 0.03 s of device time
between them and their whole cost is the host stall, which is why an op-mix table built on
device time understates them.

Thirteen thousand of those `nonzero` calls are the decode's im2col loop, and only that one was
worth changing. `conv_flex_gemm.sparse_conv3d_forward` resolved the no-neighbour sentinel with
boolean indexing (`flat[found]`, `im2col[found] = ...`), and each of those runs a `nonzero`,
which reads a length back to the host and stalls the device. It now gathers with the sentinel
clamped to row 0 and zeroes the missing taps with a multiply, so a chunk has no data-dependent
shape left. `scripts\bench_conv_im2col.py` times the loop against a copy of the replaced body on
a fixed input of a size the decode stage sees (150k voxels, 512 channels, 62 chunks): **93.7 and
92.6 ms per call with 124 `aten::nonzero`, against 76.9 ms with none**, so about 17% off the
loop. The values are identical: `scripts\check_conv_im2col.py` compares it against a float64
reference built straight from the neighbour map with integer-valued features and weights, so the
GEMM's accumulation order and the reference sum have to agree bit for bit — they do, on
multi-chunk inputs with 3,410 of 10,800 taps missing, in both the resident and the offloaded
output path.

The end-to-end effect on a dense case cannot be read from a re-run. Case05 on the changed code
took 1042 s against 1001 s, decoders 84.6 s against 65.7 s — but that run's dual-contouring
remesh produced 8,520,990 vertices and 38,021 clusters where the earlier one produced 4,704,292
and 4,213, so it decoded several times the geometry and the comparison says nothing about this
loop. That remesh variance, below, is larger than the change, which is why the loop is measured
directly rather than end to end.

The same reasoning removed the other per-call host work in the attention path:
`full_attn.py` rebuilt `cu_seqlens` with a cumsum, cat, cast and blocking upload on every
attention call, about 70 per forward, and now reads the layout's cached `cum_seqlen` and only
casts it. `scripts\check_attention_varlen.py` checks that against a per-batch reference with
unequal batch lengths, for both the self- and the cross-attention call shapes.

What the measurements ruled out, and what is left:

- **Attention (31.5% of GPU time) is honest arithmetic, not waste.** The self-attention runs
  at 43.5 TFLOPS and the full-bank cross-attention at 43.3, both about half the 3070 Ti's fp16
  tensor peak, on head_dim 128 with fp32 accumulation. The trace's `cu_seqlens` are length 2,
  so the 20 cross calls per forward are plain full-bank attention over all four views rather
  than the routed layout, and 41.3 ms is what that costs. Only the host-side rebuild above was
  removable.
- **The sparse convolutions have nothing cheaper to switch to.** No convolution runs twice per
  forward: CFG doubles only the flow transformers, which hold no sparse convolutions, and the
  three VAE decoders each run once. The installed wheel exposes CUDA hashmap and grid-sample
  kernels but no Triton ones, so `implicit_gemm`, `splitk`, `masked` and their sorted variants
  all raise `ImportError` and explicit GEMM is the only algorithm; there is no `force_fp16`,
  `fused` or `sorted` flag to set. Neighbour maps are cached per coordinates set as intended.
  The one uncached path is the `stream_from_cpu` branch, which rebuilds a host neighbour map
  per convolution for any level above the 256 MB budget; no case was confirmed to reach that
  level, so it is left as is.
- **The sampler is first-order Euler with exactly two evaluations per step under CFG**, and
  the repository contains no second-order solver, `eta`, or stochastic option. Step count is
  therefore the only lever, and it is a quality decision. Holding the 102.6 s model load and
  the 145.6 s of non-sampling work (DINO, stage-1 decode, shape-512 decode, decoders, export)
  fixed, case05 costs:

  | Steps per stage | Sampling | Stages + floor | End to end |
  | --- | --- | --- | --- |
  | 10 | 434 s | 580 s | 682 s |
  | 12 | 521 s | 667 s | 769 s |
  | 14 | 608 s | 753 s | 855 s |
  | 16 | 695 s | 841 s | 943 s |
  | 20 | 868 s | 1014 s | 1116 s |

  The config ships stage 1 at 12 steps and the other three stages at 20, which is the 1001 s of
  stage time measured on case05 in the table above. Taking the three heavy stages from 20 to 12
  steps is worth about 334 s and to 10 about 421 s. Texture and shape step counts are
  independent, but shape-512's step count sets the 1024-stage token counts through
  `surface_voxelize` of its decoded mesh, so it is not a free knob either. The `"name": "cfg"`
  key in the shape sampler blocks is dead — nothing reads it.
- **Export is dominated by xatlas, which is CPU-only.** Of the 67.7 s, 46 s is the chart pass
  over 4,213 clusters, and the GPU is idle for all of it: decode and export are strictly serial
  because the rasterizer needs the UVs. It is not idle CPU, though. `scripts\bench_uv_unwrap.py`
  times the same steps on synthetic meshes and reports wall against process CPU time:
  `compute_charts` measured 14.9 and 20.1 cores' worth of CPU on a million-face mesh, so the
  library already spreads over every core, and the cost tracks chart count rather than face
  count — 13 charts over a million faces parameterized in 0.6 s, while 499,849 two-face charts
  took 65 s, most of it `pack_charts` at 2.5 cores. There is no GPU version of xatlas and no
  drop-in replacement, so "unwrapping on the GPU" would mean a different parameterizer with a
  different UV layout, which is a project rather than a setting (surveyed 2026-10-16, see the
  deferred note at the end of this section). The repository never forwards
  xatlas's own chart parameters, so it runs on defaults, but it does control the GPU clustering
  that decides how many charts there are (`compute_charts_kwargs` at `texture_baking.py:219-224`),
  and that is the knob that would cut this time — at the cost of coarser charts and more texel
  crowding. The other way to reclaim it is not to shorten it but to hide it: the window is
  CPU-only, so in a batch its 46 s could carry the next case's GPU work, which for a single case
  has nothing to overlap with. Two things in that stage are provably wasted work but
  were left alone: `mesh.simplify(pre_simplify_target)` at `texture.py:127` has a target
  (16.78M faces) above the mesh it is given (15.96M), so it cannot decimate — but its result
  being identical is not provable from the code, only the target comparison is; and
  `to_glb` fills holes on a mesh the caller already filled.

Startup costs about half of what upstream spends: the cascade flow models skip the
`nn.Linear` initialisation a strict checkpoint load immediately overwrites, and the VGGT
conditioner parses its 3.90 GB checkpoint once instead of twice. Together those took the in
situ load from **143.3 s to 101.9 s**.

The pipeline is not run-to-run deterministic. One case decoded to 3,888,430 and 3,912,679
vertices and a 512-grid shell of 14,186,174 and 3,230,344 faces in two runs of identical
code. The load-time changes cannot be the cause — every parameter is overwritten by a strict
state dict load and no model they touch registers buffers — so the variation comes from the
CUDA meshing/dual-contouring kernels. Expect GLB size and the remesh counts to move between
runs; timings and VRAM stay inside the ranges above.

Left alone deliberately: the 0.95 GB shape decoder is instantiated twice because Shape-512
and Shape-1024 need different grid resolutions, and the texture stage repeats the Shape-1024
VGGT conditioner encode (about a second). The init skip is an explicit `skip_weight_init`
flag instead of `torch.nn.utils.skip_init`/`meta` construction because `__init__`-computed
buffers are absent from the checkpoints — `loaders.py` has to `setdefault` the dense stage's
`rope_phases` for exactly that reason, so a `to_empty` construction would inject
uninitialised RoPE. The flag skips parameters only, and every load it covers is `strict=True`.

### Host memory: staged loading and low-RAM mode

The stack is 25 GiB of checkpoints, and it used to be built in full before the first case ran:
101.9 s of wall clock before the first denoiser step, with every stage's weights then held for
the life of the process. Weights now load one group per stage, in the order a case needs them:

| Group | Models | Load | Peak commit |
| --- | --- | --- | --- |
| `preprocess` | DINOv3 encoder, RMBG-2.0 background remover | 1.0 s | 5.30 GB |
| `structure` | Stage-1 DiT + VGGT backbone, sparse-structure decoder | 19.1 s | 23.37 GB |
| `conditioner` | the VGGT conditioner Shape-512/1024 and texture share | 5.5 s | 12.14 GB |
| `shape512` | Shape-512 denoiser + decoder | 10.7 s | 14.60 GB |
| `shape1024` | Shape-1024 denoiser | 7.9 s | 14.60 GB |
| `texture` | texture denoiser, Shape-1024 and texture decoders | 12.6 s | 14.60 GB |

Each row is that group loaded on its own, with the checkpoints already in the OS file cache:
the load times come from `scripts\check_stage_loading.py --low-ram`, which walks the groups in
stage order, and `Peak commit` from one fresh process per group. `Peak commit` is that
process's committed-memory high-water mark, which starts from a **2.41 GB** baseline of
mapped libraries and CUDA context; the six loads cost about 57 s in sequence.

A run begins a stage as soon as that stage's own group is in memory, and the store's worker
thread fetches the rest while the GPU works. On case01 the first denoiser step started at
**21 s** instead of waiting for the whole stack (101.9 s in the profile above; 77.6 s in the
warm-cache run below), and no later stage waited: the Shape-512 group arrived one second
before its stage asked for it, and the texture group 55 s before. Stage times and VRAM are
unchanged — the same run reproduced the per-stage peaks in the table above.

| Loader, end to end | Inputs | Case time | Peak commit | Peak working set |
| --- | --- | --- | --- | --- |
| Pre-change (all groups loaded up front, kept) | case02's four views | 237.2 s | 49.7 GB | 35.3 GB |
| Staged, default (prefetched, all resident) | the same four views | 268.5 s | 42.0 GB | 26.7 GB |
| Staged, `--low-ram` (one group at a time, dropped after use) | case01's four views | 309.2 s | 23.1 GB | 14.0 GB |

The first two rows are the same four images in the same order, so they compare directly: the
staged loader peaks 7.7 GB lower committed and 8.6 GB lower resident, and reaches the GLB
sooner in wall clock as well, because the pre-change run spent 77.6 s loading before its case
began (314.8 s from launch against 268.5 s). Its case time is about 31 s longer, which is the
Stage-1 wait it can no longer hide — some 20 s of it on this case, the rest spread: the same
stage took 20.2 s and 47.9 s in those two runs, and the first stage-1 call in a process is
measurably slower and numerically different from later ones (last two subsections).

`--low-ram` drops each group as soon as the case is done with it — after DINO features, after
Stage 1, after Shape-512, after Shape-1024, after texture sampling — and never prefetches.
`scripts\check_stage_loading.py --low-ram` asserts at each of those points that nothing outside
the current stage is resident, and passes. `scripts\check_model_store.py` covers the store
itself — lazy build, prefetch order, release, and one build shared by concurrent callers —
with fake groups, so it runs in a second and loads no weights. Low-RAM costs about 68 s on
case01, the lighter case, which takes 241.2 s in default mode: its group loads are no longer
hidden behind the stage in front of them. A low-RAM job run through the web UI peaked at
**24.6 GB** committed and 16.5 GB resident, Gradio included.

The peak is set by *building* the model, not by holding it. Before it halves anything, the
loader has a model of fp32 parameters plus the fp32 checkpoint tensors that go into it. For
the `structure` group on its own, in a fresh process:

| Step | Committed |
| --- | --- |
| library imports and CUDA context | 2.41 GB |
| model constructed (`skip_weight_init`, so no random fill) | 15.32 GB |
| + the bf16 DiT base it reads (2.41 GB) | 17.74 GB |
| + the fp32 VGGT backbone (4.31 GB) | 22.06 GB |
| + the fp32 Stage-1 checkpoint (1.15 GB) | 23.37 GB |
| after halving to fp16 | 8.92 GB |

The constructor allocates 12.9 GB of fp32 parameters whatever the checkpoints hold, and the
checkpoints add their own 8.1 GB as they are read, so both copies exist at once. The group
ends up holding 4.91 GB of weights in mixed precision and 8.89 GB committed. A 16 GB machine
will page during that load in every residency mode: staged loading bounds what a run
*retains*, not this transient.

#### Half-precision checkpoints on disk

The same halving, done once on disk, removes the checkpoint half of that term and 10 GB of
disk. `scripts\convert_weights_fp16.py` (or the `convert_weights_fp16.bat` launcher) writes
fp16 copies of the eight fp32 checkpoints — the three denoisers, the VGGT backbone, the Stage-1
checkpoint and the three adapters — reads each one back and compares it tensor by tensor, and
then re-runs `make_local_config.py`, which owns `weights\pipeline.local.json` and now prefers
the copies. Every tool and launcher reads that one file, so nothing else has to change, and
`check_env.py` prints which set is in use.

```powershell
scripts\convert_weights_fp16.bat          # ~90 s: convert, verify, switch over
.\.venv\Scripts\python.exe scripts\check_fp16_weights.py   # the slower gate, ~5 min
scripts\revert_weights_fp16.bat           # back to the originals
```

| Group | Peak commit, fp32 files | Peak commit, fp16 files | Settled commit after the load |
| --- | --- | --- | --- |
| `structure` | 23.37 GB | 22.91 GB | 10.51 -> 8.89 GB |
| `shape512` | 14.60 GB | 14.36 GB | 10.02 -> 9.96 GB |

Most of the 90 s is building the four flow models to read off which tensors they halve; the
conversion itself is 13 s of reading and writing. The copies are not whole files: the pipeline
halves the Linear/Conv/Sparse weights of the flow models and keeps the VGGT attention
projections, the LoRA matrices, the normalisations and the LayerScale gammas in fp32, and
rounding those too would have changed values it deliberately keeps (up to 4.8e-4 relative, on
101M of the `structure` group's 2534M values). So the converter mirrors that split exactly,
which is what lets `check_fp16_weights.py` require every tensor of every group to match bit for
bit — it does: 7,225 tensors, 8.55B values, no difference. Use that check as the gate for any
change here.

Generation itself does not get faster: the models end up with the same dtypes and the same
values, so the GPU does the same work. What does change is how long a *load* takes, which
matters most in `--low-ram`, where every case reloads all six groups. Walking the groups in
stage order on this machine:

| Weights | dino | stage1 | shape512 | shape1024 | texture | total |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 files, run 1 | 1.3 s | 30.9 s | 31.0 s | 15.2 s | 14.3 s | 92.7 s |
| fp32 files, run 2 | 1.3 s | 30.4 s | 26.4 s | 15.5 s | 24.5 s | 98.1 s |
| fp16 copies | 1.2 s | 26.5 s | 23.4 s | 12.4 s | 18.2 s | 81.7 s |

About 15 s per low-RAM case, from reading half the checkpoint bytes; the two fp32 runs show
the spread this measurement carries. In default mode the loads are hidden behind the stage in
front, so the effect is on the first case's wait and on the resident commit, not on the case
time.

`weights\pipeline.local.json` now points at the copies, so a `--weights-dtype float32` or
`auto` request on top of it would run fp32 compute over fp16 values; the pipeline prints a
warning when that combination appears and points at `make_local_config.py --fp32`, which
writes the checkpoint-faithful config (the README's command-line section has the two lines
that use it).

#### Deleting the originals afterwards

The copies are what a run reads, so once `check_fp16_weights.py` has passed, the eight fp32
`*.pt` files can go: `weights/` drops from 40.87 GiB to 19.42 GiB. The manifest records where
each copy came from, and `half_precision_copies()` now treats a *missing* original as "the
copy is the source" rather than as a mismatch, so the config generator keeps the preference and
`weights\pipeline.local.json` stays valid.

Three things stop working until `scripts\fetch.py weights` brings the originals back (21.45 GiB
of download):

- the checkpoint-faithful `--weights-dtype float32` and `--no-low-vram` modes, which is what
  `pipeline.local.fp32.json` exists for;
- `check_fp16_weights.py`, which needs both sides to compare — it says
  `cannot verify: N of 8 fp32 originals are gone` and exits non-zero;
- `convert_weights_fp16.py --revert`, which refuses, because deleting the copies would leave the
  fp32 config naming files that do not exist.

#### Still open: building the denoisers in fp16

That is the 12.9 GB half of the peak, and no disk conversion can touch it. It means
constructing the mix-precision modules in the target dtype instead of converting them
afterwards, keeping the norms, LoRA and VGGT parameters fp32 exactly as they are now — the
same equivalence `check_fp16_weights.py` already tests. `skip_weight_init=True` means the
constructor does no random fill, which is what makes it plausible at all, but it changes the
construction path every mode shares. Designed, not written.

#### What the fp32 pins cost (measured 2026-10-17)

Those pins are 388 MiB of the structure model: the normalisations, the LayerScale gammas, the
VGGT raw attention projections and the LoRA matrices. They run fp32 compute, so it is worth
knowing what they cost. `scripts\bench_fp16_stage1.py` runs stage 1 on case01's images under
that policy and under one that halves every floating parameter and buffer of the flow models —
one arm per process, the same seeded noise, one warm-up call discarded:

| Arm | Call times | Peak VRAM | fp32 left | Voxels |
| --- | --- | --- | --- | --- |
| shipped (fp32 pins) | 20.0 s, 20.0 s | 5.15 GB | 388 MiB | 1568 |
| all fp16 | 19.7 s, 19.7 s | 4.95 GB | 0 MiB | 1567 |

About 1%, and 200 MB of VRAM, in exchange for results that are no longer bit-exact (1567
against 1568 voxels). The pins are not a speed problem, and the fp16 construction above is
about host memory rather than time: generation is as fast as it is for reasons that have
nothing to do with this change.

#### Host commit, not CUDA: what the access violations were

Two runs have died with `CUDA error: an illegal memory access` — the web UI's `--low-ram` mode
on 2026-10-16 (case02, sampling step 10 of 12) and the A/B harness on 2026-10-17 (a second
pipeline's structure model, step 1 of 12) — and one with `Stage1 decoded an empty sparse
structure`. On 2026-10-17 the same class of failure reproduced with a readable error instead of
an access violation, on a script that loads all six groups at once:

```
OSError: The paging file is too small for this operation to complete. (os error 1455)
```

The machine at that moment: 82.2 GB of commit limit (63.9 GB of RAM plus a page file set to
16 GB initial / 32 GB maximum), **65.7 GB already committed and 16.5 GB of headroom**, of which
the web UI server held 32.55 GB on its own and ComfyUI another 4 GB. A full `load_all()` asks for
two to three times what a run ever asks for, because a run holds one group at a time. Windows
reports an exhausted commit as an access violation in whatever allocates next, which is why the
earlier two looked like CUDA bugs in the VGGT attention: they were the host running out, not the
card.

`scripts/check_env.py --config` now builds the six groups one at a time and releases each, which
is what a run does anyway, and passes with the server still resident (commit headroom 23.2 GB).
`scripts\check_stage1_repeat.py` remains the harness for repeating the stage, but nothing points
at repeating stage 1 as the cause. If you run the UI and a CLI batch together, give the page file
more room: it is capped at 32 GB, and a resident server plus a case is already ~40 GB of commit.

Separately, and still open: the *first* stage-1 call in a process has been measured at 42.0 s
against 20.3 s for later calls with identical noise, and it produced a different latent (a 6.39
maximum difference over a ±4.6 range), so the stage-time spread quoted earlier is at least partly
warm-up rather than run-to-run noise.


### Deferred: unwrapping on the GPU (surveyed 2026-10-16)

Nothing adoptable exists today. Recorded so the same ground is not covered twice, and with the
conditions that would change the answer.

- **xatlas has no GPU port and is not getting one.** 61 GitHub repositories match `xatlas` and every
  one of them is a binding or a port — Python, C#, Rust, Swift, D, Odin, Mojo, WASM, three.js — and
  none moves chart computation off the CPU. Upstream's open issues contain no GPU work; the nearest
  is #125 "Optimisation for high-load input", open since 2023. The second half of
  `cumesh.uv_unwrap` therefore stays CPU.
- **That half is already multithreaded**, so a GPU port would not free CPU cores either (measurement
  above: 14.9 and 20.1 cores' worth of CPU during `compute_charts`). It would only make the work
  able to run at the same time as the GPU's own work.
- **GPU packing exists; packing is not the bottleneck.** UVPackmaster (commercial Blender/Maya
  addon, CUDA and Vulkan) and RizomUV (GPU packing since 2025) accelerate packing, not
  parameterization, and TABI (2026) is research packing. At the chart counts this pipeline produces,
  `pack_charts` is about 0.6 s of the 46 s. It only becomes material if chart counts explode the way
  the 2026-10-15 case05 run did — 38,021 clusters, where the synthetic equivalent spent 64 s in
  `pack_charts` at 2.5 cores.
- **GPU parameterization exists only as research code.** `Biped-Potato/pyrxmesh_parameterization`
  (MIT, pushed 2026-10-07, no stars) minimizes a Dirichlet energy by stochastic gradient descent on
  RXMesh; it takes an initial UV for large meshes and performs neither chart segmentation nor
  packing. SLIM (ETH IGL) remains the quality reference and its implementations are CPU. NVIDIA
  Kaolin has no UV or atlas operation at all.
- **There is no library that returns charts, UVs and a packed atlas in one GPU call**, so "unwrap on
  the GPU" means writing that library, and its UV layout would differ from xatlas's in seams and
  texel crowding — a quality decision, not a speed setting.

Revisit when one of these holds: chart counts reach the tens of thousands on real cases and packing
becomes the dominant term; a permissively licensed GPU parameterizer ships; or the export stage has
to fit a hard wall-clock budget. Until then the levers are the ones already listed — fewer charts, or
overlapping this CPU-only window with the next case in a batch.

## 6. Troubleshooting

- **`AttributeError: module 'flex_gemm.kernels' has no attribute 'triton'`** — the
  installed flex_gemm wheel has no Triton kernels. Keep
  `FLEX_GEMM_ALGO = 'explicit_gemm'` in `uniforge3d/modules/sparse/conv/config.py`;
  `python scripts\check_env.py` reports this combination before a model load.
- **CUDA OOM in a stage** — re-run with `--vram-debug` to see which stage peaks, and
  close other GPU users (browser hardware acceleration is a common one).
- **`CUDA error: an illegal memory access` in a batch run, or a stage that crawls with
  0 MB free** — the process has gone past the card and the driver is paging through system
  RAM (WDDM). `--vram-debug` names the stage, and a process `peak` above ~8 GB allocated
  is that regime. This is the failure the 512 remesh cap in section 4.8 exists to prevent.
- **`CUDA error: misaligned address` (or `an illegal memory access`) mid-case while there is
  still free VRAM** — a stochastic device-side fault, distinct from the paging failure above:
  in the one occurrence seen it hit a sampling attention kernel at step 14 of 20, right after
  a case that had been paging heavily, and the same case then completed clean on a re-run. A
  CUDA context cannot be trusted after a device-side fault, so restart the process: cases that
  already wrote `prediction.glb` are skipped, so a restart resumes where it stopped.
- **Power draw oscillates (for example 90-170 W) and the stage crawls** — the card is
  stalled, not throttled. `nvidia-smi -q -d PERFORMANCE` distinguishes them: if `SW Power
  Cap`, `HW Slowdown` and `HW Thermal Slowdown` are all `Not Active` while power swings, the
  GPU is waiting on memory. `nvidia-smi dmon -s pucm -d 1 -c 10` shows the shape of it —
  `sm` at 100% with `mem` low and `fb` at capacity means the SMs are occupied but faulting.
  Confirm the spill with `Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`: a non-zero
  value is host memory the driver handed back to itself. Fix the stage (section 4.8), and
  close other GPU clients if `Get-Counter '\GPU Process Memory(*)\Dedicated Usage'` shows
  them holding VRAM. `--vram-debug`'s `free at start` line gives the real budget up front.
- **`RuntimeError: The size of tensor a (128) must match the size of tensor b (0)` from a
  sparse residual** — a module read its input's features after a convolution released them
  to make room for an output above the `OFFLOAD_BYTES` budget. A caller that needs those
  features afterwards parks its own copy in host RAM before the call, as
  `SparseConvNeXtBlock3d` does for its residual. `python scripts\check_offload.py`
  reproduces the release, checks the parked residual against the resident one, and reports
  the device's free VRAM (about 30 s).
- **Startup time** — `python scripts\profile_load.py --config weights\pipeline.local.json`
  builds every model on the CPU and prints the per-model breakdown; it needs no GPU.
- **One case much slower than its neighbours** — cost tracks the shape's token count, not
  VRAM (section 5). `python scripts\profile_sampling.py assets\case05 --steps 3` runs a case
  with the CUDA profiler at a reduced step count and prints the operators that dominate
  device time, with their input shapes, plus a chrome trace for Perfetto (a few minutes).
  `python scripts\analyze_trace.py logs\sampling_trace.json` then reads that trace and reports
  real GPU busy time, the per-kernel-name split, the host copies and syncs by owning operator,
  and their distribution over time; the profiler's printed "summed device
  time" counts every kernel twice, and both the trace export and CPU-side stages such as
  xatlas are inflated, so use it for shares rather than for wall-clock costs.
- **After changing a sparse convolution or the sparse attention** — `python
  scripts\check_conv_im2col.py` compares a sparse convolution against a reference built from
  its neighbour map (bit-exact, integer-valued features, multi-chunk, both output paths) and
  `python scripts\check_attention_varlen.py` compares varlen attention against a per-batch
  reference with unequal batch lengths. Both take well under a minute; `check_offload.py`
  covers the residual/offload interaction. `python scripts\bench_conv_im2col.py` times the
  im2col loop itself against a copy of the replaced version and counts the `aten::nonzero`
  calls each makes, which is the only way to see this change: the decode stage's own timing
  moves by more than the change does between runs of identical code.
- **`ImportError: flash_attn`** — the sparse attention blocks require it; install
  `scripts/fetch.py wheels` output or re-run the installer without `-WithoutFlashAttn`.
- **Weights missing** — `python scripts\make_local_config.py --check` lists every file
  that is still absent.
- **Slow or stalled downloads** — always use `fetch.py`, never `huggingface-cli`;
  re-running resumes.
