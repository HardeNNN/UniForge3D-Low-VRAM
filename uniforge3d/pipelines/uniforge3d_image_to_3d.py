

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import trimesh
from huggingface_hub import snapshot_download

from ..models.loaders import MODEL_GROUPS, load_model_group, resolve_checkpoint
from ..models.store import StageModelStore
from ..modules.utils import str_to_dtype
from ..utils.device import release_cuda
from ..utils.mesh import (
    display_to_model,
    export_glb,
    normalize_to_cube,
    official_shape_only_postprocess,
    surface_voxelize,
)
from ..utils.preprocessing import BackgroundRemover, InputBatch, build_input_batch
from ..utils.texture_baking import save_texture_maps
from .shape import decode_shape_latent, sample_shape_latent
from .sparse_structure import decode_sparse_structure, decoded_coordinates, sample_sparse_structure
from .texture import (
    decode_shape_and_texture,
    denormalize_shape_latent,
    remesh_and_bake_texture,
    sample_texture,
)
from .types import NormalizedShapeLatent, SparseStructure


def _resolve_weights_dtype(weights_dtype, low_vram: bool) -> torch.dtype | None:
    """Resolve the inference weight dtype; low-VRAM defaults to fp16 weights.

    ``None`` keeps every checkpoint in the precision it was published with.
    """
    if weights_dtype is None:
        return torch.float16 if low_vram else None
    if isinstance(weights_dtype, torch.dtype):
        return weights_dtype
    return str_to_dtype(str(weights_dtype))


class UniForge3DImageTo3DPipeline:


    def __init__(
        self,
        config: dict,
        *,
        device: int = 0,
        low_vram: bool = True,
        weights_dtype: torch.dtype | str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        low_ram: bool = False,
        prefetch: bool = True,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda", int(device))
        self.low_vram = low_vram
        self.low_ram = low_ram
        self.weights_dtype = _resolve_weights_dtype(weights_dtype, low_vram)
        # The local config prefers the half-precision copies when they exist, and those hold
        # fp16 values wherever the pipeline halves. Asking for fp32 or `auto` on top of them
        # would quietly give up the precision those requests exist to keep.
        if config.get("half_precision_copies") and self.weights_dtype not in (
            torch.float16, torch.bfloat16,
        ):
            precision = (
                "the published precision"
                if self.weights_dtype is None
                else str(self.weights_dtype).removeprefix("torch.")
            )
            print(
                "[weights] this config points at the half-precision copies, which cannot "
                "raise their precision again: the model runs at "
                f"{precision} over fp16 values. Use a config built with "
                "`make_local_config.py --fp32` for checkpoint-faithful weights.",
                flush=True,
            )
        # Flow models run under autocast, whose dtype is what actually selects the compute
        # precision. Autocast only accepts fp16/bf16, so an fp32 weight request keeps the
        # upstream bf16 autocast and lets it cast per op. The VAE decoders manage their
        # own dtype internally.
        self.compute_dtype = (
            self.weights_dtype
            if self.weights_dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )
        self.vram_debug = os.environ.get("UNIFORGE_VRAM_DEBUG", "") not in ("", "0", "false")
        self._stage_started = time.perf_counter()

        # Weights arrive a group at a time: either when a stage first asks for them, or
        # ahead of time on the store's worker thread while the GPU runs the stage before.
        # Low-RAM mode never prefetches, because holding the next stage's weights is the
        # memory it exists to avoid.
        self._models = StageModelStore(
            {
                name: partial(
                    load_model_group, name, config, device=self.device,
                    cache_dir=cache_dir, local_files_only=local_files_only,
                    weights_dtype=self.weights_dtype,
                )
                for name in MODEL_GROUPS
            },
            order=MODEL_GROUPS,
            prefetch=prefetch and not low_ram,
        )

    # Every stage's weights live in the store until the stage that needs them runs, so
    # these are accessors rather than finished loads: the first use in a case builds the
    # group (a few seconds to half a minute) and every later use is a dictionary lookup.
    # The group each model belongs to is the one named in its docstring.

    @property
    def remove_background(self) -> BackgroundRemover:
        """Background remover, part of the ``preprocess`` group."""
        return self._models.get("preprocess")["background_remover"]

    @property
    def dino(self):
        """DINOv3 encoder, part of the ``preprocess`` group."""
        return self._models.get("preprocess")["dino"]

    @property
    def structure_model(self):
        """Sparse-structure flow model, part of the ``structure`` group."""
        return self._models.get("structure")["structure_model"]

    @property
    def structure_decoder(self):
        """Sparse-structure VAE decoder, part of the ``structure`` group."""
        return self._models.get("structure")["structure_decoder"]

    @property
    def conditioner(self):
        """VGGT conditioner shared by both shape stages and the texture stage."""
        return self._models.get("conditioner")["conditioner"]

    @property
    def shape512_model(self):
        """Shape-512 flow model, part of the ``shape512`` group."""
        return self._models.get("shape512")["shape512_model"]

    @property
    def shape512_decoder(self):
        """Shape-512 VAE decoder, part of the ``shape512`` group."""
        return self._models.get("shape512")["shape512_decoder"]

    @property
    def shape1024_model(self):
        """Shape-1024 flow model, part of the ``shape1024`` group."""
        return self._models.get("shape1024")["shape1024_model"]

    @property
    def shape1024_decoder(self):
        """Shape-1024 VAE decoder, loaded with the ``texture`` group."""
        return self._models.get("texture")["shape1024_decoder"]

    @property
    def texture_model(self):
        """Texture flow model, part of the ``texture`` group."""
        return self._models.get("texture")["texture_model"]

    @property
    def texture_decoder(self):
        """PBR VAE decoder, part of the ``texture`` group."""
        return self._models.get("texture")["texture_decoder"]

    def _release(self, *groups: str) -> None:
        """Drop the weights of stages a case is done with (low-RAM mode only)."""
        if not self.low_ram:
            return
        for group in groups:
            self._models.release(group)

    def load_all(self) -> dict[str, torch.nn.Module]:
        """Build every stage's weights now and return them by name.

        A case pays for its groups as it reaches them, which is what keeps the first
        result early. This is for the callers that want the opposite: fail fast on a
        checkpoint that cannot be read, or know the resident parameter count.
        """
        models: dict[str, torch.nn.Module] = {}
        for group in MODEL_GROUPS:
            models.update(self._models.get(group))
        return models

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: int = 0, low_vram: bool = True,
        weights_dtype: torch.dtype | str | None = None,
        revision: str | None = None, cache_dir: str | Path | None = None,
        local_files_only: bool = False, low_ram: bool = False, prefetch: bool = True,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a local model directory or a Hugging Face repository."""
        root = Path(path).expanduser()
        if not root.is_dir():
            root = Path(snapshot_download(
                str(path), revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["pipeline.json", "uniforge3d/**/*.pt"],
            ))
        return cls.from_config(
            root / "pipeline.json", weights_root=root, device=device,
            low_vram=low_vram, weights_dtype=weights_dtype, cache_dir=cache_dir,
            local_files_only=local_files_only, low_ram=low_ram, prefetch=prefetch,
        )

    @classmethod
    def from_config(
        cls, config_path: str | Path, *, device: int = 0, low_vram: bool = True,
        weights_dtype: torch.dtype | str | None = None,
        weights_root: str | Path | None = None,
        cache_dir: str | Path | None = None, local_files_only: bool = False,
        low_ram: bool = False, prefetch: bool = True,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a JSON config; local weights default to the repository's weights/."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        root = (Path(weights_root).expanduser() if weights_root is not None
                else Path(__file__).resolve().parents[2] / "weights")
        options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
        for stage in ("sparse_structure", "structured_latent_512", "structured_latent_1024", "texture"):
            weights = config[stage]["weights"]
            for key in ("checkpoint", "adapter_checkpoint", "dit_base", "vggt_base", "base_checkpoint"):
                if key in weights:
                    weights[key] = resolve_checkpoint(weights[key], root, **options)
            if "decoder" in config[stage]:
                decoder = config[stage]["decoder"]
                decoder["checkpoint"] = resolve_checkpoint(
                    decoder["checkpoint"], root, decoder=True, **options,
                )
        for key in ("shape_decoder_checkpoint", "texture_decoder_checkpoint"):
            config["texture"][key] = resolve_checkpoint(
                config["texture"][key], root, decoder=True, **options,
            )
        vggt = config["conditioner"]["vggt_feature_extractor"]
        vggt["checkpoint_path"] = resolve_checkpoint(vggt["checkpoint_path"], root, **options)
        return cls(
            config, device=device, low_vram=low_vram, weights_dtype=weights_dtype,
            low_ram=low_ram, prefetch=prefetch, **options
        )

    @contextmanager
    def _use_models(self, *models):

        try:
            for model in models:
                model.to(self.device)
            yield
        finally:
            if self.low_vram:
                for model in models:
                    model.to("cpu")
                release_cuda()

    def _autocast(self):
        """Autocast context matching the resident weight precision."""
        return torch.autocast(device_type=self.device.type, dtype=self.compute_dtype)

    def _report_vram(self, label: str) -> None:
        """Print allocator statistics and the stage's elapsed time when debugging."""
        if not self.vram_debug or not torch.cuda.is_available():
            return
        now = time.perf_counter()
        elapsed, self._stage_started = now - self._stage_started, now
        free, total = torch.cuda.mem_get_info(self.device)
        print(
            f"[vram] {label:<30} "
            f"{elapsed:6.1f}s "
            f"allocated {torch.cuda.memory_allocated(self.device) / 2**30:5.2f} GB "
            f"reserved {torch.cuda.memory_reserved(self.device) / 2**30:5.2f} GB "
            f"peak {torch.cuda.max_memory_allocated(self.device) / 2**30:5.2f} GB "
            f"free {free / 2**30:5.2f}/{total / 2**30:5.2f} GB",
            flush=True,
        )

    def _preprocess_images(self, images: Sequence[str | Path]) -> InputBatch:

        with ExitStack() as stack:
            background_loaded = False

            def remove_background(image):
                nonlocal background_loaded
                if not background_loaded:
                    stack.enter_context(self._use_models(self.remove_background.model))
                    background_loaded = True
                return self.remove_background(image)

            return build_input_batch(
                images,
                self.config["image_preprocess"],
                remove_background,
            )

    @torch.no_grad()
    def _encode_dino(self, batch: InputBatch) -> dict[str, torch.Tensor]:
        """Encode once per resolution and share 512px features across stages."""
        with self._use_models(self.dino):
            images512 = batch.dino_images512.to(self.device, dtype=torch.float32)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                structure_features = self.dino(images512).cpu()
                shape1024_features = self.dino(
                    batch.dino_images1024.to(self.device, dtype=torch.float32)
                ).cpu()
        return {
            "sparse_structure": structure_features,
            "shape512": structure_features,
            "shape1024": shape1024_features,
        }

    def _generate_structure(
        self,
        batch: InputBatch,
        dino_features: torch.Tensor,
    ) -> SparseStructure:
        sampling = self.config["sparse_structure"]["sampler"]
        dit = self.structure_model.dit
        resolution = dit.resolution
        noise = torch.randn(
            (1, dit.in_channels, resolution, resolution, resolution),
            device="cpu",
        ).to(self.device, self.compute_dtype)
        with self._use_models(self.structure_model):
            vggt_images = batch.vggt_images.to(self.device)
            with self._autocast():
                latent = sample_sparse_structure(
                    self.structure_model,
                    noise,
                    vggt_images,
                    dino_features.to(self.device),
                    **sampling["args"],
                    **sampling["params"],
                )
        self._report_vram("stage1 dit")
        with self._use_models(self.structure_decoder):
            occupancy_logits = decode_sparse_structure(
                self.structure_decoder,
                latent,
                self.config["sparse_structure"]["decoder"].get("normalization"),
            )
        coords32, coords64 = decoded_coordinates(occupancy_logits)
        return SparseStructure(coords32.cpu(), coords64.cpu())

    def _sample_shape(
        self,
        model: torch.nn.Module,
        config: dict,
        batch: InputBatch,
        *,
        latent_coords: torch.Tensor,
        structure_conditioned_coords: torch.Tensor,
        dino_features: torch.Tensor,
        return_view_assignment: bool,
    ):
        latent_coords = latent_coords.int().contiguous()
        structure_conditioned_coords = structure_conditioned_coords.int().contiguous()
        with self._use_models(model, self.conditioner):
            conditioning = config["conditioning"]
            latent_coords = latent_coords.to(device=self.device, dtype=torch.int32)
            condition = self.conditioner.encode(
                model.adapter,
                dino_features=dino_features.to(device=self.device, dtype=torch.float32),
                vggt_images=batch.vggt_images.to(device=self.device, dtype=torch.float32),
                structure_conditioned_coords=structure_conditioned_coords,
                input_masks=batch.input_masks.to(device=self.device, dtype=torch.bool),
                latent_coords=latent_coords,
                router_grid=int(conditioning["grid_resolution"]),
            )
            latent = sample_shape_latent(
                model,
                latent_coords,
                condition,
                config["sampler"],
                description=f"Shape-{config['decoder']['resolution']} sampling",
                dtype=self.compute_dtype,
            )
            view_assignment = (
                {"selected_view": condition["cond"]["selected_view"].cpu()}
                if return_view_assignment else None
            )
        if not torch.isfinite(latent.feats).all():
            raise RuntimeError("Shape sampling produced non-finite latent features")
        self._report_vram(f"shape{config['decoder']['resolution']} denoiser")
        return latent, view_assignment

    def _generate_shape512(
        self,
        batch: InputBatch,
        structure: SparseStructure,
        dino_features: torch.Tensor,
    ) -> trimesh.Trimesh:
        latent, _ = self._sample_shape(
            self.shape512_model,
            self.config["structured_latent_512"],
            batch,
            latent_coords=structure.coords32,
            structure_conditioned_coords=structure.coords64,
            dino_features=dino_features,
            return_view_assignment=False,
        )
        with self._use_models(self.shape512_decoder):
            mesh = decode_shape_latent(
                self.shape512_decoder,
                latent,
                self.config["structured_latent_512"]["decoder"].get("normalization"),
            )[0]
        return official_shape_only_postprocess(
            mesh.vertices,
            mesh.faces,
            **self.config["mesh_postprocess"],
        )

    def _generate_shape1024(
        self,
        batch: InputBatch,
        coords64: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> tuple[NormalizedShapeLatent, dict[str, torch.Tensor]]:
        latent, view_assignment = self._sample_shape(
            self.shape1024_model,
            self.config["structured_latent_1024"],
            batch,
            latent_coords=coords64,
            structure_conditioned_coords=coords64,
            dino_features=dino_features,
            return_view_assignment=True,
        )
        if view_assignment is None:
            raise RuntimeError("Shape1024 must provide its texture routing result")
        shape = NormalizedShapeLatent(coords=latent.coords.cpu(), feats=latent.feats.cpu())
        return shape, view_assignment

    def _shape512_to_coords64(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Convert the already-remeshed Shape-512 surface to coords64."""
        config = self.config["structured_latent_handoff"]
        model_vertices = display_to_model(mesh.vertices)
        normalized_vertices, _ = normalize_to_cube(model_vertices, config["target_max_side"])
        xyz = torch.from_numpy(
            surface_voxelize(normalized_vertices, mesh.faces, config["surface_voxel_resolution"])
        )
        return torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz.int()), dim=1)

    def _generate_texture(
        self,
        batch: InputBatch,
        normalized_shape_latent: NormalizedShapeLatent,
        view_assignment: dict[str, torch.Tensor],
        dino_features: torch.Tensor,
    ) -> trimesh.Trimesh:
        config = self.config["texture"]
        # Sampling and decoding are separate residencies: the texture flow model and
        # the VGGT conditioner (~5.5GB in fp16) are released before the two decoders
        # run, which is what makes the final stage fit alongside its activations.
        with self._use_models(self.texture_model, self.conditioner):
            shape_vae_latent = denormalize_shape_latent(
                config,
                normalized_shape_latent,
                self.device,
            )
            texture_vae_latent = sample_texture(
                self.texture_model,
                config,
                self.conditioner,
                shape_vae_latent,
                batch,
                dino_features,
                view_assignment,
                self.device,
                dtype=self.compute_dtype,
            )
        self._report_vram("texture denoiser")
        with self._use_models(self.shape1024_decoder, self.texture_decoder):
            reference_mesh, pbr_voxel = decode_shape_and_texture(
                self.shape1024_decoder, self.texture_decoder, shape_vae_latent, texture_vae_latent
            )
            del shape_vae_latent, texture_vae_latent
        self._report_vram("texture decoders")
        return remesh_and_bake_texture(
            config,
            self.config["mesh_postprocess"],
            reference_mesh,
            pbr_voxel,
        )

    @torch.no_grad()
    def run(
        self,
        images: Sequence[str | Path],
        output: str | Path,
        *,
        seed: int | None = None,
        overwrite: bool = False,
        save_textures: bool = False,
    ) -> Path:

        output_path = Path(output).expanduser().resolve()
        if output_path.exists() and (not overwrite or not output_path.is_file()):
            raise FileExistsError(
                f"Output already exists: {output_path}; use --overwrite to recompute"
            )
        seed = self.config["default_seed"] if seed is None else seed

        batch = self._preprocess_images(images)

        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        self._stage_started = time.perf_counter()
        if self.vram_debug:
            torch.cuda.reset_peak_memory_stats(self.device)
            print(
                f"[vram] low_vram={self.low_vram} weights={self.weights_dtype} "
                f"compute={self.compute_dtype} "
                f"total {torch.cuda.get_device_properties(self.device).total_memory / 2**30:.2f} GB",
                flush=True,
            )
            # Everything a stage allocates has to fit in what is left after the desktop,
            # browser and other GPU clients take their share.
            free, total = torch.cuda.mem_get_info(self.device)
            print(
                f"[vram] free at start {free / 2**30:.2f}/{total / 2**30:.2f} GB "
                f"({(total - free) / 2**30:.2f} GB held by other processes or the driver)",
                flush=True,
            )
        try:
            # Share one DINO model while preserving each stage's original precision.
            dino_features = self._encode_dino(batch)
            self._release("preprocess")
            self._report_vram("dino features")

            sparse_structure = self._generate_structure(batch, dino_features["sparse_structure"])
            self._release("structure")
            self._report_vram("stage1 decode")

            shape512_mesh = self._generate_shape512(
                batch, sparse_structure, dino_features["shape512"]
            )
            self._release("shape512")
            self._report_vram("stage2 shape512 decode")


            structure_conditioned_coords = self._shape512_to_coords64(shape512_mesh)
            shape1024_latent, view_assignment = self._generate_shape1024(
                batch,
                structure_conditioned_coords,
                dino_features["shape1024"],
            )
            self._release("shape1024")

            # Texture reuses the Shape1024 latent and its per-token view routing.
            textured_mesh = self._generate_texture(
                batch,
                shape1024_latent,
                view_assignment,
                dino_features["shape1024"],
            )
            self._release("texture", "conditioner")
            self._report_vram("texture export")
            published = export_glb(textured_mesh, output_path, overwrite=overwrite)
            if save_textures:
                for path in save_texture_maps(textured_mesh, published.parent / "textures"):
                    print(f"[textures] wrote {path}", flush=True)
            return published
        finally:
            release_cuda()
