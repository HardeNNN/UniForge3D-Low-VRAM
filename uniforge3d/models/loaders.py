"""Initialize models and load original training weights over their pretrained bases."""

import json
import re
from pathlib import Path
from typing import Any

import torch
from easydict import EasyDict as edict
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, load_model

from ..modules.utils import convert_model_to_dtype
from . import __getattr__ as get_model_class
from .conditioning import StructuredLatentConditioner
from .dino import DinoV3Encoder
from .lora import attach_stage1_lora
from .structured_latent_flow import VGGTConditionedSLatFlowModel
from .vaes.flexi_dual_grid import FlexiDualGridVaeDecoder
from .vaes.sparse_structure import SparseStructureDecoder
from .vaes.sparse_unet import SparseUnetVaeDecoder
from ..utils.preprocessing import BackgroundRemover


def resolve_checkpoint(
    checkpoint: str | Path | dict[str, str],
    root: Path,
    *,
    decoder: bool = False,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str:
    """Resolve a local checkpoint or an explicit Hub reference."""
    if isinstance(checkpoint, (str, Path)):
        return str(root / Path(checkpoint).expanduser())
    filename = checkpoint["filename"]
    options = {
        "repo_id": checkpoint["repo_id"],
        "revision": checkpoint.get("revision"),
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    if decoder:
        folder = snapshot_download(
            **options,
            allow_patterns=[f"{filename}.json", f"{filename}.safetensors"],
        )
        return str(Path(folder) / filename)
    return hf_hub_download(filename=filename, **options)


def _decoder_args(checkpoint: str | Path, expected_name: str) -> dict[str, Any]:
    """Read and validate the constructor arguments beside a decoder checkpoint."""
    config_path = Path(f"{checkpoint}.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["name"] != expected_name:
        raise ValueError(
            f"Expected {expected_name} in {config_path}, found {config['name']}"
        )
    return config["args"]

def _load_decoder_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Strictly load the Safetensors file beside a decoder configuration."""
    weights_path = Path(f"{checkpoint}.safetensors")
    load_model(model, weights_path, strict=True, device="cpu")
    model.eval().requires_grad_(False)

def load_shape_decoder(
    checkpoint: str | Path,
    resolution: int,
) -> FlexiDualGridVaeDecoder:
    """Construct the shape decoder and load its frozen inference weights."""
    model = FlexiDualGridVaeDecoder(
        **_decoder_args(checkpoint, "FlexiDualGridVaeDecoder")
    )
    _load_decoder_weights(model, checkpoint)
    model.set_resolution(int(resolution))
    return model

def load_sparse_structure_decoder(checkpoint: str | Path) -> SparseStructureDecoder:
    model = SparseStructureDecoder(**_decoder_args(checkpoint, "SparseStructureDecoder"))
    _load_decoder_weights(model, checkpoint)
    return model

def load_texture_decoder(checkpoint: str | Path) -> SparseUnetVaeDecoder:
    """Construct the PBR decoder and load its frozen inference weights."""
    model = SparseUnetVaeDecoder(
        **_decoder_args(checkpoint, "SparseUnetVaeDecoder")
    )
    _load_decoder_weights(model, checkpoint)
    return model

def _read_weights(checkpoint: str) -> dict[str, torch.Tensor]:
    if str(checkpoint).endswith(".safetensors"):
        return load_file(checkpoint, device="cpu")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    return state["model"] if "model" in state else state


def _load_trainable_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """A compact checkpoint must cover every trainable parameter over its loaded base."""
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = required - state.keys()
    if missing:
        raise RuntimeError(f"Missing training weights: {sorted(missing)[:20]}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected training weights: {result.unexpected_keys[:20]}")


def _load_structure_weights(model: torch.nn.Module, weights: dict) -> None:
    # Bases are loaded into FP32 modules, just as in the training constructor.
    dit = _read_weights(weights["dit_base"])
    # RoPE phases are deterministic buffers, absent from the official base file.
    dit.setdefault("rope_phases", model.dit.rope_phases)
    model.dit.load_state_dict(dit, strict=True)
    vggt = _read_weights(weights["vggt_base"])
    vggt = {name.removeprefix("model."): value for name, value in vggt.items()}
    # The published model omits the unused camera head and mask token.
    vggt = {
        name: value for name, value in vggt.items()
        if not name.startswith("camera_head.") and name != "aggregator.patch_embed.mask_token"
    }
    model.vggt.load_state_dict(vggt, strict=True)
    attach_stage1_lora(model.dit, **weights["dit_lora"])
    attach_stage1_lora(model.vggt, **weights["vggt_lora"])

    state = _read_weights(weights["checkpoint"])
    state = {
        re.sub(r"^fusion_groups\.(\d+)\.f_from_z\.", r"dit_to_vggt.\1.", name): value
        for name, value in state.items()
        if not re.match(r"^fusion_groups\.\d+\.z_from_f\.", name)
    }
    _load_trainable_weights(model, state)
    model.eval().requires_grad_(False)


def _load_flow_model(spec: edict, weights: dict):
    if spec.name != "VGGTConditionedSLatFlowModel":
        raise ValueError(f"Unsupported flow model: {spec.name}")
    # Both checkpoints below are loaded strictly, so randomising several gigabytes of
    # denoiser weights first would be discarded work.
    model = VGGTConditionedSLatFlowModel(**spec.args, skip_weight_init=True)
    from peft import LoraConfig, get_peft_model

    if "base_checkpoint" in weights:
        model.denoiser.load_state_dict(_read_weights(weights["base_checkpoint"]), strict=True)
    model.denoiser.requires_grad_(False)
    model.denoiser = get_peft_model(model.denoiser, LoraConfig(**weights["lora"]))
    state = _read_weights(weights["checkpoint"])
    if "base_checkpoint" in weights:
        _load_trainable_weights(model.denoiser, state)
    else:
        model.denoiser.load_state_dict(state, strict=True)
    model.adapter.load_state_dict(_read_weights(weights["adapter_checkpoint"]), strict=True)
    model.eval().requires_grad_(False)
    return model


def _apply_weights_dtype(models: dict[str, Any], weights_dtype: torch.dtype) -> None:
    """Halve the memory-dominant weights of the large denoising models.

    The flow models run under autocast, so converting their Linear/Conv/Sparse weights
    and the ``dtype`` attribute they feed to ``manual_cast`` is enough. The VAE decoders
    instead cast their activations themselves and are built from the precision their own
    checkpoint declares, so only an fp16 request changes them - and then only their
    torso, which is the extent of their internal cast boundary. DINO and the VGGT
    conditioner keep the precision they were published with.
    """
    for name in ("structure_model", "shape512_model", "shape1024_model", "texture_model"):
        if name in models:
            convert_model_to_dtype(models[name], weights_dtype)
    if weights_dtype != torch.float16:
        return
    for name in ("structure_decoder", "shape512_decoder", "shape1024_decoder", "texture_decoder"):
        if name not in models:
            continue
        decoder = models[name]
        decoder.convert_to_fp16()
        # convert_to_fp16 only halves the torso; the decoder reads `dtype` to cast the
        # activations it feeds into that torso, so the two have to agree.
        decoder.dtype = weights_dtype


# The groups in the order the pipeline first needs them: preprocessing (background
# remover and DINO), sparse structure, the VGGT conditioner that all three latent stages
# share, then Shape-512, Shape-1024 and texture.
MODEL_GROUPS: tuple[str, ...] = (
    "preprocess",
    "structure",
    "conditioner",
    "shape512",
    "shape1024",
    "texture",
)


def load_model_group(
    name: str, config: dict[str, Any], *, device: torch.device,
    cache_dir: str | Path | None = None, local_files_only: bool = False,
    weights_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """The models one stage needs and nothing else, in the order above.

    Loading a group at a time is what lets the pipeline start working before the whole
    25 GiB stack is in memory. ``device`` is only used by the background remover, which
    binds the device its forward pass moves its input to: every model is built on CPU.
    """
    structure = edict(config["sparse_structure"])
    shape512 = edict(config["structured_latent_512"])
    shape1024 = edict(config["structured_latent_1024"])
    texture = edict(config["texture"])
    download = {"cache_dir": cache_dir, "local_files_only": local_files_only}

    if name == "preprocess":
        models = {
            "dino": DinoV3Encoder(config["external_models"]["dino"], **download)
            .eval().requires_grad_(False),
            "background_remover": BackgroundRemover(
                config["image_preprocess"]["rembg_model"], device, **download
            ),
        }
    elif name == "structure":
        spec = structure.model
        structure_model = get_model_class(spec.name)(**spec.args)
        _load_structure_weights(structure_model, structure.weights)
        models = {
            "structure_model": structure_model,
            "structure_decoder": load_sparse_structure_decoder(structure.decoder.checkpoint),
        }
    elif name == "conditioner":
        models = {"conditioner": StructuredLatentConditioner(**config["conditioner"])}
    elif name == "shape512":
        models = {
            "shape512_model": _load_flow_model(shape512.model, shape512.weights),
            "shape512_decoder": load_shape_decoder(
                shape512.decoder.checkpoint, int(shape512.decoder.resolution),
            ),
        }
    elif name == "shape1024":
        models = {"shape1024_model": _load_flow_model(shape1024.model, shape1024.weights)}
    elif name == "texture":
        models = {
            "texture_model": _load_flow_model(texture.model, texture.weights),
            "shape1024_decoder": load_shape_decoder(
                texture.shape_decoder_checkpoint, int(texture.resolution)
            ),
            "texture_decoder": load_texture_decoder(texture.texture_decoder_checkpoint),
        }
    else:
        raise ValueError(f"Unknown model group: {name!r}")

    if weights_dtype is not None:
        _apply_weights_dtype(models, weights_dtype)
    return models
