"""Frozen VGGT feature extraction and four-layer DINO residual adapter."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniforge3d.models.vggt.models.aggregator import Aggregator

FUSION_LAYERS = (4, 7, 11, 14, 17, 20, 23)

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def load_prefixed_state(module: nn.Module, state: Mapping, prefix: str, label: str) -> None:
    """Strictly load one named submodule from a composite VGGT checkpoint."""
    module_state = OrderedDict(
        (str(key)[len(prefix) :], value) for key, value in state.items() if str(key).startswith(prefix)
    )
    expected, actual = set(module.state_dict()), set(module_state)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint mismatch: missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    module.load_state_dict(module_state, strict=True)


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShapeCoordEncoder(nn.Module):
    """Exact coords64 encoder used to train the source ShapeConditionedVGGT."""

    def __init__(self, grid_resolution: int = 64) -> None:
        super().__init__()
        self.grid_resolution = int(grid_resolution)
        self.coord_mlp = TwoLayerMLP(63, 512, 1024)
        self.token_mlp = TwoLayerMLP(1024, 1024, 1024)

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        xyz = (coords.float() + 0.5) / float(self.grid_resolution) - 0.5
        freqs = 2.0 ** torch.arange(10, device=xyz.device, dtype=xyz.dtype)
        phase = 2.0 * math.pi * xyz[..., None] * freqs
        return torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f"Expected packed coords64 [sumN,3], got {tuple(coords.shape)}")
        return self.token_mlp(self.coord_mlp(self.positional_encoding(coords)))


class PackedCrossAttention(nn.Module):
    """Exact ragged shape-token cross-attention used by the source checkpoint."""

    def __init__(self, q_dim: int, kv_dim: int, attn_dim: int, heads: int) -> None:
        super().__init__()
        if attn_dim % heads:
            raise ValueError("attn_dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = int(attn_dim) // self.heads
        self.q_norm = nn.LayerNorm(q_dim)
        self.k_norm = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(q_dim, attn_dim)
        self.to_k = nn.Linear(kv_dim, attn_dim)
        self.to_v = nn.Linear(kv_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, q_dim)

    def _attention(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        num_query, num_key = query.shape[0], key_value.shape[0]
        query_heads = self.to_q(self.q_norm(query)).view(num_query, self.heads, self.head_dim).transpose(0, 1)
        key_heads = self.to_k(self.k_norm(key_value)).view(num_key, self.heads, self.head_dim).transpose(0, 1)
        value_heads = self.to_v(key_value).view(num_key, self.heads, self.head_dim).transpose(0, 1)
        output = F.scaled_dot_product_attention(
            query_heads[None], key_heads[None], value_heads[None], dropout_p=0.0
        )
        return output[0].transpose(0, 1).reshape(num_query, -1)

    def forward(
        self,
        query: torch.Tensor,
        shape_tokens: torch.Tensor,
        shape_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or query.shape[0] + 1 != shape_offsets.numel():
            raise ValueError("Query batch and packed shape_offsets disagree")
        chunks = []
        for batch_idx in range(query.shape[0]):
            lo = int(shape_offsets[batch_idx])
            hi = int(shape_offsets[batch_idx + 1])
            if hi <= lo:
                raise ValueError(f"Sample {batch_idx} has no coords64 condition tokens")
            residual = self.out_proj(self._attention(query[batch_idx], shape_tokens[lo:hi]))
            chunks.append(query[batch_idx] + residual)
        return torch.stack(chunks, dim=0)


def _load_checkpoint_payload(path: str):
    kwargs = {
        "map_location": "cpu",
        "weights_only": True,
    }
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _unwrap_model_state(payload) -> Tuple[Mapping[str, torch.Tensor], Dict[str, object]]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping checkpoint, got {type(payload)}")
    if "model" in payload:
        state = payload["model"]
        metadata = {key: value for key, value in payload.items() if key != "model"}
    else:
        state = payload
        metadata = {}
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected checkpoint model state to be a mapping, got {type(state)}")
    return state, metadata


class FrozenVGGTFeatureExtractor:
    """Frozen coords64-conditioned VGGT backbone outside optimizer/DDP/checkpoints."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        feature_layers: Sequence[int] = (4, 11, 17, 23),
        shape_grid_resolution: int = 64,
        conditioning_layers: Sequence[int] = FUSION_LAYERS,
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(f"VGGT checkpoint not found: {self.checkpoint_path}")

        self.feature_layers = tuple(int(layer) for layer in feature_layers)
        if self.feature_layers != (4, 11, 17, 23):
            raise ValueError(
                f"Frozen VGGT point-head features must use layers (4, 11, 17, 23), got {self.feature_layers}"
            )
        dtype_key = str(dtype).lower()
        if dtype_key not in _DTYPES:
            raise ValueError(f"Unsupported frozen VGGT dtype {dtype!r}; choose from {sorted(_DTYPES)}")
        self.dtype = _DTYPES[dtype_key]
        self.device = torch.device(device)
        self.shape_grid_resolution = int(shape_grid_resolution)
        if self.shape_grid_resolution != 64:
            raise ValueError(
                "The source ShapeConditionedVGGT checkpoint requires coords64, "
                f"got grid resolution {self.shape_grid_resolution}"
            )
        self.conditioning_layers = tuple(int(layer) for layer in conditioning_layers)
        if self.conditioning_layers != FUSION_LAYERS:
            raise ValueError(
                f"Conditioning layers must match source training {FUSION_LAYERS}, "
                f"got {self.conditioning_layers}"
            )

        aggregator = Aggregator(cached_layer_indices=self.feature_layers)
        shape_encoder = ShapeCoordEncoder(grid_resolution=self.shape_grid_resolution)
        backbone_shape_adapters = nn.ModuleDict(
            {
                str(layer_idx): PackedCrossAttention(1024, 1024, 1024, 16)
                for layer_idx in self.conditioning_layers
            }
        )
        payload = _load_checkpoint_payload(self.checkpoint_path)
        state, metadata = _unwrap_model_state(payload)

        load_prefixed_state(aggregator, state, "aggregator.", "VGGT aggregator")
        load_prefixed_state(shape_encoder, state, "shape_encoder.", "ShapeCoordEncoder")
        load_prefixed_state(
            backbone_shape_adapters,
            state,
            "backbone_shape_adapters.",
            "VGGT backbone shape adapters",
        )
        checkpoint_grid_resolution = metadata.get("shape_grid_resolution")
        if checkpoint_grid_resolution is not None and int(checkpoint_grid_resolution) != 64:
            raise ValueError(f"Checkpoint shape_grid_resolution must be 64, got {checkpoint_grid_resolution}")

        # Extra checkpoint tensors of a subclass are taken here, while the archive is
        # already open, instead of decoding the multi-gigabyte file a second time.
        self._load_shared_checkpoint(state)
        del state, payload
        for module in (aggregator, shape_encoder, backbone_shape_adapters):
            module.requires_grad_(False)
            module.eval()
            module.to(device=self.device, dtype=self.dtype)
        self.aggregator = aggregator
        self.shape_encoder = shape_encoder
        self.backbone_shape_adapters = backbone_shape_adapters

    def _load_shared_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        """Load extra tensors that a subclass takes from the same checkpoint."""

    def _conditioned_aggregate(
        self,
        images: torch.Tensor,
        shape_tokens: torch.Tensor,
        shape_offsets: torch.Tensor,
    ):
        aggregator = self.aggregator
        batch, views, channels, height, width = images.shape
        normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
        flat_images = normalized.view(batch * views, channels, height, width)
        patch = aggregator.patch_embed(flat_images)
        if isinstance(patch, dict):
            patch = patch["x_norm_patchtokens"]

        camera = aggregator.camera_token[:, :1].expand(batch, 1, 1, -1)
        if views > 1:
            camera = torch.cat(
                (camera, aggregator.camera_token[:, 1:2].expand(batch, views - 1, 1, -1)),
                dim=1,
            )
        register = aggregator.register_token[:, :1].expand(batch, 1, -1, -1)
        if views > 1:
            register = torch.cat(
                (
                    register,
                    aggregator.register_token[:, 1:2].expand(batch, views - 1, -1, -1),
                ),
                dim=1,
            )
        tokens = torch.cat(
            (
                camera.reshape(batch * views, 1, -1),
                register.reshape(batch * views, 4, -1),
                patch,
            ),
            dim=1,
        )
        token_count, token_dim = tokens.shape[1:]
        position = aggregator.position_getter(
            batch * views,
            height // aggregator.patch_size,
            width // aggregator.patch_size,
            device=images.device,
        )
        position = torch.cat(
            (
                torch.zeros(
                    batch * views,
                    aggregator.patch_start_idx,
                    2,
                    device=images.device,
                    dtype=position.dtype,
                ),
                position + 1,
            ),
            dim=1,
        )

        frame_idx = 0
        global_idx = 0
        outputs = []
        for layer_idx in range(aggregator.depth):
            tokens, frame_idx, frame_intermediate = aggregator._process_frame_attention(
                tokens,
                batch,
                views,
                token_count,
                token_dim,
                frame_idx,
                pos=position,
            )
            tokens, global_idx, global_intermediate = aggregator._process_global_attention(
                tokens,
                batch,
                views,
                token_count,
                token_dim,
                global_idx,
                pos=position,
            )
            global_value = global_intermediate[-1]
            if layer_idx in self.conditioning_layers:
                conditioned = self.backbone_shape_adapters[str(layer_idx)](
                    global_value.reshape(batch, views * token_count, token_dim),
                    shape_tokens,
                    shape_offsets,
                )
                global_value = conditioned.view(batch, views, token_count, token_dim)
                tokens = global_value.reshape(batch * views, token_count, token_dim)
            if layer_idx in self.feature_layers:
                outputs.append(torch.cat((frame_intermediate[-1], global_value), dim=-1))
            else:
                outputs.append(None)
        return outputs, aggregator.patch_start_idx

    def __call__(
        self,
        images: torch.Tensor,
        shape_coords: torch.Tensor,
        shape_offsets: torch.Tensor,
    ):
        if images.ndim != 5:
            raise ValueError(f"Expected VGGT images [B,V,3,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"Expected RGB VGGT images, got {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != (448, 448):
            raise ValueError(f"Expected 448x448 VGGT images, got {tuple(images.shape[-2:])}")
        if shape_coords.ndim != 2 or shape_coords.shape[-1] != 3:
            raise ValueError(f"Expected packed coords64 [sumN,3], got {tuple(shape_coords.shape)}")
        if shape_offsets.ndim != 1 or shape_offsets.numel() != images.shape[0] + 1:
            raise ValueError(
                f"Expected shape_offsets [B+1]={images.shape[0] + 1}, got {tuple(shape_offsets.shape)}"
            )
        if int(shape_offsets[0]) != 0 or int(shape_offsets[-1]) != shape_coords.shape[0]:
            raise ValueError("shape_offsets must start at 0 and end at packed coords64 length")
        if bool((shape_offsets[1:] <= shape_offsets[:-1]).any()):
            raise ValueError("Every sample must contain at least one coords64 token")
        if shape_coords.numel() == 0:
            raise ValueError("Packed coords64 condition cannot be empty")
        coord_min = int(shape_coords.min())
        coord_max = int(shape_coords.max())
        if coord_min < 0 or coord_max >= self.shape_grid_resolution:
            raise ValueError(f"coords64 values must be in [0,63], got min={coord_min} max={coord_max}")
        autocast_enabled = self.device.type == "cuda" and self.dtype != torch.float32
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=autocast_enabled,
            ),
        ):
            shape_tokens = self.shape_encoder(shape_coords.to(device=self.device, non_blocking=True))
            tokens, patch_start_idx = self._conditioned_aggregate(
                images.to(device=self.device, dtype=self.dtype, non_blocking=True),
                shape_tokens,
                shape_offsets.to(device=self.device, dtype=torch.long, non_blocking=True),
            )
        return tokens, int(patch_start_idx)


class VGGT4LayerDINOAdapter(nn.Module):
    """Map four conditioned-VGGT layers to a DINO-width spatial residual.

    Shape512 keeps the native 32x32 patch layout. Shape1024 requests a 2x
    nearest-neighbour expansion followed by a zero-initialized refinement.
    """

    def __init__(
        self,
        feature_layers: List[int] = (4, 11, 17, 23),
        feature_dim: int = 2048,
        per_layer_dim: int = 1024,
        hidden_dim: int = 4096,
        output_dim: int = 1024,
        input_patch_grid_size: int = 32,
        upsample_scale: int = 1,
    ) -> None:
        super().__init__()
        self.feature_layers = tuple(int(layer) for layer in feature_layers)
        if self.feature_layers != (4, 11, 17, 23):
            raise ValueError(
                f"Frozen VGGT point-head features must use layers (4, 11, 17, 23), got {self.feature_layers}"
            )
        self.feature_dim = int(feature_dim)
        self.per_layer_dim = int(per_layer_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.input_patch_grid_size = int(input_patch_grid_size)
        self.upsample_scale = int(upsample_scale)
        if self.input_patch_grid_size <= 0:
            raise ValueError("input_patch_grid_size must be positive")
        if self.upsample_scale not in (1, 2):
            raise ValueError(f"upsample_scale must be 1 or 2, got {self.upsample_scale}")
        expected_hidden = len(self.feature_layers) * self.per_layer_dim
        if self.hidden_dim != expected_hidden:
            raise ValueError(
                f"hidden_dim must equal four-layer concat width {expected_hidden}, got {self.hidden_dim}"
            )

        # Four independent, newly initialized LayerNorm -> Linear projections.
        # No point_head.* parameter is loaded from the VGGT checkpoint.
        self.input_norms = nn.ModuleList([nn.LayerNorm(self.feature_dim) for _ in self.feature_layers])
        self.input_projections = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.per_layer_dim) for _ in self.feature_layers]
        )

        # First MLP: mix the four concatenated VGGT layers while preserving width.
        self.hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.hidden_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Second MLP: produce the DINO-width residual. Its last layer is zero-init.
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.zeros_(self.output_fc2.weight)
        nn.init.zeros_(self.output_fc2.bias)

        self.up_refine = None
        if self.upsample_scale == 2:
            self.up_refine = nn.Sequential(
                nn.Conv2d(
                    self.output_dim,
                    self.output_dim,
                    kernel_size=3,
                    padding=1,
                    groups=self.output_dim,
                ),
                nn.GELU(),
                nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1),
            )
            nn.init.zeros_(self.up_refine[-1].weight)
            nn.init.zeros_(self.up_refine[-1].bias)

    def forward(self, aggregated_tokens, patch_start_idx: int) -> torch.Tensor:
        projected = []
        patch_count = None
        for norm, projection, layer_idx in zip(
            self.input_norms,
            self.input_projections,
            self.feature_layers,
        ):
            if layer_idx >= len(aggregated_tokens) or aggregated_tokens[layer_idx] is None:
                raise RuntimeError(f"VGGT aggregator did not cache requested layer {layer_idx}")
            layer_tokens = aggregated_tokens[layer_idx]
            patch_tokens = layer_tokens[:, :, int(patch_start_idx) :]
            if patch_tokens.shape[-1] != self.feature_dim:
                raise RuntimeError(
                    f"Layer {layer_idx} feature width is {patch_tokens.shape[-1]}, "
                    f"expected {self.feature_dim}"
                )
            if patch_count is None:
                patch_count = int(patch_tokens.shape[2])
            elif int(patch_tokens.shape[2]) != patch_count:
                raise RuntimeError(
                    f"Inconsistent VGGT patch counts: {patch_count} vs {patch_tokens.shape[2]}"
                )
            projected.append(projection(norm(patch_tokens)))

        hidden_input = torch.cat(projected, dim=-1)
        if hidden_input.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"Concatenated VGGT width is {hidden_input.shape[-1]}, expected {self.hidden_dim}"
            )

        hidden_delta = self.hidden_fc2(F.gelu(self.hidden_fc1(self.hidden_norm(hidden_input))))
        hidden = hidden_input + hidden_delta
        output_hidden = F.gelu(self.output_fc1(self.output_norm(hidden)))
        delta = self.output_fc2(output_hidden)

        batch, views, patch_count, channels = delta.shape
        expected_patch_count = self.input_patch_grid_size**2
        if patch_count != expected_patch_count:
            raise RuntimeError(
                "Expected native VGGT patch grid "
                f"{self.input_patch_grid_size}x{self.input_patch_grid_size}="
                f"{expected_patch_count}, got {patch_count} patches"
            )
        if channels != self.output_dim:
            raise RuntimeError(f"Adapter output width is {channels}, expected {self.output_dim}")
        if self.upsample_scale == 1:
            return delta

        delta_map = (
            delta.reshape(
                batch * views,
                self.input_patch_grid_size,
                self.input_patch_grid_size,
                channels,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        repeated = F.interpolate(
            delta_map,
            scale_factor=self.upsample_scale,
            mode="nearest",
        )
        aligned = repeated + self.up_refine(repeated)
        output_grid_size = self.input_patch_grid_size * self.upsample_scale
        return (
            aligned.permute(0, 2, 3, 1)
            .reshape(
                batch,
                views,
                output_grid_size**2,
                channels,
            )
            .contiguous()
        )
