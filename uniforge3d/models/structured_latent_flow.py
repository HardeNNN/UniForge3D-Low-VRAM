from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from ..modules.sparse.attention.routed_full_only_flash import (
    PackedRoutedFullOnlyLayoutBuilder,
    RoutedFullOnlyFlashCrossAttention,
)
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from .sparse_structure_flow import TimestepEmbedder
from .vggt_adapter import VGGT4LayerDINOAdapter


class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        skip_weight_init: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        if not skip_weight_init:
            self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)

            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs
    ) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


class RoutedFullOnlyCrossBlock(ModulatedSparseTransformerCrossBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.cross_attn
        new = RoutedFullOnlyFlashCrossAttention(
            old.channels,
            old.ctx_channels,
            old.num_heads,
            old.to_q.bias is not None,
            old.qk_rms_norm,
        )
        new.to_q.load_state_dict(old.to_q.state_dict())
        new.to_kv.load_state_dict(old.to_kv.state_dict())
        new.to_out.load_state_dict(old.to_out.state_dict())
        if old.qk_rms_norm:
            new.q_rms_norm.load_state_dict(old.q_rms_norm.state_dict())
            new.k_rms_norm.load_state_dict(old.k_rms_norm.state_dict())
        self.cross_attn = new

    def forward(self, x, mod, context):
        """Evaluate self-attention, routed cross-attention, and FFN."""
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation + mod
            ).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

        def self_part(value):
            h = value.replace(self.norm1(value.feats))
            h = h * (1 + scale_msa) + shift_msa
            return value + self.self_attn(h) * gate_msa

        def ffn_part(value):
            h = value.replace(self.norm3(value.feats))
            h = h * (1 + scale_mlp) + shift_mlp
            return value + self.mlp(h) * gate_mlp

        x = self_part(x)
        x = x + self.cross_attn(x.replace(self.norm2(x.feats)), context)
        x = ffn_part(x)
        return x


class RoutedFullOnlySLatFlowModel(SLatFlowModel):
    def __init__(
        self,
        *args,
        routed_layers=(5, 7, 10, 12, 15, 17, 20, 22, 25, 27),
        **kwargs,
    ):
        self.routed_layers = tuple(map(int, routed_layers))
        super().__init__(*args, **kwargs)
        if self.resolution not in (32, 64):
            raise ValueError("Routed shape/texture resolution must be 32 or 64")
        self.full_tokens = self.resolution ** 2 + 5
        if len(set(self.routed_layers)) != len(self.routed_layers) or any(
            index < 0 or index >= self.num_blocks for index in self.routed_layers
        ):
            raise ValueError(f"invalid routed_layers={self.routed_layers}")

        for index in self.routed_layers:
            old = self.blocks[index]
            new = RoutedFullOnlyCrossBlock(
                self.model_channels,
                self.cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode="full",
                use_rope=self.pe_mode == "rope",
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            if missing or unexpected:
                raise RuntimeError((missing, unexpected))
            self.blocks[index] = new
            self.blocks[index].cross_attn.full_tokens = self.full_tokens

        self.route_layout_builder = PackedRoutedFullOnlyLayoutBuilder(grid_resolution=self.resolution)
        self.convert_to(self.dtype)

    def prepare_condition(
        self,
        x: sp.SparseTensor,
        cond: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor | int]:
        """Prepare per-view routing and all-view tokens once before denoising.

        Reuse only while image features, view assignments and sparse-row order
        remain unchanged. CFG's positive and negative conditions are separate.
        """
        if not isinstance(cond, dict):
            raise TypeError("structured routed condition required")
        image_features = manual_cast(cond["full_bank"], self.dtype)
        if image_features.ndim != 4 or int(image_features.shape[2]) != self.full_tokens:
            raise ValueError(
                f"Routed condition must be [B,V,{self.full_tokens},C], "
                f"got {tuple(image_features.shape)}"
            )
        batch_size, num_views = map(int, image_features.shape[:2])
        attention_layout = self.route_layout_builder.prepare(
            x=x,
            selected_view=cond["selected_view"],
            batch_size=batch_size,
            views=num_views,
        )
        all_view_tokens = image_features.reshape(batch_size, -1, image_features.shape[-1]).contiguous()
        return {
            **cond,
            "full_bank": image_features,
            "all_view_tokens": all_view_tokens,
            **attention_layout,
        }

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor | int],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs,
    ):
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        # Samplers prepare both CFG branches once; direct model calls also work.
        if "all_view_tokens" not in cond:
            cond = self.prepare_condition(x, cond)

        hidden = manual_cast(self.input_layer(x), self.dtype)
        time_condition = self.t_embedder(t)
        if self.share_mod:
            time_condition = self.adaLN_modulation(time_condition)
        time_condition = manual_cast(time_condition, self.dtype)

        for index, block in enumerate(self.blocks):
            # Routed layers use one selected view per voxel; others use all views.
            block_condition = cond if index in self.routed_layers else cond["all_view_tokens"]
            hidden = block(hidden, time_condition, block_condition)

        hidden = manual_cast(hidden, x.dtype)
        hidden = hidden.replace(F.layer_norm(hidden.feats, hidden.feats.shape[-1:]))
        return self.out_layer(hidden)


class VGGTConditionedSLatFlowModel(nn.Module):
    """One stage model containing its flow denoiser and VGGT-to-DINO adapter."""

    def __init__(
        self,
        denoiser_args: dict,
        adapter_args: dict,
        *,
        skip_weight_init: bool = False,
    ) -> None:
        super().__init__()
        self.denoiser = RoutedFullOnlySLatFlowModel(
            **denoiser_args, skip_weight_init=skip_weight_init
        )
        self.adapter = VGGT4LayerDINOAdapter(**adapter_args)

    @property
    def in_channels(self) -> int:
        return self.denoiser.in_channels

    def prepare_condition(self, *args, **kwargs):
        return self.denoiser.prepare_condition(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.denoiser(*args, **kwargs)
