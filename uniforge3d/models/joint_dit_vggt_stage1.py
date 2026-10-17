"""Inference-only Stage1 DiT/VGGT model topology."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniforge3d.models.vggt.models.aggregator import slice_expand_and_flatten
from uniforge3d.models.vggt.models.vggt import VGGT

from ..modules.utils import manual_cast
from .sparse_structure_flow import SparseStructureFlowModel


def _square_grid_size(num_tokens: int) -> tuple[int, int]:
    size = int(round(math.sqrt(int(num_tokens))))
    if size * size != int(num_tokens):
        raise ValueError(f"Expected a square token grid, got {num_tokens} tokens")
    return size, size


def _vggt_patch_map(tokens: torch.Tensor, patch_start_idx: int) -> torch.Tensor:
    batch, views, _token_count, channels = tokens.shape
    patch_tokens = tokens[:, :, patch_start_idx:]
    height, width = _square_grid_size(patch_tokens.shape[2])
    return (
        patch_tokens.reshape(batch * views, height, width, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _align_vggt_condition(
    frame_tokens: torch.Tensor,
    global_tokens: torch.Tensor,
    patch_start_idx: int,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    frame_map = _vggt_patch_map(frame_tokens, patch_start_idx)
    global_map = _vggt_patch_map(global_tokens, patch_start_idx)
    if tuple(frame_map.shape[-2:]) != target_hw:
        raise ValueError(
            f"VGGT patch grid {tuple(frame_map.shape[-2:])} does not match "
            f"DINO condition grid {target_hw}"
        )
    return torch.cat((frame_map, global_map), dim=1)


class BlockDinoCondDeltaMLP(nn.Module):
    """Block-specific zero-init residual on DINO patch tokens, predicted from aligned VGGT features."""

    def __init__(self, dim: int = 1024, hidden_dim: int = 1024, special_tokens: int = 5):
        super().__init__()
        self.dim = int(dim)
        self.special_tokens = int(special_tokens)
        self.net = nn.Sequential(
            nn.GroupNorm(32, self.dim * 2),
            nn.Conv2d(self.dim * 2, int(hidden_dim), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1, groups=int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), self.dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, dino_cond_bank: torch.Tensor, aligned_vggt: torch.Tensor) -> torch.Tensor:
        if dino_cond_bank.ndim != 4:
            raise ValueError(f"Expected dino_cond_bank [B,V,N,C], got {tuple(dino_cond_bank.shape)}")
        batch, views, cond_tokens, dim = dino_cond_bank.shape
        if dim != self.dim:
            raise ValueError(f"Expected cond dim={self.dim}, got {dim}")
        cond_patch_tokens = int(cond_tokens) - self.special_tokens
        cond_hw = _square_grid_size(cond_patch_tokens)
        expected_shape = (batch * views, self.dim * 2, cond_hw[0], cond_hw[1])
        if tuple(aligned_vggt.shape) != expected_shape:
            raise ValueError(f"Expected aligned_vggt {expected_shape}, got {tuple(aligned_vggt.shape)}")
        delta = self.net(aligned_vggt).permute(0, 2, 3, 1).reshape(batch, views, cond_patch_tokens, dim)
        out = dino_cond_bank.clone()
        out[:, :, self.special_tokens:] = out[:, :, self.special_tokens:] + delta.to(dtype=out.dtype)
        return out


class DiTToVGGTResidual(nn.Module):
    """Inject a timestep-modulated DiT residual into the VGGT token stream."""

    def __init__(
        self,
        vggt_dim: int,
        dit_dim: int,
        t_dim: int,
        heads: int = 8,
        gate_cap: float = 1.0,
    ):
        super().__init__()
        self.gate_cap = float(gate_cap)
        self.norm_q = nn.LayerNorm(vggt_dim)
        self.norm_ctx = nn.LayerNorm(dit_dim)
        self.attn = nn.MultiheadAttention(
            vggt_dim,
            heads,
            kdim=dit_dim,
            vdim=dit_dim,
            batch_first=True,
        )
        self.delta_proj = nn.Linear(vggt_dim, vggt_dim)
        self.t_mod = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, vggt_dim * 3))
        nn.init.zeros_(self.t_mod[-1].weight)
        nn.init.zeros_(self.t_mod[-1].bias)

    def forward(
        self,
        vggt_tokens: torch.Tensor,
        dit_tokens: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm_q(vggt_tokens)
        ctx = self.norm_ctx(dit_tokens)
        shift, scale, gate = self.t_mod(t_emb).chunk(3, dim=-1)
        shift = shift[:, None, :].to(dtype=q.dtype)
        scale = scale[:, None, :].to(dtype=q.dtype)
        gate = gate[:, None, :].to(dtype=q.dtype)
        q = q * (1.0 + scale) + shift
        h, _ = self.attn(q, ctx, ctx, need_weights=False)
        delta = self.delta_proj(h)
        return vggt_tokens + self.gate_cap * gate.to(dtype=delta.dtype) * delta


class JointDiTVGGTStage1Model(nn.Module):
    """Inference-only Stage1 sparse-structure model with DINO/VGGT fusion."""

    def __init__(
        self,
        *,
        dit_args: dict[str, Any],
        dino_image_size: int,
        vggt_img_size: int = 518,
        vggt_patch_size: int = 14,
        fusion_vggt_layers: Sequence[int] = (2, 5, 8, 11, 14, 17, 20, 23),
        fusion_dit_blocks: Sequence[int] = (3, 7, 11, 15, 19, 23, 26, 29),
        anchor_view: int = 0,
        cond_fusion_hidden_dim: int = 1024,
        dit_to_vggt_gate_cap: float = 0.1,
    ):
        super().__init__()
        self.anchor_view = int(anchor_view)
        dino_image_size = int(dino_image_size)
        self.dino_tokens = (dino_image_size // 16) ** 2 + 5
        self.dit = SparseStructureFlowModel(**dit_args)
        self.vggt = VGGT(
            img_size=vggt_img_size,
            patch_size=vggt_patch_size,
            enable_camera=False,
            enable_point=True,
        )
        # Stage1 never supplies patch masks, so masked-pretraining state is omitted.
        self.vggt.aggregator.patch_embed.mask_token = None

        fusion_dit_blocks = tuple(int(index) for index in fusion_dit_blocks)
        self.vggt_to_block = dict(zip(fusion_vggt_layers, fusion_dit_blocks))
        t_dim = self.dit.model_channels
        z_dim = self.dit.model_channels
        f_dim = 1024
        total_dit_blocks = int(len(self.dit.blocks))
        stage_boundaries = (*fusion_dit_blocks, total_dit_blocks)
        self.dit_stage_blocks = {
            start: tuple(range(start, min(end, total_dit_blocks)))
            for start, end in zip(stage_boundaries, stage_boundaries[1:])
            if start < total_dit_blocks
        }
        all_fusion_dit_blocks = sorted(
            {block_idx for stage_blocks in self.dit_stage_blocks.values() for block_idx in stage_blocks}
        )
        self.cond_delta_mlps = nn.ModuleDict(
            {
                str(block_idx): BlockDinoCondDeltaMLP(f_dim, hidden_dim=cond_fusion_hidden_dim)
                for block_idx in all_fusion_dit_blocks
            }
        )
        self.dit_to_vggt = nn.ModuleDict(
            {
                str(block_idx): DiTToVGGTResidual(
                    vggt_dim=f_dim,
                    dit_dim=z_dim,
                    t_dim=t_dim,
                    heads=8,
                    gate_cap=dit_to_vggt_gate_cap,
                )
                for block_idx in fusion_dit_blocks
            }
        )

    @staticmethod
    def _flatten_condition(cond_bank: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if cond_bank.ndim != 4:
            raise ValueError(f"Expected condition bank [B,V,N,C], got {tuple(cond_bank.shape)}")
        batch, views, tokens, channels = cond_bank.shape
        return manual_cast(cond_bank.reshape(batch, views * tokens, channels), dtype)

    def _zero_condition(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dim = int(getattr(self.dit, "cond_channels", self.dit.model_channels))
        return torch.zeros(
            int(batch_size),
            self.dino_tokens,
            dim,
            device=device,
            dtype=dtype,
        )

    def forward_unconditional(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        cond = self._zero_condition(
            batch_size=x_t.shape[0],
            device=x_t.device,
            dtype=x_t.dtype,
        )
        return self.dit(x_t, t.to(dtype=torch.float32) * 1000.0, cond)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        vggt_images: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity conditioned on the input views."""
        if vggt_images.ndim != 5:
            raise ValueError("Stage1 VGGT images must be [B,V,3,H,W]")
        expected_features = (*vggt_images.shape[:2], self.dino_tokens, 1024)
        if tuple(dino_features.shape) != expected_features:
            raise ValueError(
                f"Expected cached Stage1 DINO features {expected_features}, "
                f"got {tuple(dino_features.shape)}"
            )
        anchor_idx = min(max(self.anchor_view, 0), vggt_images.shape[1] - 1)
        cond = dino_features[:, anchor_idx]
        return self.forward_joint_with_fusion(
            x_t,
            t,
            cond,
            vggt_images,
            dino_cond_bank=dino_features,
        )

    def _init_vggt_tokens(self, images: torch.Tensor):
        aggregator = self.vggt.aggregator
        bsz, views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected 3 input channels, got {channels}")

        images = (images - aggregator._resnet_mean) / aggregator._resnet_std
        flat_images = images.view(bsz * views, channels, height, width)
        patch_tokens = aggregator.patch_embed(flat_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        camera_token = slice_expand_and_flatten(aggregator.camera_token, bsz, views)
        register_token = slice_expand_and_flatten(aggregator.register_token, bsz, views)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if aggregator.rope is not None:
            pos = aggregator.position_getter(
                bsz * views,
                height // aggregator.patch_size,
                width // aggregator.patch_size,
                device=images.device,
            )

        if aggregator.patch_start_idx > 0 and pos is not None:
            pos = pos + 1
            pos_special = torch.zeros(
                bsz * views,
                aggregator.patch_start_idx,
                2,
                device=images.device,
                dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, token_count, embed_dim = tokens.shape
        return tokens, pos, bsz, views, token_count, embed_dim, aggregator.patch_start_idx

    def forward_joint_with_fusion(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        vggt_images: torch.Tensor,
        dino_cond_bank: torch.Tensor,
    ) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.dit.in_channels, *[self.dit.resolution] * 3], (
            f"Input shape mismatch, got {x.shape}"
        )
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.dit.input_layer(h)
        if self.dit.pe_mode == "ape":
            h = h + self.dit.pos_emb[None]
        t_raw = self.dit.t_embedder(t * 1000.0)
        t_mod = self.dit.adaLN_modulation(t_raw) if self.dit.share_mod else t_raw
        t_mod = manual_cast(t_mod, self.dit.dtype)
        h = manual_cast(h, self.dit.dtype)
        cond = manual_cast(cond, self.dit.dtype)

        aggregator = self.vggt.aggregator
        tokens, pos, bsz, views, token_count, embed_dim, patch_start_idx = self._init_vggt_tokens(vggt_images)
        dino_cond_bank = dino_cond_bank[:, :views].to(device=cond.device, dtype=cond.dtype).contiguous()

        frame_idx = 0
        global_idx = 0
        next_dit_block = 0

        for layer_idx in range(aggregator.depth):
            tokens, frame_idx, frame_intermediates = aggregator._process_frame_attention(
                tokens, bsz, views, token_count, embed_dim, frame_idx, pos=pos
            )
            tokens, global_idx, global_intermediates = aggregator._process_global_attention(
                tokens, bsz, views, token_count, embed_dim, global_idx, pos=pos
            )

            frame_tokens = frame_intermediates[-1]
            global_tokens = global_intermediates[-1]
            if layer_idx in self.vggt_to_block:
                stage_start_block = int(self.vggt_to_block[layer_idx])
                stage_blocks = self.dit_stage_blocks[stage_start_block]
                global_tokens_raw = global_tokens
                while next_dit_block < stage_start_block:
                    h = self.dit.blocks[next_dit_block](h, t_mod, cond, self.dit.rope_phases)
                    next_dit_block += 1

                cond_patch_tokens = int(dino_cond_bank.shape[2]) - 5
                cond_hw = _square_grid_size(cond_patch_tokens)
                aligned_vggt = _align_vggt_condition(
                    frame_tokens,
                    global_tokens_raw,
                    patch_start_idx,
                    cond_hw,
                )

                for block_idx in stage_blocks:
                    if next_dit_block != block_idx:
                        raise RuntimeError(f"Expected DiT block {block_idx}, got {next_dit_block}")
                    fused_cond = self.cond_delta_mlps[str(block_idx)](dino_cond_bank, aligned_vggt)
                    block_context = self._flatten_condition(fused_cond, self.dit.dtype)
                    h = self.dit.blocks[block_idx](h, t_mod, block_context, self.dit.rope_phases)
                    next_dit_block += 1

                # Write the completed DiT stage back to the VGGT global stream.
                f_query_tokens = manual_cast(global_tokens_raw, self.dit.dtype)
                batch_q, views_q, query_tokens, dim_q = f_query_tokens.shape
                f_query_flat = f_query_tokens.reshape(batch_q, views_q * query_tokens, dim_q)
                f_fused_flat = self.dit_to_vggt[str(stage_start_block)](
                    f_query_flat,
                    h,
                    t_raw,
                )
                global_tokens_fused = f_fused_flat.reshape(batch_q, views_q, query_tokens, dim_q)
                global_tokens = manual_cast(global_tokens_fused, global_tokens_raw.dtype)
                tokens = global_tokens.reshape(bsz * views, token_count, embed_dim)

        while next_dit_block < len(self.dit.blocks):
            block = self.dit.blocks[next_dit_block]
            h = block(h, t_mod, cond, self.dit.rope_phases)
            next_dit_block += 1

        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.dit.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.dit.resolution] * 3).contiguous()
        return h
