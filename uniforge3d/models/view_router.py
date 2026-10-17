"""Full-image view routing shared by the 32^3 shape and 64^3 shape/PBR stages."""

from __future__ import annotations

import math

import torch


class GPUVoxelViewRouter:
    """Route each target voxel to one full image, without local-window computation.

    Shape512 uses a 32^3 query grid and a 32x32 patch grid. Shape1024 and
    texture use 64^3 / 64x64. Both consume the same 448x448 point-map format;
    the VGGT structure-conditioning grid is configured separately.
    """

    def __init__(
        self,
        confidence_keep_ratio: float = 0.95,
        component_iterations: int = 64,
        grid_resolution: int = 64,
        patch_grid_size: int = 64,
        image_size: int = 448,
    ) -> None:
        self.keep = float(confidence_keep_ratio)
        self.iters = int(component_iterations)
        self.grid = int(grid_resolution)
        self.patch_grid = int(patch_grid_size)
        self.image_size = int(image_size)
        self.volume = self.grid**3
        self.patch_count = self.patch_grid**2
        if not 0.0 < self.keep <= 1.0:
            raise ValueError("confidence_keep_ratio must be in (0,1]")
        if self.iters <= 0:
            raise ValueError("component_iterations must be positive")
        if (self.grid, self.patch_grid) not in ((32, 32), (64, 64)) or self.image_size != 448:
            raise ValueError("Router requires (grid, patch_grid)=(32,32) or (64,64), image_size=448")
        if self.image_size % self.patch_grid:
            raise ValueError("image_size must be divisible by patch_grid_size")

    @staticmethod
    def _sum(src: torch.Tensor, idx: torch.Tensor, size: int) -> torch.Tensor:
        return torch.zeros(size, dtype=src.dtype, device=src.device).scatter_add_(0, idx, src)

    def _top_confidence(self, confidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        keep = torch.zeros_like(valid)
        batch, views, _ = valid.shape
        for batch_idx in range(batch):
            for view_idx in range(views):
                ids = torch.nonzero(valid[batch_idx, view_idx]).flatten()
                if ids.numel() == 0:
                    continue
                count = math.ceil(ids.numel() * self.keep)
                order = torch.argsort(
                    confidence[batch_idx, view_idx, ids],
                    descending=True,
                    stable=True,
                )
                keep[batch_idx, view_idx, ids[order[:count]]] = True
        return keep

    def _entries(
        self,
        points: torch.Tensor,
        confidence: torch.Tensor,
        input_masks: torch.Tensor,
    ):
        batch, views, height, width, _ = points.shape
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"router expects {self.image_size}x{self.image_size} points, got {(height, width)}"
            )
        flat_points = points.reshape(batch, views, -1, 3).float()
        flat_conf = confidence.reshape(batch, views, -1).float()
        valid = (
            input_masks.reshape(batch, views, -1).bool()
            & torch.isfinite(flat_points).all(-1)
            & torch.isfinite(flat_conf)
            & ((flat_points >= -0.5) & (flat_points < 0.5)).all(-1)
        )
        valid &= self._top_confidence(flat_conf, valid)

        pixels = torch.arange(height * width, device=points.device).view(1, 1, -1)
        patch_pixels = self.image_size // self.patch_grid
        patch = ((pixels // width) // patch_pixels) * self.patch_grid + ((pixels % width) // patch_pixels)
        xyz = torch.floor((flat_points + 0.5) * self.grid).long()
        voxel = xyz[..., 0] * self.grid * self.grid + xyz[..., 1] * self.grid + xyz[..., 2]
        batch_ids = torch.arange(batch, device=points.device).view(batch, 1, 1)
        batch_ids = batch_ids.expand_as(valid)
        view_ids = torch.arange(views, device=points.device).view(1, views, 1)
        view_ids = view_ids.expand_as(valid)
        key = (
            ((batch_ids * views + view_ids) * self.volume + voxel) * self.patch_count + patch.expand_as(valid)
        )[valid]
        conf = flat_conf[valid]
        if key.numel() == 0:
            return None

        order = torch.argsort(key, stable=True)
        key, conf = key[order], conf[order]
        unique, inverse, counts = torch.unique_consecutive(key, return_inverse=True, return_counts=True)
        return {
            "key": unique,
            "group": unique // self.patch_count,
            "patch": unique % self.patch_count,
            "count": counts.long(),
            "conf": self._sum(conf, inverse, unique.numel()),
        }

    def _summaries(self, entries):
        if entries is None:
            return None
        key = entries["key"]
        group = entries["group"]
        patch = entries["patch"]
        count = entries["count"]
        conf_sum = entries["conf"]
        num_entries = key.numel()
        unique_groups, entry_group_ids = torch.unique_consecutive(group, return_inverse=True)
        num_groups = unique_groups.numel()

        # 8-connected components on the configured patch grid. The early exit only
        # checks convergence; it does not alter the deterministic update rule.
        labels = torch.arange(num_entries, device=key.device)
        patch_y, patch_x = patch // self.patch_grid, patch % self.patch_grid
        for _ in range(self.iters):
            old_labels = labels
            choices = [labels]
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor_y = patch_y + delta_y
                    neighbor_x = patch_x + delta_x
                    inside = (
                        (neighbor_y >= 0)
                        & (neighbor_y < self.patch_grid)
                        & (neighbor_x >= 0)
                        & (neighbor_x < self.patch_grid)
                    )
                    neighbor_key = (
                        group * self.patch_count
                        + neighbor_y.clamp(0, self.patch_grid - 1) * self.patch_grid
                        + neighbor_x.clamp(0, self.patch_grid - 1)
                    )
                    position = torch.searchsorted(key, neighbor_key)
                    safe = position.clamp(max=num_entries - 1)
                    found = inside & (position < num_entries) & (key[safe] == neighbor_key)
                    choices.append(torch.where(found, labels[safe], labels))
            labels = torch.stack(choices).amin(0)
            labels = labels[labels]
            if torch.equal(labels, old_labels):
                break

        component_pixels = self._sum(count, labels, num_entries)
        component_conf = self._sum(conf_sum, labels, num_entries)
        all_ids = torch.arange(num_entries, device=key.device)
        roots = torch.nonzero(labels == all_ids).flatten()
        root_group_ids = entry_group_ids[roots]

        root_support = component_pixels[roots]
        max_support = torch.full((num_groups,), -1, dtype=root_support.dtype, device=key.device)
        max_support.scatter_reduce_(0, root_group_ids, root_support, reduce="amax", include_self=True)
        support_winner = root_support == max_support[root_group_ids]

        root_mean = component_conf[roots] / root_support.clamp_min(1)
        max_mean = torch.full((num_groups,), float("-inf"), dtype=root_mean.dtype, device=key.device)
        max_mean.scatter_reduce_(
            0,
            root_group_ids,
            torch.where(support_winner, root_mean, torch.full_like(root_mean, float("-inf"))),
            reduce="amax",
            include_self=True,
        )
        root_winner = support_winner & (root_mean == max_mean[root_group_ids])
        root_sentinel = torch.full_like(roots, num_entries)
        chosen_roots = torch.full((num_groups,), num_entries, dtype=roots.dtype, device=key.device)
        chosen_roots.scatter_reduce_(
            0,
            root_group_ids,
            torch.where(root_winner, roots, root_sentinel),
            reduce="amin",
            include_self=True,
        )

        return {
            "group": unique_groups,
            "support": component_pixels[chosen_roots],
            "mean": component_conf[chosen_roots] / component_pixels[chosen_roots].clamp_min(1),
        }

    def __call__(
        self,
        points: torch.Tensor,
        confidence: torch.Tensor,
        input_masks: torch.Tensor,
        target_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, views = points.shape[:2]
        device = points.device
        total = batch * self.volume
        selected_view = torch.zeros(total, dtype=torch.long, device=device)
        summary = self._summaries(self._entries(points, confidence, input_masks))
        if summary is None:
            return {"selected_view": selected_view.view(batch, self.grid, self.grid, self.grid).detach()}

        # Direct support: most pixels, then anchor, confidence, and lowest view index.
        group = summary["group"]
        voxel = group % self.volume
        batch_view = group // self.volume
        view = batch_view % views
        sample = batch_view // views
        dense = sample * self.volume + voxel
        source_support = summary["support"]
        source_conf = summary["mean"]
        source_anchor = (view == 0).long()

        support = torch.zeros(total, dtype=source_support.dtype, device=device)
        support.scatter_reduce_(0, dense, source_support, reduce="amax", include_self=True)
        support_winner = source_support == support[dense]

        anchor = torch.full((total,), -1, dtype=source_anchor.dtype, device=device)
        anchor.scatter_reduce_(
            0, dense,
            torch.where(support_winner, source_anchor, torch.full_like(source_anchor, -1)),
            reduce="amax", include_self=True,
        )
        anchor_winner = support_winner & (source_anchor == anchor[dense])

        mean_conf = torch.full((total,), float("-inf"), dtype=source_conf.dtype, device=device)
        mean_conf.scatter_reduce_(
            0, dense,
            torch.where(anchor_winner, source_conf, torch.full_like(source_conf, float("-inf"))),
            reduce="amax", include_self=True,
        )
        conf_winner = anchor_winner & (source_conf == mean_conf[dense])
        selected_view.fill_(views)
        selected_view.scatter_reduce_(
            0, dense,
            torch.where(conf_winner, view, torch.full_like(view, views)),
            reduce="amin", include_self=True,
        )
        has_support = selected_view < views

        # Unsupported voxels default to the anchor; only missing targets need neighbors.
        selected_view.masked_fill_(~has_support, 0)
        target_batch = target_coords[:, 0]
        xyz = target_coords[:, 1:4]
        if bool(((xyz < 0) | (xyz >= self.grid)).any()):
            raise ValueError(f"target_coords contain values outside the {self.grid}^3 grid")
        target_voxel = xyz[:, 0] * self.grid * self.grid + xyz[:, 1] * self.grid + xyz[:, 2]
        target_dense = target_batch * self.volume + target_voxel
        missing = ~has_support[target_dense]
        xyz, target_batch, target_dense = xyz[missing], target_batch[missing], target_dense[missing]

        # Single-hop inheritance from directly supported 26-neighbors, with the same priorities.
        offset_axis = torch.arange(-1, 2, device=device, dtype=xyz.dtype)
        offsets = torch.stack(
            torch.meshgrid(offset_axis, offset_axis, offset_axis, indexing="ij"), -1,
        ).reshape(-1, 3)
        offsets = offsets[(offsets != 0).any(-1)]
        neighbor_xyz = xyz[:, None, :] + offsets[None]
        inside = ((neighbor_xyz >= 0) & (neighbor_xyz < self.grid)).all(-1)
        safe_xyz = neighbor_xyz.clamp(0, self.grid - 1)
        neighbor_voxel = (
            safe_xyz[..., 0] * self.grid * self.grid + safe_xyz[..., 1] * self.grid + safe_xyz[..., 2]
        )
        source_dense = target_batch[:, None] * self.volume + neighbor_voxel
        valid = inside & has_support[source_dense]

        candidate_support = support[source_dense]
        max_support = torch.where(
            valid, candidate_support, torch.full_like(candidate_support, -1),
        ).amax(1)
        support_winner = valid & (candidate_support == max_support[:, None])
        candidate_anchor = anchor[source_dense]
        max_anchor = torch.where(
            support_winner, candidate_anchor, torch.full_like(candidate_anchor, -1),
        ).amax(1)
        anchor_winner = support_winner & (candidate_anchor == max_anchor[:, None])
        candidate_conf = mean_conf[source_dense]
        max_conf = torch.where(
            anchor_winner, candidate_conf, torch.full_like(candidate_conf, float("-inf")),
        ).amax(1)
        conf_winner = anchor_winner & (candidate_conf == max_conf[:, None])
        candidate_view = selected_view[source_dense]
        neighbor_view = torch.where(
            conf_winner, candidate_view, torch.full_like(candidate_view, views),
        ).amin(1)
        chosen_neighbor = neighbor_view < views
        selected_view[target_dense[chosen_neighbor]] = neighbor_view[chosen_neighbor]
        return {"selected_view": selected_view.view(batch, self.grid, self.grid, self.grid).detach()}
