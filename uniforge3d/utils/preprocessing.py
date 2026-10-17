"""One alpha crop shared by every stage, with no disk intermediates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BackgroundRemover:


    def __init__(self, checkpoint: str | Path, device: torch.device, **download_kwargs) -> None:
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            checkpoint, trust_remote_code=True, **download_kwargs
        ).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        inputs = self.transform(image).unsqueeze(0).to(self.device)
        prediction = self.model(inputs)[-1].sigmoid().cpu()
        mask = transforms.ToPILImage()(prediction[0].squeeze()).resize(image.size)
        image.putalpha(mask)
        return image


@dataclass(frozen=True)
class ViewTensors:
    dino_images512: torch.Tensor
    dino_images1024: torch.Tensor
    vggt_images: torch.Tensor
    input_masks: torch.Tensor


@dataclass(frozen=True)
class InputBatch:

    dino_images512: torch.Tensor
    dino_images1024: torch.Tensor
    vggt_images: torch.Tensor
    input_masks: torch.Tensor


def alpha_bbox_crop_params(
    alpha: np.ndarray,
    padding_factor: float = 1.0,
    alpha_threshold: float = 0.0,
    filepath: str = "",
) -> tuple[int, int, int]:
    mask = np.asarray(alpha) > float(alpha_threshold)
    if not np.any(mask):
        raise ValueError(f"{filepath}: empty alpha mask")
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    side = max(x1 - x0 + 1, y1 - y0 + 1)
    if float(padding_factor) != 1.0:
        side = max(1, int(math.ceil(float(side) * float(padding_factor))))
    center_x = (x0 + x1 + 1) * 0.5
    center_y = (y0 + y1 + 1) * 0.5
    return int(math.floor(center_x - side * 0.5)), int(math.floor(center_y - side * 0.5)), int(side)


def crop_with_padding_np(
    array: np.ndarray,
    left: int,
    top: int,
    side: int,
    fill_value: float | int = 0,
) -> np.ndarray:
    shape = (side, side) if array.ndim == 2 else (side, side, array.shape[2])
    output = np.full(shape, fill_value, dtype=array.dtype)
    height, width = array.shape[:2]
    src_left, src_top = max(left, 0), max(top, 0)
    src_right, src_bottom = min(left + side, width), min(top + side, height)
    if src_right <= src_left or src_bottom <= src_top:
        return output
    dst_left, dst_top = src_left - left, src_top - top
    output[
        dst_top : dst_top + src_bottom - src_top,
        dst_left : dst_left + src_right - src_left,
    ] = array[src_top:src_bottom, src_left:src_right]
    return output


def crop_resize_rgb_np(rgb: np.ndarray, crop: tuple[int, int, int], image_size: int) -> torch.Tensor:
    cropped = crop_with_padding_np(rgb, *crop, fill_value=0)
    resized = Image.fromarray(cropped).resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float()


def preprocess_view(
    path: Path,
    config: dict[str, Any],
    remove_background: Callable[[Image.Image], Image.Image],
) -> ViewTensors:
    alpha_threshold = float(config["alpha_threshold"])
    with Image.open(path) as image:
        # Match TRELLIS.2: only non-opaque RGBA inputs already have an alpha mask.
        if image.mode != "RGBA" or np.all(np.asarray(image.getchannel("A")) == 255):
            image = remove_background(image.convert("RGB"))
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3]
    mask = alpha > alpha_threshold
    if not np.any(mask):
        raise ValueError(f"{path}: empty alpha mask")
    # Preserve the trained black-background alpha composite and rounding.
    rgb = (rgba[..., :3] * rgba[..., 3:4] * 255.0).clip(0, 255).astype(np.uint8)
    crop = alpha_bbox_crop_params(alpha, config["padding_factor"], alpha_threshold, str(path))
    dino_images512 = crop_resize_rgb_np(rgb, crop, 512)
    dino_images1024 = crop_resize_rgb_np(rgb, crop, 1024)
    vggt448 = crop_resize_rgb_np(rgb, crop, 448)
    alpha_crop = crop_with_padding_np(alpha, *crop, fill_value=0.0)
    resized_alpha = np.asarray(
        Image.fromarray(np.clip(alpha_crop * 255.0, 0, 255).astype(np.uint8)).resize(
            (448, 448), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )
    # Routing uses confident foreground, independently of the crop threshold.
    masks448 = torch.from_numpy((resized_alpha > 127).copy()).bool()
    return ViewTensors(dino_images512, dino_images1024, vggt448, masks448)


def build_input_batch(
    images: Sequence[str | Path],
    config: dict[str, Any],
    remove_background: Callable[[Image.Image], Image.Image],
) -> InputBatch:
    paths = [Path(image).expanduser().resolve() for image in images]
    views = [preprocess_view(path, config, remove_background) for path in paths]
    return InputBatch(
        dino_images512=torch.stack([view.dino_images512 for view in views]).unsqueeze(0),
        dino_images1024=torch.stack([view.dino_images1024 for view in views]).unsqueeze(0),
        vggt_images=torch.stack([view.vggt_images for view in views]).unsqueeze(0),
        input_masks=torch.stack([view.input_masks for view in views]).unsqueeze(0),
    )
