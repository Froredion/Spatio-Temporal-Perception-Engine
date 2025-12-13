"""
Image Utility Functions

Common image processing operations.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from PIL import Image


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor.

    Args:
        image: PIL Image (RGB)
        normalize: Whether to normalize to [0, 1]

    Returns:
        (3, H, W) tensor
    """
    img_array = np.array(image)

    # Handle grayscale
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # HWC to CHW
    tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    if normalize:
        tensor = tensor.float() / 255.0

    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image.

    Args:
        tensor: (3, H, W) or (H, W, 3) tensor

    Returns:
        PIL Image
    """
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        # CHW to HWC
        tensor = tensor.permute(1, 2, 0)

    # Handle single channel
    if tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)

    # Denormalize if in [0, 1]
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).clamp(0, 255)

    # Convert to numpy
    array = tensor.cpu().numpy().astype(np.uint8)

    return Image.fromarray(array)


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    keep_aspect_ratio: bool = True,
    method: str = 'lanczos',
) -> Image.Image:
    """
    Resize image.

    Args:
        image: PIL Image
        size: Target size (int for square, or (width, height))
        keep_aspect_ratio: Whether to maintain aspect ratio
        method: Resampling method ('lanczos', 'bilinear', 'nearest')

    Returns:
        Resized PIL Image
    """
    methods = {
        'lanczos': Image.LANCZOS,
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'bicubic': Image.BICUBIC,
    }
    resample = methods.get(method.lower(), Image.LANCZOS)

    if isinstance(size, int):
        size = (size, size)

    if keep_aspect_ratio:
        # Calculate new size maintaining aspect ratio
        orig_w, orig_h = image.size
        target_w, target_h = size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = image.resize((new_w, new_h), resample)

        # Pad to target size
        if new_w != target_w or new_h != target_h:
            padded = Image.new('RGB', size, (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            padded.paste(resized, (paste_x, paste_y))
            return padded

        return resized
    else:
        return image.resize(size, resample)


def center_crop(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
) -> Image.Image:
    """
    Center crop image to size.

    Args:
        image: PIL Image
        size: Target size (int for square, or (width, height))

    Returns:
        Cropped PIL Image
    """
    if isinstance(size, int):
        size = (size, size)

    width, height = image.size
    target_w, target_h = size

    left = (width - target_w) // 2
    top = (height - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return image.crop((left, top, right, bottom))


def pad_image(
    image: Image.Image,
    size: Tuple[int, int],
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Pad image to target size.

    Args:
        image: PIL Image
        size: Target (width, height)
        fill: Fill color (RGB)

    Returns:
        Padded PIL Image
    """
    target_w, target_h = size
    orig_w, orig_h = image.size

    if orig_w >= target_w and orig_h >= target_h:
        return image

    padded = Image.new('RGB', size, fill)

    paste_x = (target_w - orig_w) // 2
    paste_y = (target_h - orig_h) // 2

    padded.paste(image, (paste_x, paste_y))

    return padded


def normalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Normalize image tensor with ImageNet statistics.

    Args:
        image: (C, H, W) or (B, C, H, W) tensor in [0, 1]
        mean: Channel means
        std: Channel stds

    Returns:
        Normalized tensor
    """
    mean_t = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=image.device).view(-1, 1, 1)

    if image.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    return (image - mean_t) / std_t


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Reverse ImageNet normalization.

    Args:
        image: Normalized tensor
        mean: Channel means used for normalization
        std: Channel stds used for normalization

    Returns:
        Denormalized tensor in [0, 1]
    """
    mean_t = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=image.device).view(-1, 1, 1)

    if image.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    return (image * std_t + mean_t).clamp(0, 1)


def create_grid(
    images: list,
    nrow: int = 4,
    padding: int = 2,
    pad_value: float = 0,
) -> Image.Image:
    """
    Create image grid from list of PIL Images.

    Args:
        images: List of PIL Images (all same size)
        nrow: Number of images per row
        padding: Padding between images
        pad_value: Padding value (0-255)

    Returns:
        Grid PIL Image
    """
    if not images:
        return Image.new('RGB', (1, 1))

    # Get dimensions
    w, h = images[0].size
    n = len(images)
    ncol = (n + nrow - 1) // nrow

    # Create grid
    grid_w = nrow * w + (nrow + 1) * padding
    grid_h = ncol * h + (ncol + 1) * padding

    grid = Image.new('RGB', (grid_w, grid_h), (int(pad_value),) * 3)

    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow

        x = padding + col * (w + padding)
        y = padding + row * (h + padding)

        grid.paste(img, (x, y))

    return grid
