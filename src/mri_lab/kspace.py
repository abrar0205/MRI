"""K-space encoding utilities."""

from __future__ import annotations

import numpy as np


def fft2c(image: np.ndarray) -> np.ndarray:
    """Centered 2D Fourier transform."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm="ortho"))


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered inverse 2D Fourier transform."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho"))


def undersampling_mask(shape: tuple[int, int], acceleration: int = 4, center_fraction: float = 0.08) -> np.ndarray:
    """Create a Cartesian undersampling mask with a fully sampled center band."""
    if acceleration < 1:
        raise ValueError("acceleration must be >= 1")

    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    mask[::acceleration, :] = True

    center_lines = max(1, int(round(ny * center_fraction)))
    start = ny // 2 - center_lines // 2
    stop = start + center_lines
    mask[start:stop, :] = True
    return mask


def radial_sampling_mask(shape: tuple[int, int], spokes: int = 32) -> np.ndarray:
    """Approximate a radial k-space sampling mask on a Cartesian grid."""
    ny, nx = shape
    cy = (ny - 1) / 2
    cx = (nx - 1) / 2
    mask = np.zeros((ny, nx), dtype=bool)
    radius = int(np.ceil(np.sqrt(nx**2 + ny**2) / 2))

    for angle in np.linspace(0, np.pi, spokes, endpoint=False):
        for r in np.linspace(-radius, radius, 2 * radius + 1):
            y = int(round(cy + r * np.sin(angle)))
            x = int(round(cx + r * np.cos(angle)))
            if 0 <= y < ny and 0 <= x < nx:
                mask[y, x] = True
    return mask


def apply_sampling(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary sampling mask to k-space."""
    if kspace.shape != mask.shape:
        raise ValueError("kspace and mask must have the same shape")
    return np.where(mask, kspace, 0)
