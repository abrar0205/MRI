"""MRI reconstruction helpers."""

from __future__ import annotations

import numpy as np

from .kspace import apply_sampling, ifft2c


def zero_filled_reconstruction(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct an image from undersampled k-space using zero filling."""
    sampled = apply_sampling(kspace, mask)
    return np.abs(ifft2c(sampled))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image into [0, 1]."""
    image = np.asarray(image, dtype=float)
    image = image - image.min()
    max_value = image.max()
    if max_value > 0:
        image = image / max_value
    return image


def reconstruction_error(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Normalized root mean squared error."""
    reference = normalize_image(reference)
    reconstruction = normalize_image(reconstruction)
    numerator = np.linalg.norm(reference - reconstruction)
    denominator = np.linalg.norm(reference) + 1e-12
    return float(numerator / denominator)
