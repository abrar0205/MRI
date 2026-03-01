"""Image reconstruction utilities.

Provides Cartesian (FFT-based) and non-uniform (NUFFT-based) k-space to
image reconstruction routines.
"""

import numpy as np
from scipy.interpolate import griddata


def reconstruct_cartesian(signal, n_read, n_phase):
    """Reconstruct a 2-D image from Cartesian k-space data via FFT.

    The input signal is assumed to be sampled on a regular Cartesian grid
    with *n_read* frequency-encode samples per phase-encode line and
    *n_phase* lines, ordered line by line from the most negative to the
    most positive phase-encode step.

    Parameters
    ----------
    signal : np.ndarray
        Complex-valued signal, shape (n_read * n_phase,).
    n_read : int
        Number of readout (frequency-encode) samples per line.
    n_phase : int
        Number of phase-encode lines.

    Returns
    -------
    image : np.ndarray
        Reconstructed magnitude image, shape (n_read, n_phase).
    kspace : np.ndarray
        k-space matrix, shape (n_read, n_phase).
    """
    signal = np.asarray(signal)
    if signal.size != n_read * n_phase:
        raise ValueError(
            f"signal length {signal.size} does not match "
            f"n_read * n_phase = {n_read * n_phase}"
        )
    # Reshape: each contiguous block of n_read samples is one phase-encode line
    kspace = signal.reshape(n_phase, n_read).T  # (n_read, n_phase)

    # 2-D FFT with fftshift for centred image
    spectrum = np.fft.fftshift(kspace)
    image = np.fft.fft2(spectrum)
    image = np.fft.fftshift(image)

    return np.abs(image), kspace


def reconstruct_nufft(signal, kspace_loc, n_pixels, fov):
    """Reconstruct a 2-D image from non-Cartesian k-space data.

    Uses gridding interpolation (via ``scipy.interpolate.griddata``) to
    resample non-uniformly distributed k-space data onto a Cartesian grid,
    followed by a standard 2-D FFT.

    Parameters
    ----------
    signal : np.ndarray
        Complex-valued signal, shape (n_samples,).
    kspace_loc : np.ndarray
        k-space sample locations, shape (n_samples, 2) for (kx, ky).
    n_pixels : int
        Reconstruction grid size.
    fov : float
        Field of view in metres.

    Returns
    -------
    image : np.ndarray
        Reconstructed magnitude image, shape (n_pixels, n_pixels).
    kspace_grid : np.ndarray
        Gridded k-space matrix, shape (n_pixels, n_pixels).
    """
    signal = np.asarray(signal)
    kspace_loc = np.asarray(kspace_loc)

    # Target Cartesian grid in k-space
    dk = 1.0 / fov
    k_max = n_pixels * dk / 2.0
    kx_grid = np.linspace(-k_max, k_max, n_pixels, endpoint=False)
    ky_grid = np.linspace(-k_max, k_max, n_pixels, endpoint=False)
    KX, KY = np.meshgrid(kx_grid, ky_grid, indexing="xy")

    # Grid the non-uniform data using linear interpolation
    kspace_real = griddata(
        kspace_loc, signal.real, (KX, KY), method="linear", fill_value=0.0
    )
    kspace_imag = griddata(
        kspace_loc, signal.imag, (KX, KY), method="linear", fill_value=0.0
    )
    kspace_grid = kspace_real + 1j * kspace_imag

    # 2-D FFT reconstruction
    spectrum = np.fft.fftshift(kspace_grid)
    image = np.fft.fft2(spectrum)
    image = np.fft.fftshift(image)

    return np.abs(image), kspace_grid
