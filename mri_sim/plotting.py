"""Plotting and visualisation utilities for MRI sequences and images."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt


def plot_signal(signal, title="ADC Signal", n_read=None):
    """Plot real and imaginary parts of the acquired signal.

    Parameters
    ----------
    signal : np.ndarray
        Complex-valued signal.
    title : str
        Plot title.
    n_read : int or None
        If provided, adds vertical grid lines every n_read samples.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal.real, label="Real", alpha=0.8)
    ax.plot(signal.imag, label="Imag", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Signal")
    ax.legend()
    if n_read is not None:
        ticks = np.arange(0, len(signal), n_read)
        ax.set_xticks(ticks)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kspace(kspace, title="k-space"):
    """Plot the magnitude of 2-D k-space data.

    Parameters
    ----------
    kspace : np.ndarray
        2-D complex k-space matrix.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.abs(kspace), cmap="gray", aspect="auto")
    axes[0].set_title(f"{title} magnitude")
    axes[1].imshow(np.log1p(np.abs(kspace)), cmap="gray", aspect="auto")
    axes[1].set_title(f"{title} log-magnitude")
    fig.tight_layout()
    return fig


def plot_reconstruction(image, kspace=None, phantom_PD=None, title="Reconstruction"):
    """Plot reconstructed image alongside k-space and ground truth.

    Parameters
    ----------
    image : np.ndarray
        2-D magnitude image.
    kspace : np.ndarray or None
        2-D k-space matrix for display.
    phantom_PD : np.ndarray or None
        Ground-truth proton density map for comparison.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_cols = 1 + (kspace is not None) + (phantom_PD is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    idx = 0
    if kspace is not None:
        axes[idx].imshow(np.log1p(np.abs(kspace)), cmap="gray", aspect="auto")
        axes[idx].set_title("k-space (log)")
        idx += 1

    axes[idx].imshow(image, cmap="gray", aspect="auto")
    axes[idx].set_title(title)
    idx += 1

    if phantom_PD is not None:
        axes[idx].imshow(phantom_PD, cmap="gray", aspect="auto")
        axes[idx].set_title("Phantom PD")

    fig.tight_layout()
    return fig


def plot_kspace_trajectory(kspace_loc, title="k-space trajectory"):
    """Plot 2-D k-space trajectory.

    Parameters
    ----------
    kspace_loc : np.ndarray
        k-space locations, shape (n_samples, 2).
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(kspace_loc[:, 0], kspace_loc[:, 1], "b-", alpha=0.4, linewidth=0.5)
    ax.plot(kspace_loc[:, 0], kspace_loc[:, 1], "r.", markersize=1)
    ax.set_xlabel("kx (1/m)")
    ax.set_ylabel("ky (1/m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
