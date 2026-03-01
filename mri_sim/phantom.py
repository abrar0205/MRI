"""Phantom definitions for MRI simulation.

Provides configurable digital phantoms with spatially varying tissue
parameters (PD, T1, T2, B0) for use with the Bloch equation simulator.
Includes simple geometric phantoms and a numerical brain phantom.
"""

import numpy as np


class Phantom:
    """Multi-voxel digital phantom with tissue properties.

    Parameters
    ----------
    x : np.ndarray
        Voxel x-positions in metres, shape (n_voxels,).
    y : np.ndarray
        Voxel y-positions in metres, shape (n_voxels,).
    z : np.ndarray
        Voxel z-positions in metres, shape (n_voxels,).
    PD : np.ndarray
        Proton density (arbitrary units), shape (n_voxels,).
    T1 : np.ndarray
        T1 relaxation time in seconds, shape (n_voxels,).
    T2 : np.ndarray
        T2 relaxation time in seconds, shape (n_voxels,).
    B0 : np.ndarray
        B0 inhomogeneity in Tesla, shape (n_voxels,).
    """

    def __init__(self, x, y, z, PD, T1, T2, B0=None):
        self.x = np.asarray(x, dtype=np.float64).ravel()
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.z = np.asarray(z, dtype=np.float64).ravel()
        self.PD = np.asarray(PD, dtype=np.float64).ravel()
        self.T1 = np.asarray(T1, dtype=np.float64).ravel()
        self.T2 = np.asarray(T2, dtype=np.float64).ravel()
        if B0 is not None:
            self.B0 = np.asarray(B0, dtype=np.float64).ravel()
        else:
            self.B0 = np.zeros_like(self.x)

        # Validation
        n = self.x.size
        for attr_name in ("y", "z", "PD", "T1", "T2", "B0"):
            if getattr(self, attr_name).size != n:
                raise ValueError(
                    f"All arrays must have the same length; "
                    f"'{attr_name}' has size {getattr(self, attr_name).size} vs {n}"
                )

    @property
    def n_voxels(self):
        """Number of voxels."""
        return self.x.size


def single_voxel_phantom(PD=1.0, T1=1.0, T2=0.1, B0=0.0, pos=(0.0, 0.0, 0.0)):
    """Create a single-voxel phantom for basic signal testing.

    Parameters
    ----------
    PD : float
        Proton density.
    T1 : float
        T1 in seconds.
    T2 : float
        T2 in seconds.
    B0 : float
        B0 offset in Tesla.
    pos : tuple of float
        (x, y, z) position in metres.

    Returns
    -------
    Phantom
    """
    return Phantom(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        PD=[PD], T1=[T1], T2=[T2], B0=[B0],
    )


def grid_phantom(fov, n_pixels, PD=1.0, T1=1.0, T2=0.1, B0=0.0):
    """Create a uniform 2-D grid phantom.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_pixels : int
        Number of pixels along each dimension (square grid).
    PD, T1, T2, B0 : float
        Uniform tissue parameters.

    Returns
    -------
    Phantom
    """
    coords = np.linspace(-fov / 2, fov / 2, n_pixels, endpoint=False)
    coords += fov / (2 * n_pixels)  # centre pixels
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    x = xx.ravel()
    y = yy.ravel()
    z = np.zeros_like(x)
    n = x.size
    return Phantom(
        x=x, y=y, z=z,
        PD=np.full(n, PD),
        T1=np.full(n, T1),
        T2=np.full(n, T2),
        B0=np.full(n, B0),
    )


def numerical_brain_phantom(fov=0.22, n_pixels=64):
    """Generate a simplified numerical brain phantom.

    Creates a 2-D phantom with concentric elliptical regions that mimic
    white matter, grey matter, and CSF, each with distinct T1, T2, and
    PD values at 1.5 T.

    Parameters
    ----------
    fov : float
        Field of view in metres. Default 0.22 m (220 mm).
    n_pixels : int
        Grid size along each dimension.

    Returns
    -------
    Phantom
    """
    coords = np.linspace(-fov / 2, fov / 2, n_pixels, endpoint=False)
    coords += fov / (2 * n_pixels)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")

    # Normalised coordinates in [-0.5, 0.5]
    xn = xx / fov
    yn = yy / fov

    # Tissue masks using ellipse equations
    # Outer skull boundary
    skull = (xn / 0.45) ** 2 + (yn / 0.40) ** 2 <= 1.0
    # Grey matter ring
    grey = (xn / 0.40) ** 2 + (yn / 0.35) ** 2 <= 1.0
    # White matter core
    white = (xn / 0.28) ** 2 + (yn / 0.25) ** 2 <= 1.0
    # Ventricles (CSF)
    csf_left = ((xn + 0.06) / 0.04) ** 2 + (yn / 0.12) ** 2 <= 1.0
    csf_right = ((xn - 0.06) / 0.04) ** 2 + (yn / 0.12) ** 2 <= 1.0
    csf = csf_left | csf_right

    # Tissue parameters at 1.5 T (approximate)
    #                    PD    T1(s)   T2(s)
    # Background:        0     -       -
    # CSF:               1.0   4.0     2.0
    # Grey Matter:       0.86  1.33    0.11
    # White Matter:      0.77  0.83    0.08

    n = xx.size
    PD = np.zeros(n, dtype=np.float64)
    T1 = np.ones(n, dtype=np.float64) * 1.0  # default
    T2 = np.ones(n, dtype=np.float64) * 0.1  # default

    # Assign from outside in so inner regions overwrite outer
    # Skull/scalp boundary (thin ring)
    scalp = skull & ~grey
    PD[scalp.ravel()] = 0.3
    T1[scalp.ravel()] = 0.5
    T2[scalp.ravel()] = 0.04

    # Grey matter
    gm = grey & ~white
    PD[gm.ravel()] = 0.86
    T1[gm.ravel()] = 1.33
    T2[gm.ravel()] = 0.11

    # White matter
    wm = white & ~csf
    PD[wm.ravel()] = 0.77
    T1[wm.ravel()] = 0.83
    T2[wm.ravel()] = 0.08

    # CSF
    PD[csf.ravel()] = 1.0
    T1[csf.ravel()] = 4.0
    T2[csf.ravel()] = 2.0

    # Small B0 inhomogeneity: linear gradient across x
    B0 = xn.ravel() * 1e-6  # very small

    return Phantom(
        x=xx.ravel(), y=yy.ravel(), z=np.zeros(n),
        PD=PD, T1=T1, T2=T2, B0=B0,
    )
