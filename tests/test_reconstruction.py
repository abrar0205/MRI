"""Tests for image reconstruction utilities."""

import numpy as np
import pytest

from mri_sim.reconstruction import reconstruct_cartesian, reconstruct_nufft


class TestCartesianReconstruction:
    """Test FFT-based Cartesian reconstruction."""

    def test_point_source(self):
        """A single point in k-space should produce a uniform image."""
        n = 16
        kspace = np.zeros(n * n, dtype=np.complex128)
        kspace[n * n // 2 + n // 2] = 1.0  # centre of k-space
        image, ks = reconstruct_cartesian(kspace, n, n)
        # Uniform image -> all pixels same magnitude
        assert image.shape == (n, n)
        # A single point at centre means uniform magnitude after FFT
        assert np.std(image) / np.mean(image) < 0.01

    def test_output_shape(self):
        n_read, n_phase = 32, 16
        signal = np.random.randn(n_read * n_phase) + 1j * np.random.randn(n_read * n_phase)
        image, kspace = reconstruct_cartesian(signal, n_read, n_phase)
        assert image.shape == (n_read, n_phase)
        assert kspace.shape == (n_read, n_phase)

    def test_real_valued_kspace(self):
        """Real-valued k-space centred at DC should produce a real image."""
        n = 8
        kspace = np.zeros(n * n, dtype=np.complex128)
        kspace[n * n // 2 + n // 2] = 10.0
        image, _ = reconstruct_cartesian(kspace, n, n)
        assert np.all(image >= 0)


class TestNUFFTReconstruction:
    """Test gridding-based non-Cartesian reconstruction."""

    def test_output_shape(self):
        n = 16
        fov = 0.22
        n_samples = 100
        kspace_loc = np.random.uniform(-n / (2 * fov), n / (2 * fov),
                                       size=(n_samples, 2))
        signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        image, kgrid = reconstruct_nufft(signal, kspace_loc, n, fov)
        assert image.shape == (n, n)
        assert kgrid.shape == (n, n)

    def test_zero_signal_gives_zero_image(self):
        n = 16
        fov = 0.22
        n_samples = 50
        kspace_loc = np.random.uniform(-1, 1, size=(n_samples, 2))
        signal = np.zeros(n_samples, dtype=np.complex128)
        image, _ = reconstruct_nufft(signal, kspace_loc, n, fov)
        np.testing.assert_allclose(image, 0, atol=1e-10)
