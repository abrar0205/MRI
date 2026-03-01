"""Tests for the Bloch equation simulator."""

import numpy as np
import pytest

from mri_sim.bloch import (
    BlochSimulator,
    rotation_x,
    rotation_y,
    rotation_z,
    rf_rotation,
    relaxation,
    GAMMA_RAD,
)
from mri_sim.phantom import Phantom, single_voxel_phantom


class TestRotationMatrices:
    """Test rotation matrix construction."""

    def test_rotation_x_identity(self):
        R = rotation_x(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_y_identity(self):
        R = rotation_y(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_z_identity(self):
        R = rotation_z(0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_x_90deg(self):
        R = rotation_x(np.pi / 2)
        v = np.array([0, 0, 1])
        result = R @ v
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-10)

    def test_rotation_y_90deg(self):
        R = rotation_y(np.pi / 2)
        v = np.array([0, 0, 1])
        result = R @ v
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_rotation_z_90deg(self):
        R = rotation_z(np.pi / 2)
        v = np.array([1, 0, 0])
        result = R @ v
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_rotation_preserves_magnitude(self):
        v = np.array([0.5, 0.3, 0.8])
        for angle in [0.1, np.pi / 4, np.pi, 2 * np.pi]:
            for rot_fn in [rotation_x, rotation_y, rotation_z]:
                result = rot_fn(angle) @ v
                np.testing.assert_allclose(
                    np.linalg.norm(result), np.linalg.norm(v), atol=1e-10
                )

    def test_rf_rotation_90deg_x(self):
        """90° pulse about x-axis should tip Mz to -My."""
        R = rf_rotation(np.pi / 2, phase=0.0)
        M = np.array([0, 0, 1])
        result = R @ M
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-10)

    def test_rf_rotation_180deg(self):
        """180° pulse should invert Mz."""
        R = rf_rotation(np.pi, phase=0.0)
        M = np.array([0, 0, 1])
        result = R @ M
        np.testing.assert_allclose(result, [0, 0, -1], atol=1e-10)


class TestRelaxation:
    """Test relaxation decay factors."""

    def test_no_decay_at_zero_time(self):
        E1, E2 = relaxation(0.0, T1=1.0, T2=0.1)
        assert E1 == pytest.approx(1.0)
        assert E2 == pytest.approx(1.0)

    def test_exponential_decay(self):
        E1, E2 = relaxation(1.0, T1=1.0, T2=0.1)
        assert E1 == pytest.approx(np.exp(-1))
        assert E2 == pytest.approx(np.exp(-10))

    def test_array_input(self):
        T1 = np.array([1.0, 2.0])
        T2 = np.array([0.1, 0.2])
        E1, E2 = relaxation(0.5, T1, T2)
        np.testing.assert_allclose(E1, np.exp(-0.5 / T1))
        np.testing.assert_allclose(E2, np.exp(-0.5 / T2))


class TestBlochSimulator:
    """Test the Bloch simulator core functionality."""

    def test_initial_equilibrium(self):
        phantom = single_voxel_phantom(PD=1.0)
        sim = BlochSimulator(phantom)
        np.testing.assert_allclose(sim.M, [[0, 0, 1]], atol=1e-15)

    def test_90deg_excitation(self):
        """90° pulse should tip Mz to transverse plane."""
        phantom = single_voxel_phantom(PD=1.0)
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi / 2, phase=0.0)
        # After 90° about x: Mz -> -My
        assert abs(sim.M[0, 2]) < 1e-10  # Mz ≈ 0
        assert abs(sim.M[0, 1] + 1.0) < 1e-10  # My ≈ -1

    def test_180deg_inversion(self):
        """180° pulse should invert Mz."""
        phantom = single_voxel_phantom(PD=1.0)
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi, phase=0.0)
        np.testing.assert_allclose(sim.M[0, 2], -1.0, atol=1e-10)

    def test_t1_recovery(self):
        """After inversion, Mz should recover towards PD via T1."""
        phantom = single_voxel_phantom(PD=1.0, T1=1.0, T2=0.5)
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi, phase=0.0)  # Invert
        sim.apply_relaxation_and_precession(1.0)  # Wait 1*T1
        # Mz = PD * (1 - 2*exp(-t/T1))
        expected_Mz = 1.0 * (1 - 2 * np.exp(-1))
        assert sim.M[0, 2] == pytest.approx(expected_Mz, abs=1e-10)

    def test_t2_decay(self):
        """Transverse magnetisation should decay with T2."""
        phantom = single_voxel_phantom(PD=1.0, T1=10.0, T2=0.1)
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi / 2, phase=0.0)

        M_transverse_0 = np.sqrt(sim.M[0, 0] ** 2 + sim.M[0, 1] ** 2)
        sim.apply_relaxation_and_precession(0.1)  # Wait 1*T2
        M_transverse_1 = np.sqrt(sim.M[0, 0] ** 2 + sim.M[0, 1] ** 2)

        expected_ratio = np.exp(-1)
        assert M_transverse_1 / M_transverse_0 == pytest.approx(
            expected_ratio, abs=1e-5
        )

    def test_readout_signal(self):
        """After 90° excitation, readout should return non-zero signal."""
        phantom = single_voxel_phantom(PD=1.0)
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi / 2, phase=0.0)
        signal = sim.readout()
        assert abs(signal) > 0.9  # Should be close to PD

    def test_reset(self):
        phantom = single_voxel_phantom(PD=0.8)
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi / 2)
        sim.reset()
        np.testing.assert_allclose(sim.M, [[0, 0, 0.8]], atol=1e-15)

    def test_gradient_precession(self):
        """A gradient should cause phase accrual proportional to position."""
        phantom = Phantom(
            x=[0.01], y=[0.0], z=[0.0],
            PD=[1.0], T1=[100.0], T2=[100.0],
        )
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi / 2, phase=0.0)

        # Record initial phase
        phase_0 = np.angle(sim.M[0, 0] + 1j * sim.M[0, 1])

        # Apply gradient
        Gx = 10e-3  # 10 mT/m
        dt = 1e-3  # 1 ms
        sim.apply_relaxation_and_precession(dt, gradients=(Gx, 0, 0))

        phase_1 = np.angle(sim.M[0, 0] + 1j * sim.M[0, 1])
        expected_phase_change = GAMMA_RAD * Gx * 0.01 * dt
        actual_change = phase_1 - phase_0

        # Wrap to [-pi, pi]
        actual_change = np.angle(np.exp(1j * actual_change))
        expected_phase_change = np.angle(np.exp(1j * expected_phase_change))

        assert actual_change == pytest.approx(expected_phase_change, abs=1e-6)

    def test_spin_echo_refocusing(self):
        """180° pulse should refocus dephasing from B0 inhomogeneity."""
        phantom = Phantom(
            x=[0.0], y=[0.0], z=[0.0],
            PD=[1.0], T1=[100.0], T2=[100.0], B0=[1e-6],
        )
        sim = BlochSimulator(phantom)

        # 90° excitation
        sim.apply_rf(np.pi / 2, phase=0.0)
        M_init = np.sqrt(sim.M[0, 0] ** 2 + sim.M[0, 1] ** 2)

        # Free precession (dephasing)
        sim.apply_relaxation_and_precession(10e-3)

        # 180° refocusing
        sim.apply_rf(np.pi, phase=0.0)

        # Same duration of free precession (rephasing)
        sim.apply_relaxation_and_precession(10e-3)

        M_echo = np.sqrt(sim.M[0, 0] ** 2 + sim.M[0, 1] ** 2)

        # Echo magnitude should be close to initial (with minimal T2 decay)
        assert M_echo / M_init > 0.99


class TestMultiVoxel:
    """Test multi-voxel simulation."""

    def test_two_voxels_different_position(self):
        """Two voxels at different positions should acquire different phases."""
        phantom = Phantom(
            x=[0.01, -0.01], y=[0.0, 0.0], z=[0.0, 0.0],
            PD=[1.0, 1.0], T1=[100.0, 100.0], T2=[100.0, 100.0],
        )
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi / 2, phase=0.0)
        sim.apply_relaxation_and_precession(1e-6, gradients=(10e-3, 0, 0))

        phase_0 = np.angle(sim.M[0, 0] + 1j * sim.M[0, 1])
        phase_1 = np.angle(sim.M[1, 0] + 1j * sim.M[1, 1])

        # With symmetric positions and very short dt, the phase offsets from
        # the initial -π/2 should be equal and opposite.
        base_phase = -np.pi / 2
        delta_0 = np.angle(np.exp(1j * (phase_0 - base_phase)))
        delta_1 = np.angle(np.exp(1j * (phase_1 - base_phase)))
        assert abs(delta_0 + delta_1) < 1e-6

    def test_different_T1_contrast(self):
        """Voxels with different T1 should show different Mz recovery."""
        phantom = Phantom(
            x=[0.0, 0.0], y=[0.0, 0.0], z=[0.0, 0.0],
            PD=[1.0, 1.0], T1=[0.5, 2.0], T2=[0.1, 0.1],
        )
        sim = BlochSimulator(phantom)
        sim.apply_rf(np.pi, phase=0.0)  # Invert
        sim.apply_relaxation_and_precession(0.5)  # Wait

        # Shorter T1 recovers faster
        assert sim.M[0, 2] > sim.M[1, 2]
