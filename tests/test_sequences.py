"""Tests for pulse sequence definitions and builders."""

import numpy as np
import pytest

from mri_sim.sequence import (
    Sequence,
    RFEvent,
    GradientEvent,
    ADCEvent,
    DelayEvent,
    gradient_amplitude,
    readout_gradient,
    phase_encode_gradient,
)
from sequences.gre import build_gre_sequence
from sequences.fast_gre import build_fast_gre_sequence
from sequences.spin_echo import build_spin_echo_sequence
from sequences.rare import build_rare_sequence
from sequences.bssfp import build_bssfp_sequence
from sequences.non_cartesian import (
    build_radial_sequence,
    build_spiral_sequence,
    cartesian_undersampling_mask,
    random_undersampling_mask,
)


class TestSequenceEvents:
    """Test individual sequence event classes."""

    def test_rf_event(self):
        rf = RFEvent(np.pi / 2, phase=np.pi)
        assert rf.flip_angle == pytest.approx(np.pi / 2)
        assert rf.phase == pytest.approx(np.pi)

    def test_gradient_event(self):
        g = GradientEvent(1e-3, gx=0.01, gy=0.02, gz=0.0)
        assert g.duration == pytest.approx(1e-3)
        assert g.gx == pytest.approx(0.01)

    def test_adc_event(self):
        adc = ADCEvent(128, 1.28e-3, gx=0.01)
        assert adc.n_samples == 128
        assert adc.gx == pytest.approx(0.01)

    def test_delay_event(self):
        d = DelayEvent(5e-3)
        assert d.duration == pytest.approx(5e-3)


class TestSequenceContainer:
    """Test the Sequence container."""

    def test_empty_sequence(self):
        seq = Sequence("test")
        assert seq.n_events == 0
        assert seq.n_adc_samples == 0

    def test_add_events(self):
        seq = Sequence("test")
        seq.add_event(RFEvent(np.pi / 2))
        seq.add_event(ADCEvent(64, 0.64e-3))
        assert seq.n_events == 2
        assert seq.n_adc_samples == 64

    def test_add_block(self):
        seq = Sequence("test")
        seq.add_block(
            RFEvent(np.pi / 2),
            GradientEvent(1e-3, gx=0.01),
            ADCEvent(32, 0.32e-3),
        )
        assert seq.n_events == 3


class TestGradientHelpers:
    """Test gradient calculation utilities."""

    def test_readout_gradient_positive(self):
        G = readout_gradient(fov=0.22, n_samples=64, duration=0.64e-3)
        assert G > 0

    def test_phase_encode_gradient_symmetry(self):
        G_pos = phase_encode_gradient(0.22, 64, 10, 1e-3)
        G_neg = phase_encode_gradient(0.22, 64, -10, 1e-3)
        assert G_pos == pytest.approx(-G_neg)

    def test_phase_encode_zero_at_centre(self):
        G = phase_encode_gradient(0.22, 64, 0, 1e-3)
        assert G == pytest.approx(0.0)


class TestGRESequence:
    """Test GRE sequence builder."""

    def test_builds_correctly(self):
        seq = build_gre_sequence(n_read=32, n_phase=16)
        assert seq.n_adc_samples == 32 * 16
        assert seq.name == "GRE_2D"

    def test_has_rf_events(self):
        seq = build_gre_sequence(n_read=16, n_phase=8)
        rf_count = sum(1 for e in seq.events if isinstance(e, RFEvent))
        assert rf_count == 8  # One RF per phase-encode line


class TestFastGRESequence:
    """Test Fast GRE sequence builder."""

    def test_builds_correctly(self):
        seq = build_fast_gre_sequence(n_read=32, n_phase=16)
        assert seq.n_adc_samples == 32 * 16
        assert seq.name == "FastGRE_2D"

    def test_rf_spoiling_phases_differ(self):
        """RF phases should differ between TRs due to spoiling."""
        seq = build_fast_gre_sequence(n_read=16, n_phase=4)
        rf_events = [e for e in seq.events if isinstance(e, RFEvent)]
        phases = [e.phase for e in rf_events]
        # Phases should not all be the same
        assert len(set(round(p, 6) for p in phases)) > 1


class TestSpinEchoSequence:
    """Test Spin Echo sequence builder."""

    def test_builds_correctly(self):
        seq = build_spin_echo_sequence(n_read=32, n_phase=16)
        assert seq.n_adc_samples == 32 * 16

    def test_has_90_and_180_pulses(self):
        seq = build_spin_echo_sequence(n_read=16, n_phase=4)
        rf_events = [e for e in seq.events if isinstance(e, RFEvent)]
        # Each TR has 90° + 180° = 2 RF events
        assert len(rf_events) == 4 * 2

        flip_angles = sorted(set(round(e.flip_angle, 4) for e in rf_events))
        assert len(flip_angles) == 2
        assert flip_angles[0] == pytest.approx(np.pi / 2, abs=0.01)
        assert flip_angles[1] == pytest.approx(np.pi, abs=0.01)


class TestRARESequence:
    """Test RARE (TSE) sequence builder."""

    def test_builds_correctly(self):
        seq = build_rare_sequence(n_read=32, n_phase=16, echo_train_length=4)
        assert seq.n_adc_samples == 32 * 16

    def test_fewer_excitations(self):
        """RARE should have fewer 90° pulses than SE (one per shot)."""
        seq = build_rare_sequence(n_read=16, n_phase=16, echo_train_length=8)
        rf_events = [e for e in seq.events if isinstance(e, RFEvent)]
        # 90° pulses (flip ≈ π/2)
        excitations = [e for e in rf_events
                       if abs(e.flip_angle - np.pi / 2) < 0.1]
        n_shots = int(np.ceil(16 / 8))
        assert len(excitations) == n_shots


class TestBSSFPSequence:
    """Test bSSFP sequence builder."""

    def test_builds_correctly(self):
        seq = build_bssfp_sequence(n_read=32, n_phase=16)
        assert seq.n_adc_samples == 32 * 16

    def test_has_prep_pulse(self):
        """First RF should be α/2 preparation pulse."""
        seq = build_bssfp_sequence(n_read=16, n_phase=8, flip_angle_deg=30.0)
        first_rf = next(e for e in seq.events if isinstance(e, RFEvent))
        assert first_rf.flip_angle == pytest.approx(np.radians(15.0))


class TestRadialSequence:
    """Test radial acquisition sequence."""

    def test_builds_correctly(self):
        seq = build_radial_sequence(n_read=32, n_spokes=16)
        assert seq.n_adc_samples == 32 * 16

    def test_spoke_count(self):
        seq = build_radial_sequence(n_read=32, n_spokes=10)
        rf_count = sum(1 for e in seq.events if isinstance(e, RFEvent))
        assert rf_count == 10


class TestSpiralSequence:
    """Test spiral acquisition sequence."""

    def test_builds_correctly(self):
        seq = build_spiral_sequence(n_interleaves=4)
        assert seq.n_adc_samples > 0


class TestUndersamplingMasks:
    """Test undersampling mask generation."""

    def test_regular_undersampling(self):
        mask = cartesian_undersampling_mask(64, acceleration_factor=2, acs_lines=8)
        assert mask.sum() < 64
        assert mask.sum() > 0
        # ACS lines should be sampled
        centre = 32
        assert all(mask[centre - 4: centre + 4])

    def test_random_undersampling(self):
        mask = random_undersampling_mask(64, acceleration_factor=4, acs_lines=8)
        assert mask.sum() < 64
        assert mask.sum() > 0

    def test_acceleration_reduces_samples(self):
        mask_2x = cartesian_undersampling_mask(64, acceleration_factor=2)
        mask_4x = cartesian_undersampling_mask(64, acceleration_factor=4)
        assert mask_4x.sum() < mask_2x.sum()
