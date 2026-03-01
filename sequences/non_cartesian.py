"""Non-Cartesian k-space trajectories and undersampling.

Implements radial and spiral k-space trajectories for MRI acquisition,
along with Cartesian undersampling patterns. These trajectories offer
advantages in motion robustness, incoherent aliasing, and compressed
sensing compatibility compared to standard Cartesian sampling.
"""

import numpy as np
from mri_sim.sequence import (
    Sequence,
    RFEvent,
    GradientEvent,
    ADCEvent,
    DelayEvent,
    readout_gradient,
    GAMMA_RAD,
)


def build_radial_sequence(
    fov=0.22,
    n_read=64,
    n_spokes=64,
    flip_angle_deg=30.0,
    TR=10e-3,
    readout_duration=None,
    grad_duration=0.5e-3,
):
    """Build a 2-D radial acquisition sequence.

    Each spoke passes through the centre of k-space at a different angle,
    providing uniform angular coverage. The readout gradient direction
    rotates with each spoke.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_read : int
        Number of readout samples per spoke.
    n_spokes : int
        Number of radial spokes (projections).
    flip_angle_deg : float
        Flip angle in degrees.
    TR : float
        Repetition time in seconds.
    readout_duration : float or None
        Readout duration in seconds.
    grad_duration : float
        Pre-winder/rewinder gradient duration in seconds.

    Returns
    -------
    Sequence
        Configured radial acquisition sequence.
    """
    if readout_duration is None:
        readout_duration = n_read * 10e-6

    seq = Sequence(name="Radial_2D")
    flip_angle = np.radians(flip_angle_deg)

    # Base readout gradient amplitude
    G_base = readout_gradient(fov, n_read, readout_duration)

    # Golden angle increment for incoherent sampling
    golden_angle = np.pi * (np.sqrt(5) - 1) / 2  # ≈ 111.25°

    for spoke in range(n_spokes):
        angle = spoke * golden_angle

        # Gradient components along rotated axes
        Gx = G_base * np.cos(angle)
        Gy = G_base * np.sin(angle)

        # === RF excitation ===
        seq.add_event(RFEvent(flip_angle))

        # === Pre-winder: move to -k_max along spoke direction ===
        Gx_pre = -Gx / 2 * (readout_duration / grad_duration)
        Gy_pre = -Gy / 2 * (readout_duration / grad_duration)
        seq.add_event(GradientEvent(grad_duration, gx=Gx_pre, gy=Gy_pre))

        # === Readout with ADC along spoke ===
        seq.add_event(ADCEvent(n_read, readout_duration, gx=Gx, gy=Gy))

        # === Rewinder ===
        seq.add_event(GradientEvent(grad_duration, gx=Gx_pre, gy=Gy_pre))

        # === TR delay ===
        time_used = 2 * grad_duration + readout_duration
        delay_tr = TR - time_used
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

    return seq


def build_spiral_sequence(
    fov=0.22,
    n_read=64,
    n_interleaves=16,
    n_samples_per_interleave=256,
    flip_angle_deg=30.0,
    TR=20e-3,
    readout_duration=None,
):
    """Build a 2-D spiral acquisition sequence.

    Uses an Archimedean spiral trajectory where each interleaf is rotated
    by 2π/n_interleaves. The spiral winds outward from the k-space centre.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_read : int
        Nominal spatial resolution in pixels.
    n_interleaves : int
        Number of spiral interleaves.
    n_samples_per_interleave : int
        Number of readout samples per interleaf.
    flip_angle_deg : float
        Flip angle in degrees.
    TR : float
        Repetition time in seconds.
    readout_duration : float or None
        Total readout duration per interleaf.

    Returns
    -------
    Sequence
        Configured spiral acquisition sequence.
    """
    if readout_duration is None:
        readout_duration = n_samples_per_interleave * 10e-6

    seq = Sequence(name="Spiral_2D")
    flip_angle = np.radians(flip_angle_deg)

    k_max = n_read / (2 * fov)  # Maximum k-space radius

    dt = readout_duration / n_samples_per_interleave

    for interleaf in range(n_interleaves):
        rotation = 2 * np.pi * interleaf / n_interleaves

        # === RF excitation ===
        seq.add_event(RFEvent(flip_angle))

        # Compute spiral gradient waveform
        # Archimedean spiral: k(t) = k_max * (t/T) * exp(j * n_turns * 2π * t/T)
        n_turns = n_read / (2 * n_interleaves)
        t = np.linspace(0, 1, n_samples_per_interleave + 1)

        # k-space positions along spiral
        r = k_max * t
        theta = 2 * np.pi * n_turns * t + rotation
        kx = r * np.cos(theta)
        ky = r * np.sin(theta)

        # Gradients = dk/dt / gamma
        dkx = np.diff(kx) / dt
        dky = np.diff(ky) / dt

        # Convert k-space velocity to gradient amplitude
        # dk/dt = gamma/(2*pi) * G  =>  G = dk/dt * 2*pi / gamma
        Gx_arr = dkx * 2 * np.pi / GAMMA_RAD
        Gy_arr = dky * 2 * np.pi / GAMMA_RAD

        # For simplicity, use average gradient per sample as ADC event
        # with varying gradient (approximated by segments)
        n_seg = min(8, n_samples_per_interleave)
        samples_per_seg = n_samples_per_interleave // n_seg
        seg_duration = samples_per_seg * dt

        for seg in range(n_seg):
            start = seg * samples_per_seg
            end = start + samples_per_seg
            Gx_avg = float(np.mean(Gx_arr[start:end]))
            Gy_avg = float(np.mean(Gy_arr[start:end]))
            seq.add_event(ADCEvent(samples_per_seg, seg_duration,
                                   gx=Gx_avg, gy=Gy_avg))

        # === TR delay ===
        delay_tr = TR - readout_duration
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

    return seq


def cartesian_undersampling_mask(n_phase, acceleration_factor=2,
                                 acs_lines=8):
    """Generate a Cartesian undersampling mask.

    Selects every R-th phase-encode line while keeping a fully-sampled
    auto-calibration signal (ACS) region in the centre of k-space.

    Parameters
    ----------
    n_phase : int
        Total number of phase-encode lines.
    acceleration_factor : int
        Undersampling factor R.
    acs_lines : int
        Number of fully-sampled ACS lines in the centre.

    Returns
    -------
    mask : np.ndarray
        Boolean mask, shape (n_phase,). True = sampled.
    """
    mask = np.zeros(n_phase, dtype=bool)

    # Regular undersampling
    mask[::acceleration_factor] = True

    # Fully-sampled ACS region
    centre = n_phase // 2
    half_acs = acs_lines // 2
    mask[centre - half_acs: centre + half_acs] = True

    return mask


def random_undersampling_mask(n_phase, acceleration_factor=4,
                              acs_lines=8, seed=42):
    """Generate a random Cartesian undersampling mask.

    Randomly selects phase-encode lines with higher probability near the
    k-space centre (variable density), suitable for compressed sensing.

    Parameters
    ----------
    n_phase : int
        Total number of phase-encode lines.
    acceleration_factor : int
        Nominal undersampling factor.
    acs_lines : int
        Number of fully-sampled ACS lines.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mask : np.ndarray
        Boolean mask, shape (n_phase,).
    """
    rng = np.random.RandomState(seed)
    n_target = min(max(n_phase // acceleration_factor, acs_lines), n_phase)

    # Variable density: higher probability near centre
    line_indices = np.arange(n_phase)
    centre = n_phase / 2.0
    # Probability proportional to 1/(1 + distance_from_centre)
    probs = 1.0 / (1.0 + np.abs(line_indices - centre))
    probs /= probs.sum()

    selected = rng.choice(n_phase, size=n_target, replace=False, p=probs)

    mask = np.zeros(n_phase, dtype=bool)
    mask[selected] = True

    # Ensure ACS lines are included
    half_acs = acs_lines // 2
    c = n_phase // 2
    mask[c - half_acs: c + half_acs] = True

    return mask


if __name__ == "__main__":
    seq_rad = build_radial_sequence()
    print(seq_rad)
    print(f"Radial ADC samples: {seq_rad.n_adc_samples}")

    seq_spi = build_spiral_sequence()
    print(seq_spi)
    print(f"Spiral ADC samples: {seq_spi.n_adc_samples}")

    mask_reg = cartesian_undersampling_mask(64, acceleration_factor=2)
    print(f"Regular undersampling: {mask_reg.sum()}/{len(mask_reg)} lines")

    mask_rand = random_undersampling_mask(64, acceleration_factor=4)
    print(f"Random undersampling: {mask_rand.sum()}/{len(mask_rand)} lines")
