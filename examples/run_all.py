"""Run all MRI pulse sequence examples.

Demonstrates all implemented pulse sequences by building each sequence,
running a Bloch simulation on a numerical brain phantom, and reconstructing
the resulting MR images. This validates the complete pipeline from sequence
definition through signal acquisition to image formation.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mri_sim.bloch import BlochSimulator
from mri_sim.phantom import numerical_brain_phantom, single_voxel_phantom
from mri_sim.reconstruction import reconstruct_cartesian, reconstruct_nufft

from sequences.gre import build_gre_sequence
from sequences.fast_gre import build_fast_gre_sequence
from sequences.spin_echo import build_spin_echo_sequence
from sequences.rare import build_rare_sequence
from sequences.bssfp import build_bssfp_sequence
from sequences.non_cartesian import build_radial_sequence


def run_sequence_simulation(name, seq, phantom, n_read, n_phase, cartesian=True, fov=0.22):
    """Run a sequence simulation and reconstruct the image.

    Parameters
    ----------
    name : str
        Descriptive name for the sequence.
    seq : Sequence
        The pulse sequence to simulate.
    phantom : Phantom
        The phantom to simulate on.
    n_read : int
        Number of readout samples.
    n_phase : int
        Number of phase-encode lines.
    cartesian : bool
        Whether to use Cartesian reconstruction.
    fov : float
        Field of view in metres.

    Returns
    -------
    image : np.ndarray
        Reconstructed magnitude image.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Sequence: {seq}")
    print(f"  Phantom voxels: {phantom.n_voxels}")
    print(f"  Expected ADC samples: {seq.n_adc_samples}")

    sim = BlochSimulator(phantom)
    signal, kspace_loc = sim.run_sequence(seq)

    print(f"  Acquired {len(signal)} ADC samples")
    print(f"  Signal magnitude range: [{np.abs(signal).min():.6f}, {np.abs(signal).max():.6f}]")

    if cartesian:
        image, kspace = reconstruct_cartesian(signal, n_read, n_phase)
    else:
        image, kspace = reconstruct_nufft(signal, kspace_loc, n_read, fov)

    print(f"  Image shape: {image.shape}")
    print(f"  Image intensity range: [{image.min():.6f}, {image.max():.6f}]")
    print(f"  DONE")

    return image


def main():
    """Run all sequence demonstrations."""
    print("MRI Pulse Sequence Programming - Simulation Framework")
    print("=" * 60)

    # Use small phantom for fast demo
    fov = 0.22
    n_pixels = 16
    n_read = 16
    n_phase = 16

    phantom = numerical_brain_phantom(fov=fov, n_pixels=n_pixels)
    print(f"Phantom: {n_pixels}x{n_pixels} brain, FOV={fov*1e3:.0f}mm")
    print(f"Total voxels: {phantom.n_voxels}")

    # --- 1. GRE ---
    seq_gre = build_gre_sequence(
        fov=fov, n_read=n_read, n_phase=n_phase,
        flip_angle_deg=15, TE=5e-3, TR=20e-3,
    )
    img_gre = run_sequence_simulation("Gradient Recalled Echo (GRE)", seq_gre,
                                      phantom, n_read, n_phase, fov=fov)

    # --- 2. Fast GRE ---
    seq_fgre = build_fast_gre_sequence(
        fov=fov, n_read=n_read, n_phase=n_phase,
        flip_angle_deg=10, TE=3e-3, TR=10e-3,
    )
    img_fgre = run_sequence_simulation("Fast GRE (FLASH/SPGR)", seq_fgre,
                                       phantom, n_read, n_phase, fov=fov)

    # --- 3. Spin Echo ---
    seq_se = build_spin_echo_sequence(
        fov=fov, n_read=n_read, n_phase=n_phase,
        TE=15e-3, TR=200e-3,
    )
    img_se = run_sequence_simulation("Spin Echo", seq_se,
                                     phantom, n_read, n_phase, fov=fov)

    # --- 4. RARE ---
    seq_rare = build_rare_sequence(
        fov=fov, n_read=n_read, n_phase=n_phase,
        echo_train_length=4, esp=10e-3, TR=500e-3,
    )
    img_rare = run_sequence_simulation("RARE (Turbo Spin Echo)", seq_rare,
                                       phantom, n_read, n_phase, fov=fov)

    # --- 5. bSSFP ---
    seq_bssfp = build_bssfp_sequence(
        fov=fov, n_read=n_read, n_phase=n_phase,
        flip_angle_deg=30, TR=6e-3,
    )
    img_bssfp = run_sequence_simulation("Balanced SSFP (TrueFISP)", seq_bssfp,
                                        phantom, n_read, n_phase, fov=fov)

    # --- 6. Radial (non-Cartesian) ---
    seq_radial = build_radial_sequence(
        fov=fov, n_read=n_read, n_spokes=n_phase,
        flip_angle_deg=30, TR=10e-3,
    )
    img_radial = run_sequence_simulation("Radial (non-Cartesian)", seq_radial,
                                         phantom, n_read, n_phase,
                                         cartesian=False, fov=fov)

    print(f"\n{'='*60}")
    print("  All sequences completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
