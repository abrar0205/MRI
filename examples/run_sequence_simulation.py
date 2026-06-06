"""Run a reproducible MRI sequence simulation demo.

Usage:
    python examples/run_sequence_simulation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mri_lab.bloch import compare_sequence_contrast, rare_echo_train
from mri_lab.kspace import fft2c, radial_sampling_mask, undersampling_mask
from mri_lab.phantom import shepp_logan_like
from mri_lab.reconstruction import reconstruction_error, zero_filled_reconstruction
from mri_lab.sequences import RAREParams, rare_echo_times
from mri_lab.visualization import save_bar_chart, save_comparison, save_image


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    phantom = shepp_logan_like(size=128)
    save_image(phantom, output_dir / "01_phantom.png", title="Synthetic MRI phantom")

    sequence_signals = compare_sequence_contrast(t1=900, t2=80, pd=1.0)
    save_bar_chart(sequence_signals, output_dir / "02_sequence_contrast.png")

    rare_params = RAREParams(echo_spacing_ms=12, echo_train_length=12)
    echo_times = rare_echo_times(rare_params)
    echo_train = rare_echo_train(echo_times, tr_ms=rare_params.tr_ms)

    np.savetxt(
        output_dir / "03_rare_echo_train.csv",
        np.column_stack([echo_times, echo_train]),
        delimiter=",",
        header="echo_time_ms,relative_signal",
        comments="",
    )

    kspace = fft2c(phantom)
    save_image(np.log1p(np.abs(kspace)), output_dir / "04_kspace_log_magnitude.png", title="k-space log magnitude")

    cart_mask = undersampling_mask(kspace.shape, acceleration=4)
    radial_mask = radial_sampling_mask(kspace.shape, spokes=32)

    cart_recon = zero_filled_reconstruction(kspace, cart_mask)
    radial_recon = zero_filled_reconstruction(kspace, radial_mask)

    save_image(cart_mask.astype(float), output_dir / "05_cartesian_mask.png", title="Cartesian undersampling mask")
    save_image(radial_mask.astype(float), output_dir / "06_radial_mask.png", title="Radial-style sampling mask")

    save_comparison(
        {
            "Reference": phantom,
            "Cartesian R=4": cart_recon,
            "Radial-style": radial_recon,
        },
        output_dir / "07_reconstruction_comparison.png",
    )

    cart_error = reconstruction_error(phantom, cart_recon)
    radial_error = reconstruction_error(phantom, radial_recon)

    print("MRI simulation complete.")
    print(f"Cartesian R=4 NRMSE: {cart_error:.3f}")
    print(f"Radial-style NRMSE: {radial_error:.3f}")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
