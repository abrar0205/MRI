import numpy as np

from mri_lab.bloch import compare_sequence_contrast, simulate_single_echo
from mri_lab.kspace import fft2c, ifft2c, undersampling_mask
from mri_lab.phantom import shepp_logan_like
from mri_lab.reconstruction import reconstruction_error, zero_filled_reconstruction


def test_phantom_shape_and_range():
    phantom = shepp_logan_like(size=64)
    assert phantom.shape == (64, 64)
    assert phantom.min() >= 0
    assert phantom.max() <= 1


def test_fft_round_trip():
    image = shepp_logan_like(size=32)
    recovered = np.abs(ifft2c(fft2c(image)))
    assert np.allclose(image, recovered, atol=1e-10)


def test_undersampling_reconstruction_runs():
    image = shepp_logan_like(size=32)
    kspace = fft2c(image)
    mask = undersampling_mask(kspace.shape, acceleration=4)
    recon = zero_filled_reconstruction(kspace, mask)
    assert recon.shape == image.shape
    assert reconstruction_error(image, recon) >= 0


def test_bloch_signal_outputs():
    signal = simulate_single_echo()
    assert isinstance(signal, complex)

    contrast = compare_sequence_contrast()
    assert {"GRE/FLASH", "Spin Echo", "bSSFP"} <= set(contrast.keys())
    assert all(value >= 0 for value in contrast.values())
