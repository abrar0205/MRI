# MRI Pulse Sequence Simulation Lab

Academic Python project for MRI sequence concepts, signal modelling, k-space encoding, undersampling, and reconstruction.

## Highlights

- GRE / FLASH, Spin Echo, RARE-style, and bSSFP signal concepts
- Magnetization contrast modelling with TE, TR, T1, T2, and flip angle
- Centered FFT / inverse FFT k-space utilities
- Cartesian and radial-style sampling masks
- Zero-filled reconstruction and error metrics
- Synthetic phantom generation
- Example script and unit tests

## Run

```bash
pip install -r requirements.txt
pip install -e .
python examples/run_sequence_simulation.py
```

## Example

```python
from mri_lab.phantom import shepp_logan_like
from mri_lab.kspace import fft2c, undersampling_mask
from mri_lab.reconstruction import zero_filled_reconstruction

phantom = shepp_logan_like(size=128)
kspace = fft2c(phantom)
mask = undersampling_mask(kspace.shape, acceleration=4)
recon = zero_filled_reconstruction(kspace, mask)
```

## Academic context

This repository reflects coursework in MRI pulse sequence programming and simulation, including timing parameters, acquisition settings, signal formation, sequence comparison, k-space sampling, undersampling, reconstruction quality, and artifact analysis.
