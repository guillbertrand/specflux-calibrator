# specflux-calibrator

A simple Python tool to calibrate astronomical spectra in absolute flux using the AB magnitude system.

## Features

- Calibrate spectra in FITS format using a filter transmission curve (CSV).
- Uses the AB magnitude system for flux calibration.
- Supports Bessel V or any other filter transmission provided as CSV.
- Preserves original FITS header metadata.
- Optionally plot the calibrated spectrum with matplotlib.

## Usage

```bash
python calibrate_spectrum.py <spectrum.fits> <filter.csv> <magnitude_V> [--plot]
```

- `<spectrum.fits>` : Path to the input spectrum FITS file (WCS1D format).
- `<filter.csv>` : Path to CSV file with filter transmission. CSV format:
  - 1st column: wavelength in nm
  - 2nd column: transmission in %
- `<magnitude_V>` : Target magnitude in the V filter (AB system).
- `[--plot]` : Optional flag to display a plot of the calibrated spectrum.

Example:

```bash
python calibrate_spectrum.py data/spectrum.fits filters/bessel_v.csv 12.3 --plot
```

## Output

- The calibrated spectrum is saved as a new FITS file named `<originalname>-abs.fits`.
- The FITS file contains the calibrated flux in units of `ergs/cm2/s/Å`.
- The header metadata from the original file is preserved.
- A comment is added noting the filter and magnitude used.

## Dependencies

- Python 3
- numpy
- astropy
- specutils
- scipy
- matplotlib (only needed if using `--plot` option)

Install with:

```bash
pip install numpy astropy specutils scipy matplotlib
```

---
