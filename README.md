# specflux-calibrator

A Python tool to calibrate FITS spectra to absolute flux using AB magnitudes and filter transmission curves.

## Features

- Reads 1D spectra in FITS format.
- Uses filter transmission curves (CSV files with wavelength in nm and transmission in %).
- Converts spectra from arbitrary units to absolute flux (erg/s/cm²/Å) based on a user-provided AB magnitude.
- Saves calibrated spectra as new FITS files, preserving original headers.
- Uses `specutils`, `astropy`, and `scipy`.

## Installation

Make sure you have Python 3 and the following packages installed:

```bash
pip install numpy astropy specutils scipy
```

## Usage

```bash
python calibrate_spectrum.py path/to/spectrum.fits filter_name magnitude_v
```

- `path/to/spectrum.fits`: input spectrum in FITS format.
- `filter_name`: name of the filter CSV file (located in `filters/` folder), e.g. `bessel_v`.
- `magnitude_v`: AB magnitude to calibrate the spectrum.

Example:

```bash
python calibrate_spectrum.py data/spec.fits bessel_v 14.2
```

The calibrated spectrum will be saved as `data/spec-abs.fits`.

## Filter Transmission Files

Filter transmission files should be CSV files with two columns:

- Wavelength in nm (first column)
- Transmission in % (second column)

Place them in the `filters/` directory.


