# specflux-calibrator

A simple Python tool to calibrate astronomical spectra in absolute flux using either the **AB** or **Vega** magnitude system.

## Features

- Calibrate spectra in WCS1D FITS format using a filter transmission curve (CSV).
- Supports both **AB** and **Vega** photometric systems.
- Uses any filter (e.g., Bessel V) provided as a CSV transmission file.
- Computes synthetic flux and rescales the spectrum accordingly.
- Preserves original FITS header metadata.
- Adds system/magnitude/filter info as a comment in the output.
- Optionally plots the input filter or calibrated spectrum using matplotlib.

## Usage

```bash
python calibrate_spectrum.py <spectrum.fits> <filter.csv> <magnitude> [--system=ab|vega] [--plot]
```

## Parameters

- `<spectrum.fits>`: Path to the input spectrum FITS file (WCS1D format).
- `<filter.csv>`: Path to the CSV file containing filter transmission. Format:
  - Column 1: wavelength in nanometers (nm)
  - Column 2: transmission (fraction, between 0 and 1)
- `<magnitude>`: Target magnitude of the source in the chosen system.
- `--system=ab|vega` *(optional)*: Selects the photometric system. Defaults to `ab`.
- `--plot` *(optional)*: Displays a plot of the filter transmission and/or calibrated spectrum.

## Examples

Calibrate using the **AB system**:

```bash
python calibrate_spectrum.py ./sample/_hd128998_20250515_861.fits filters/bessel-v-baader.csv 5.83 --system=ab --plot
```

Calibrate using the **Vega system**:

```bash
python calibrate_spectrum.py ./sample/_hd128998_20250515_861.fits filters/bessel-v-baader.csv 5.83 --system=vega
```

## Output

- The calibrated spectrum is saved to a new FITS file named:

  - `<original_name>-abs-AB.fits`
  - `<original_name>-abs-VEGA.fits`

- The output spectrum flux is calibrated in units of `erg / cm² / s / Å`.
- The original FITS header is preserved.
- A `COMMENT` header line is added, summarizing the calibration system, magnitude, and filter.

## Dependencies

- Python ≥ 3.8
- numpy
- astropy
- specutils
- scipy
- matplotlib (optional, only required with `--plot`)

Install with:

```bash
pip install numpy astropy specutils scipy matplotlib
```

