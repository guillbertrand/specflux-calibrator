import sys
import os
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from specutils import Spectrum1D
from scipy.interpolate import interp1d
from scipy.integrate import simps

def compute_lambda_eff(wave, trans):
    """Compute the effective wavelength of the filter (weighted by transmission)."""
    return np.sum(wave * trans) / np.sum(trans)

def main():
    if len(sys.argv) < 4:
        print("Usage: python calibrate_spectrum.py spectrum_path.fits filter_name magnitude_v")
        print("Example: python calibrate_spectrum.py data/spec.fits bessel_v 14.2")
        sys.exit(1)

    path_fits = sys.argv[1]
    filter_name = sys.argv[2]
    mag_v = float(sys.argv[3])

    fits_out = path_fits.replace(".fits", "-abs.fits")
    filter_path = os.path.join("filters", f"{filter_name}.csv")

    # --- Read FITS spectrum ---
    try:
        with fits.open(path_fits) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            wave = data['wavelength'] * u.Unit(header.get('TUNIT1', 'Angstrom'))
            flux = data['flux'] * u.Unit(header.get('TUNIT2', ''))
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        sys.exit(1)

    spec = Spectrum1D(spectral_axis=wave, flux=flux)

    # --- Read filter transmission ---
    try:
        tdata = np.loadtxt(filter_path, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Error reading filter CSV file: {e}")
        sys.exit(1)

    filt_wave_nm = tdata[:, 0]
    filt_trans = tdata[:, 1] / 100.0  # percent to fraction

    filt_wave_aa = filt_wave_nm * 10.0  # nm to Angstrom
    lambda_eff = compute_lambda_eff(filt_wave_aa, filt_trans)

    filt_interp = interp1d(filt_wave_aa, filt_trans, bounds_error=False, fill_value=0)
    trans_spec = filt_interp(spec.spectral_axis.to(u.AA).value)

    # --- Synthetic flux calculation ---
    wl = spec.spectral_axis.to(u.AA).value
    fl = spec.flux.to(u.Unit("erg / (s cm2 Angstrom)")).value

    numerator = simps(fl * trans_spec * wl, wl)
    denominator = simps(trans_spec * wl, wl)
    flux_synth = numerator / denominator  # erg/s/cm²/Å

    # --- Reference flux for AB system ---
    c_aa_s = c.to(u.AA / u.s).value  # speed of light in Angstrom/s
    f_nu = 10**(-0.4 * (mag_v + 48.6))  # erg/s/cm²/Hz
    f_lambda_ref = f_nu * c_aa_s / lambda_eff**2

    scale = f_lambda_ref / flux_synth
    flux_corrected = fl * scale

    # --- Save calibrated spectrum as FITS ---
    col1 = fits.Column(name='wavelength', array=wl, format='D', unit='Angstrom')
    col2 = fits.Column(name='flux', array=flux_corrected, format='D', unit='erg / (s cm2 Angstrom)')
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    hdu.name = 'SPECTRUM'

    for key in header:
        if key not in ["TUNIT1", "TUNIT2", '', 'COMMENT']:
            hdu.header[key] = header[key]

    hdu.header['TUNIT1'] = 'Angstrom'
    hdu.header['TUNIT2'] = 'erg / (s cm2 Angstrom)'
    hdu.header['COMMENT'] = f"Flux calibrated (AB), filter={filter_name}, mag={mag_v}, lambda_eff={lambda_eff:.1f} Å"

    hdu.writeto(fits_out, overwrite=True)
    print(f"✅ Output file written: {fits_out}")

if __name__ == "__main__":
    main()
