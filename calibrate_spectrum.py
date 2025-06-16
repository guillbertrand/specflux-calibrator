import sys
import os
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from specutils import Spectrum
from scipy.interpolate import interp1d
from scipy.integrate import simpson

def compute_lambda_eff(wave, trans):
    """Compute effective wavelength of the filter (weighted by transmission)."""
    return np.sum(wave * trans) / np.sum(trans)

def main():
    if len(sys.argv) < 4:
        print("Usage: python calibrate_spectrum.py spectrum.fits filter_name magnitude_v")
        sys.exit(1)

    path_fits = sys.argv[1]
    filter_path = sys.argv[2]
    mag_v = float(sys.argv[3])

    fits_out = path_fits.replace(".fits", "-abs.fits")
 

    # --- Load spectrum using specutils ---
    try:
        spec = Spectrum.read(path_fits, format='wcs1d-fits')
    except Exception as e:
        print(f"Error reading FITS spectrum: {e}")
        sys.exit(1)

    # --- Load filter transmission ---
    try:
        tdata = np.loadtxt(filter_path, delimiter=";", skiprows=1)
    except Exception as e:
        print(f"Error reading filter CSV file: {e}")
        sys.exit(1)

    filt_wave_nm = tdata[:, 0]
    filt_trans = tdata[:, 1] / 100.0  # convert % to fraction
    filt_wave_aa = filt_wave_nm * 10.0  # convert nm to Ångström

    lambda_eff = compute_lambda_eff(filt_wave_aa, filt_trans)
    filt_interp = interp1d(filt_wave_aa, filt_trans, bounds_error=False, fill_value=0)
    trans_spec = filt_interp(spec.spectral_axis.to(u.AA).value)

    # --- Synthetic flux calculation ---
    wl = spec.spectral_axis.to(u.AA).value
    fl = spec.flux

    numerator = simpson(fl * trans_spec * wl, wl)
    denominator = simpson(trans_spec * wl, wl)
    flux_synth = numerator / denominator

    # --- AB reference flux ---
    c_aa_s = c.to(u.AA / u.s).value
    f_nu = 10**(-0.4 * (mag_v + 48.6))
    f_lambda_ref = f_nu * c_aa_s / lambda_eff**2

    scale = f_lambda_ref / flux_synth
    flux_corrected = fl * scale

    # Create a Primary HDU with the original header
    with fits.open(path_fits) as hdul:
        primary_header = hdul[0].header.copy()
        primary_hdu = fits.PrimaryHDU(header=primary_header)

    # Create binary table HDU for the calibrated data
    col1 = fits.Column(name='wavelength', array=wl, format='D', unit='Angstrom')
    col2 = fits.Column(name='flux', array=flux_corrected, format='D', unit='ergs/cm2/s/A')
    table_hdu = fits.BinTableHDU.from_columns([col1, col2])
    table_hdu.name = 'SPECTRUM'
    table_hdu.header['CUNIT1'] = 'Angstrom'
    table_hdu.header['CUNIT2'] = 'ergs/cm2/s/A'
    table_hdu.header['COMMENT'] = f"Flux calibrated (AB), filter={filter_path.split('/')[-1].split('.')[0]}, magnitude={mag_v}"

    # Write both HDUs to a new FITS file
    hdulist = fits.HDUList([primary_hdu, table_hdu])
    hdulist.writeto(fits_out, overwrite=True, output_verify='ignore')


    print(f"✅ Output written to: {fits_out}")

if __name__ == "__main__":
    main()
