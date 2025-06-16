import sys
import os
import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.io import fits
from specutils import Spectrum1D
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt


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
        spec = Spectrum1D.read(path_fits, format='wcs1d-fits')
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
    filt_trans = tdata[:, 1] 
    filt_wave_aa = filt_wave_nm * 10.0  # convert nm to Ångström

    filt_interp = interp1d(filt_wave_aa, filt_trans, bounds_error=False)
    trans_spec = filt_interp(spec.spectral_axis.to(u.AA).value)

    # Optional plot
    if len(sys.argv) > 4 and sys.argv[4].lower() == '--plot':
        plt.figure(figsize=(10,6))
        plt.plot(spec.spectral_axis.to(u.AA).value, trans_spec, label='Flux')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Intensity')
        plt.title(f'Filter: {filter_path.split("/")[-1].split(".")[0]}, Mag: {mag_v}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Synthetic flux calculation ---
    wl = spec.spectral_axis.to(u.AA).value
    fl = spec.flux

    numerator = simpson((fl * trans_spec), wl) 
    denominator = simpson(trans_spec, wl) 
    flux_synth = numerator / denominator

    # --- AB reference flux ---
    lambda_eff_aa = compute_lambda_eff(filt_wave_aa, filt_trans)
    c_aa_per_s = c.to("Angstrom / s").value
    f_nu = 10**(-0.4 * (mag_v + 48.6))  # AB system
    f_lambda_ref = f_nu * c_aa_per_s / (lambda_eff_aa ** 2)  

    scale = f_lambda_ref / flux_synth
    flux_corrected = fl * scale

    # Create a Primary HDU with the original header
    with fits.open(path_fits) as hdul:
        primary_header = hdul[0].header.copy()

    primary_header['COMMENT'] = f"Flux calibrated ergs/cm2/s/A (AB), filter={filter_path.split('/')[-1].split('.')[0]}, magnitude={mag_v}"

    y = flux_corrected.astype(np.float32)
    DiskHDU = fits.PrimaryHDU(y, header=primary_header)
    DiskHDU.writeto(fits_out, overwrite='True')



    print(f"✅ Output written to: {fits_out}")

    # Optional plot
    if len(sys.argv) > 4 and sys.argv[4].lower() == '--plot':
        plt.figure(figsize=(10,6))
        plt.plot(wl, flux_corrected, label='Calibrated Flux')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux (ergs/cm2/s/Å)')
        plt.title(f'Calibrated Spectrum - Filter: {filter_path.split("/")[-1].split(".")[0]}, Mag: {mag_v}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
