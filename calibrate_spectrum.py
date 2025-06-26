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

def compute_vega_flux(filter_wave_aa, filter_trans, vega_fits_path):
    """Compute synthetic Vega flux through a given filter."""
    try:
        vega_spec = Spectrum1D.read(vega_fits_path, format='wcs1d-fits')
        vega_spec = Spectrum1D(spectral_axis=vega_spec.spectral_axis,
                               flux=vega_spec.flux * (u.erg / (u.cm**2 * u.s * u.AA)))
    except Exception as e:
        print(f"Error reading Vega spectrum: {e}")
        sys.exit(1)

    vega_wave = vega_spec.spectral_axis.to(u.AA).value
    vega_flux = vega_spec.flux.value

    filt_interp = interp1d(filter_wave_aa, filter_trans, bounds_error=False, fill_value=0.0)
    trans_vega = filt_interp(vega_wave)

    num = simpson(vega_flux * trans_vega, vega_wave)
    denom = simpson(trans_vega, vega_wave)
    return num / denom

def compute_lambda_eff(wave, trans):
    """Compute effective wavelength of the filter (weighted by transmission)."""
    return np.sum(wave * trans) / np.sum(trans)

def main():
    if len(sys.argv) < 4:
        print("Usage: python calibrate_spectrum.py spectrum.fits filter.csv magnitude_v [--plot] [--system ab|vega]")
        sys.exit(1)

    path_fits = sys.argv[1]
    filter_path = sys.argv[2]
    mag_v = float(sys.argv[3])

    # Optional flags
    plot_flag = '--plot' in sys.argv
    system_flag = 'ab'  # default
    for arg in sys.argv[4:]:
        if arg.startswith('--system'):
            parts = arg.split('=')
            if len(parts) == 2 and parts[1].lower() in ['ab', 'vega']:
                system_flag = parts[1].lower()

    system_label = system_flag.upper()
    fits_out = path_fits.replace(".fits", f"-abs-{system_label}.fits")

    # Load spectrum
    try:
        spec = Spectrum1D.read(path_fits, format='wcs1d-fits')
        spec = Spectrum1D(spectral_axis=spec.spectral_axis,
                          flux=spec.flux * (u.erg / (u.cm**2 * u.s * u.AA)))
    except Exception as e:
        print(f"Error reading input spectrum: {e}")
        sys.exit(1)

    # Load filter transmission curve
    try:
        tdata = np.loadtxt(filter_path, delimiter=";", skiprows=1)
    except Exception as e:
        print(f"Error reading filter file: {e}")
        sys.exit(1)

    filt_wave_nm = tdata[:, 0]
    filt_trans = tdata[:, 1]
    filt_wave_aa = filt_wave_nm * 10.0  # Convert nm to Å

    # Interpolate filter transmission on spectrum wavelength grid
    wl = spec.spectral_axis.to(u.AA).value
    fl = spec.flux
    filt_interp = interp1d(filt_wave_aa, filt_trans, bounds_error=False, fill_value=0.0)
    trans_spec = filt_interp(wl)

    # Optional diagnostic plot of the filter
    if plot_flag:
        plt.figure(figsize=(10, 6))
        plt.plot(wl, trans_spec, label='Filter Transmission')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Transmission')
        plt.title(f'Filter: {os.path.basename(filter_path)} | Mag: {mag_v}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Compute synthetic flux of the object in the filter
    flux_synth = simpson(fl * trans_spec, wl) / simpson(trans_spec, wl)

    # Compute reference flux based on selected system
    if system_flag == 'ab':
        lambda_eff_aa = compute_lambda_eff(filt_wave_aa, filt_trans)
        c_aa_per_s = c.to("Angstrom / s").value
        f_nu = 10**(-0.4 * (mag_v + 48.6))  # AB system zero point
        f_lambda_ref = f_nu * c_aa_per_s / (lambda_eff_aa ** 2)
    elif system_flag == 'vega':
        vega_fits = "./calspec/vega.fits"
        flux_vega_filter = compute_vega_flux(filt_wave_aa, filt_trans, vega_fits)
        f_lambda_ref = flux_vega_filter * 10**(-0.4 * mag_v)
    else:
        print(f"Unknown system: {system_flag}")
        sys.exit(1)

    # Compute scale factor
    scale = f_lambda_ref / flux_synth

    # Diagnostics print
    print(f"System: {system_label}")
    print(f"λ_eff (Å): {compute_lambda_eff(filt_wave_aa, filt_trans):.2f}")
    print(f"Synthetic flux (erg/cm²/s/Å): {flux_synth:.3e}")
    print(f"Reference flux ({system_label}): {f_lambda_ref:.3e}")
    print(f"Scaling factor: {scale:.5e}")

    # Apply flux correction
    flux_corrected = fl * scale

    # Save calibrated spectrum to new FITS
    with fits.open(path_fits) as hdul:
        primary_header = hdul[0].header.copy()

    primary_header['COMMENT'] = f"Flux calibrated in {system_label} system, filter={os.path.basename(filter_path)}, magnitude={mag_v}"

    y = flux_corrected.astype(np.float32)
    hdu = fits.PrimaryHDU(y, header=primary_header)
    hdu.writeto(fits_out, overwrite=True)

    print(f"✅ Output written to: {fits_out}")

    # Optional plot of calibrated spectrum
    if plot_flag:
        plt.figure(figsize=(10, 6))
        plt.plot(wl, flux_corrected, label=f'Calibrated Flux ({system_label})')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux (erg/cm²/s/Å)')
        plt.title(f'Calibrated Spectrum ({system_label})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
