import numpy as np

from specutils import Spectrum1D,SpectralRegion
from astropy.modeling import models, fitting
from specutils.manipulation import extract_region
from astropy.cosmology import Planck18 as cosmo  
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.constants import c


from specutils.spectra import SpectralRegion

from matplotlib.gridspec import GridSpec
import numpy as np
from specutils import Spectrum1D, SpectralRegion
from astropy.modeling import models, fitting, Fittable1DModel, Parameter
import astropy.units as u

from specutils.fitting import fit_lines


from dust_extinction.parameter_averages import CCM89
import extinction

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' 

def fit_hbeta_and_measure_z(spectrum, approx_hbeta_angstrom, R):
 
    # Define fitting region ±50 Å around approx Hβ
    region_width = 90 * u.AA
    flux_region = SpectralRegion((approx_hbeta_angstrom - region_width),
                            (approx_hbeta_angstrom + region_width))
    sub_spec = extract_region(spectrum, flux_region)

    # Initial Voigt profile guess
    flux = sub_spec.flux
    wavelength = sub_spec.spectral_axis

    # Nettoyage des valeurs non finies
    valid = np.isfinite(flux)
    wavelength = wavelength[valid]
    flux = flux[valid]

    # Construire un modèle initial robuste
    fit_init = models.Lorentz1D(x_0=approx_hbeta_angstrom)
    sub_spec=sub_spec-np.mean(sub_spec.flux[:10].value)
    # Fit sécurisé
    fitter = fitting.LevMarLSQFitter()
    fit = fit_lines(sub_spec, fit_init, fitter=fitter)

    # Fitted central wavelength (observed)
    fitted_center = fit.x_0.value

    # Calculate redshift z from rest wavelength Hβ = 4861 Å
    z = (fitted_center / 4861.34) - 1

    # Affiche le fit et les composantes
    lam = wavelength.to_value()
    fitted_flux = fit(wavelength)

    return z, fitted_center, lam, sub_spec.flux, fitted_flux

def plot_combined(spectrum_full, spectrum_zoom, approx_hbeta_angstrom=4876*u.AA, R=850):

    z, fitted_center, lam, flux_zoom, fitted_flux = fit_hbeta_and_measure_z(spectrum_zoom, approx_hbeta_angstrom, R)

    # Données full spectrum
    valid_full = np.isfinite(spectrum_full.flux)
    flux_full = spectrum_full.flux[valid_full].value
    wavelength_full = spectrum_full.spectral_axis[valid_full].to_value()

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[.8, 1.2], hspace=0.3, wspace=0.2)

    # Full spectrum (en haut, 2 colonnes)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(wavelength_full, flux_full, color='k', lw=1.0)
    ax0.set_xlabel(r'Wavelength (\AA)')
    ax0.set_ylabel(r'Flux in erg/cm2/s/Å')
    ax0.set_title(
        r'\textbf{3C273}' + '\n'
        r'2025/06/18 - La Montagne - G.Bertrand' + '\n'
        r'Newton 150 f/5 - StarEx LR (300 l/mm, 80x80, 26$\mu$m slit) - IMX585'
    )
    ax0.grid(False)

    # Fit Hβ (zoom) à gauche en bas
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(lam, flux_zoom, 'k', label='Observed spectrum', lw=1.0)
    ax1.plot(lam, fitted_flux, '--', label=rf'Model $z={z:.6f}$')
    #ax1.axvline(4861, color='gray', linestyle=':', label=r'H$\beta$ rest-frame (4861\,\AA)')
    ax1.set_xlabel(r'Wavelength (\AA)')
    ax1.set_ylabel(r'Flux')
    ax1.set_title(
        r'$\mathbf{z = %.6f}$' % z + '\n' +
        r'H$\beta$ region with Lorentzian Fit',
        loc='center'
    )
    ax1.legend(fontsize='small')
    ax1.grid(False)

    region_width = 200 * u.AA
    center = 4963 * u.AA
    flux_region = SpectralRegion((center - region_width),
                            (center + region_width - 60*u.AA))

    spectrum_rest = shift_spectrum_to_rest(spectrum, z)

    sub_spec = extract_region(spectrum_rest, flux_region)
    fit, fitter = fit_multiple_gaussians_with_tied_constraints(sub_spec)

    # Fit Hβ (zoom) à droite en bas (placeholder pour futur plot)
    ax2 = fig.add_subplot(gs[1,1])
    ax2 = plot_fit_result(ax2, sub_spec, fit) 

    ymin1, ymax1 = ax1.get_ylim()
    ax1.set_ylim(ymin1, ymax1 * 1.8) 
    ymin2, ymax2 = ax2.get_ylim()
    ax2.set_ylim(ymin2, ymax2 * 1.5)
    
    ax1.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1, 1))
    ax2.legend(ncol=2, fontsize='small', loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig("3c273.png", dpi=150, bbox_inches='tight')
    plt.show()

    return z, fitted_center, fit,fitter, spectrum_rest






def shift_spectrum_to_rest(spectrum, z):
    """Décale la longueur d’onde pour ramener à la position de repos."""
    rest_wavelength = spectrum.spectral_axis / (1 + z)
    return Spectrum1D(spectral_axis=rest_wavelength, flux=spectrum.flux)

def fit_multiple_gaussians_with_tied_constraints(spectrum):
    # region = [(4760 * u.AA, 4780 * u.AA), (5100 * u.AA, 5120 * u.AA)]
    # with warnings.catch_warnings():  # Ignore warnings
    #     warnings.simplefilter('ignore')
    #     g1_fit = fit_continuum(spectrum, window=region)

    # y_continuum_fitted = g1_fit(spectrum.spectral_axis)

    # spectrum -= y_continuum_fitted

    flux = spectrum.flux.value
    wave = spectrum.spectral_axis.value

    # Use the (vacuum) rest wavelengths of known lines as initial values
    # for the fit.
    Hbeta = 4861
    O3_4959 = 5007


    # Create Gaussian1D models for each of the H-beta and [OIII] lines.
    hbeta_broad = models.Gaussian1D( mean=Hbeta, stddev=10)
    hbeta_narrow = models.Gaussian1D(mean=Hbeta, stddev=5)
    hbeta_narrow.mean.bounds = (4856, 4896)  # autorise un déplacement limité
    o3_4959 = models.Gaussian1D(amplitude=0.25, mean=O3_4959, stddev=13)


    # Create a polynomial model to fit the continuum.
    mean_flux = flux.mean()
    cont = np.where(flux > mean_flux, mean_flux, flux)
    linfitter = fitting.LinearLSQFitter()
    poly_cont = linfitter(models.Polynomial1D(1), wave, cont)

    # Create a compound model for the four emission lines and the continuum.
    model = hbeta_narrow + hbeta_broad + o3_4959  + poly_cont

    # 1. Broad Hβ suit la même longueur d’onde que narrow
    def tie_hb_broad_mean(m):
        return m.mean_0 +5  # mean_0 = Hβ narrow

    def tie_hb_broad_ampl(m):
        return m.amplitude_0 * 0.7  # broad plus faible que narrow

    hbeta_broad.mean.tied = tie_hb_broad_mean
    hbeta_broad.amplitude.tied = tie_hb_broad_ampl

    # 2. [OIII] suit le même redshift que Hβ
    def tie_o3_mean(m):
        return m.mean_0 * O3_4959 / Hbeta

    o3_4959.mean.tied = tie_o3_mean

    hbeta_broad.stddev.bounds = (8, 40)
    hbeta_narrow.stddev.bounds = (5, 15)
    o3_4959.stddev.bounds = (12, 15)


    # Simultaneously fit all the emission lines and continuum.
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, wave, flux)



    return fitted_model, fitter

def plot_fit_result(ax, spectrum, model):

    region_width = 200 * u.AA
    center = 4963 * u.AA
    flux_region = SpectralRegion((center - region_width),
                            (center + region_width))
    
    spectrum = extract_region(spectrum, flux_region)

    lam = spectrum.spectral_axis.to_value()
    flux = spectrum.flux.value
    fitted_flux = model(lam)

    # Récupère les composantes si modèle composé
    if hasattr(model, 'n_submodels'):
        components = [model[i] for i in range(model.n_submodels)]
    else:
        components = [model]

    ax.plot(lam, flux, color='k', lw=1.1, label='Data')
   
    ax.plot(lam, fitted_flux, 'r', lw=0.8, label='Total')
    lines = [ r"Narrow H$\beta$ component", r"Broad H$\beta$ component", r"[OIII] $\lambda$4959,5007", r"Continnuum"]
    #lines_color = ["g--", "g--", "b--", "o--", "r--"]
    for i, comp in enumerate(components):
        plt.plot(lam, comp(lam), "--", lw=1, label=lines[i])

    ax.set_xlabel(r'Wavelength (\AA)')
    ax.set_ylabel(r'Flux')
    ax.set_title(r'Rest-frame multi-Gaussian fit')

    return ax


def fwhm_instr_from_R(lambda_rest, R):
    """
    Calcule la FWHM instrumentale en km/s à partir de la longueur d'onde et la résolution R.
    """
    c = 299792.458
    delta_lambda = lambda_rest / R
    fwhm_instr = c * delta_lambda / lambda_rest
    return fwhm_instr

def calculate_mbh_vestergaard(f_lambda, fwhm_kms, z, EBV):
    # Distance luminosité en cm
    d_l = cosmo.luminosity_distance(z).to(u.cm).value
    R_V = 3.1
    wavelength = np.array([5100], dtype=float)
    A_lambda = extinction.ccm89(wavelength, R_V * EBV, R_V)[0]
    f_lambda_corr = f_lambda * 10**(0.4 * A_lambda)
    L_lambda = 4 * np.pi * d_l**2 * f_lambda_corr  # erg/s/Å
    lambda_L_lambda = 5100 * L_lambda         # erg/s

    # Logarithmes
    log_L = np.log10(lambda_L_lambda / 1e44)
    log_FWHM = np.log10(fwhm_kms / 1000)
    log_M_BH = 6.91 + 0.5 * log_L + 2 * log_FWHM
    M_BH = 10**log_M_BH

    return log_M_BH,  M_BH, d_l
    
def calculate_mbh_feng(f_lambda, fwhm_kms, z, EBV):
    # Distance de luminosité (en cm)
    d_l = cosmo.luminosity_distance(z).to(u.cm).value

    # Luminosité monochromatique à 5100 Å
    R_V = 3.1
    wavelength = np.array([5100], dtype=float)
    A_lambda = extinction.ccm89(wavelength, R_V * EBV, R_V)[0]
    f_lambda_corr = f_lambda * 10**(0.4 * A_lambda)
    L_lambda = 4 * np.pi * d_l**2 * f_lambda_corr  # erg/s/Å
    lambda_L_lambda = 5100 * L_lambda         # erg/s

    # Logarithmes
    log_L = np.log10(lambda_L_lambda / 1e44)
    log_FWHM = np.log10(fwhm_kms)

    # Masse du trou noir en log(M_sun)
    log_M_BH = 3.602 + 0.504 * log_L + 1.200 * log_FWHM
    M_BH = 10**log_M_BH



    return log_M_BH, M_BH, d_l
    

def flux_and_error(mag_V, mag_err, F0=3.631e-9):
    flux = F0 * 10**(-0.4 * mag_V)
    flux_err = 0.921 * flux * mag_err
    return flux, flux_err

spectrum = Spectrum1D.read("C:\\Users\\g-ber\\Documents\\ASTRO\\starex\\ccdciel\\20250618\\_3c273_20250618_89.fits", format='wcs1d-fits')
spectrum_abs = Spectrum1D.read("C:\\Users\\g-ber\\Documents\\ASTRO\\starex\\ccdciel\\20250618\\_3c273_20250618_89-abs.fits", format='wcs1d-fits')

z, fitted_center, fit,fitter, spectrum_rest = plot_combined(spectrum_abs, spectrum, approx_hbeta_angstrom=5632*u.AA, R=850)
print(f"z: {z:.8f}")
print(f"Fitted Hβ center: {fitted_center:.8f}")

def format_with_error(value, error, significant_digits=2):
    """
    Affiche value ± error dans la même puissance de 10,
    basée sur l’exposant de value uniquement.
    """
    if value == 0:
        exponent = int(np.floor(np.log10(abs(error)))) if error != 0 else 0
    else:
        exponent = int(np.floor(np.log10(abs(value))))

    val_scaled = value / 10**exponent
    err_scaled = error / 10**exponent

    val_str = f"{val_scaled:.{significant_digits}g}"
    err_str = f"{err_scaled:.{significant_digits}g}"

    # Pour avoir ×10^n format classique
    return f"({val_str} ± {err_str}) × 10^{exponent}"


def log_gaussian_fwhm_in_velocity_with_error(compound_model, fitter):
    broad_hbeta_fwhm = 0
    broad_hbeta_fwhm_err = 0
    print("=== Fit Components Summary ===")
    param_names = compound_model.param_names
    param_cov = fitter.fit_info['param_cov']

    # Chercher stddev_0 dans les noms des paramètres
    try:
        stddev_idx = param_names.index('stddev_0')
    except ValueError:
        print("stddev_0 not found in param_names:", param_names)
        return None, None

    for i, submodel in enumerate(compound_model):
        if isinstance(submodel, models.Gaussian1D):
            fwhm_angstrom = submodel.fwhm
            center_angstrom = submodel.mean.value
            fwhm_kms = (fwhm_angstrom / center_angstrom) * c / 1000  # km/s
            # Calcul erreur FWHM via erreur stddev
            sigma_stddev = np.sqrt(param_cov[stddev_idx, stddev_idx])
            fwhm_err = 2.355 * sigma_stddev
            fwhm_kms_err = (fwhm_err / center_angstrom) * c / 1000

            print(f"Component {i}:")
            print(f" - Center: {center_angstrom:.2f} Å")
            print(f" - FWHM: {fwhm_angstrom:.2f} Å")
            print(f" - FWHM: {fwhm_kms:.2f} ± {fwhm_kms_err:.2f} km/s\n")

            if i == 1:
                broad_hbeta_fwhm = fwhm_kms
                broad_hbeta_fwhm_err = fwhm_kms_err
    return broad_hbeta_fwhm, broad_hbeta_fwhm_err


broad_hbeta_fwhm, broad_hbeta_fwhm_err = log_gaussian_fwhm_in_velocity_with_error(fit, fitter)

# Calcul résolution instrumentale en km/s
fwhm_inst_kms = fwhm_instr_from_R(broad_hbeta_fwhm, R=800)

# Correction résolution instrumentale
fwhm_intrinsic_kms = np.sqrt(broad_hbeta_fwhm**2 - fwhm_inst_kms**2)

print(f"FWHM of Hβ (km/s): {broad_hbeta_fwhm:.1f}")
print(f"FWHM of Hβ (km/s) intrinsic: {fwhm_intrinsic_kms:.1f}")

# Measure flux near 5100 Å rest frame
flux_region = SpectralRegion(5095 * u.AA, 5105 * u.AA)
sub_flux = extract_region(spectrum_abs, flux_region)
flux_5100 = np.nanmean(sub_flux.flux.value)

mag_V = 13.36
mag_err = 0.07
flux, flux_err = flux_and_error(mag_V, mag_err)
print(f"Flux = {format_with_error(flux, flux_err)}")

log_M_BH, M_BH, d_l = calculate_mbh_vestergaard(flux, fwhm_intrinsic_kms, z,0.022)
log_M_BH2, M_BH2, d_l2 = calculate_mbh_feng(flux, fwhm_intrinsic_kms, z,0.022)

print(f"log(M_BH / M_sun) = {format_with_error(log_M_BH, 0)}, {format_with_error(log_M_BH2, 0)}")
print(f"M_BH = {format_with_error(M_BH, 0)}, {format_with_error(M_BH2, 0)}") 
# Convertir la distance de luminosité en Mpc
d_l_mpc = (d_l * u.cm).to(u.Mpc).value
d_l2_mpc = (d_l2 * u.cm).to(u.Mpc).value

print(f"Distance luminosité = {d_l:.2e} cm ({d_l_mpc:.3f} Mpc), {d_l2:.2e} cm ({d_l2_mpc:.3f} Mpc)")

