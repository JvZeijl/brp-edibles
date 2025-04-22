from typing import Self
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error
from astropy.io import fits
from astropy import constants as cst, units as u
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.integrate import simpson
from datetime import datetime

from dib_profile import DibProfile
from models import n_gaussian
from atomic_lines import AtomicLine

class Spectrum:
    def __init__(self, target: str, wavelength: np.ndarray, flux: np.ndarray):
        self.target = target
        self.wavelength = wavelength
        self.flux = flux

    def plot(self, ax: plt.Axes, title: str = None, x_label: str = None, y_label: str = None):
        ax.plot(self.wavelength, self.flux, '.', ms=2)
        ax.set_title(title or self.target)
        ax.set_xlabel(x_label or 'Wavelength [Å]')
        ax.set_ylabel(y_label or 'Flux')

    def format_obs_date(self):
        return 'unknown'

    def identify_atomic_line(
        self,
        line1: AtomicLine,
        line2: AtomicLine,
        search_window = (10, 10),
        tollerance = 1,
        axes: tuple[plt.Axes, plt.Axes] = None,
        draw_expected = False,
        output_radial_velocity = False,
        output_difference = False
    ) -> np.ndarray:
        # Preparing expected wavelength information
        min_wvl, max_wvl = np.sort([line1.wavelength, line2.wavelength])
        expected_peak_diff = max_wvl - min_wvl

        # Finding the peaks in the selected range
        range_mask = (self.wavelength > min_wvl - search_window[0]) & (self.wavelength < max_wvl + search_window[1])
        wavelength = self.wavelength[range_mask]
        flux = self.flux[range_mask]
        peaks, _ = find_peaks(-flux, height=-0.99, prominence=0.01)

        # Calculate the difference between all peaks using a matrix
        peaks_wvl = wavelength[peaks]
        peaks_wvl_matrix = np.reshape(peaks_wvl, (-1, 1))
        peaks_diff_matrix = peaks_wvl_matrix - peaks_wvl_matrix.transpose()
        diff_from_expectation = np.abs(peaks_diff_matrix - expected_peak_diff)

        # Getting the pairs of absorption peak
        candidate_pairs = None

        # If the tollerance is None, only return the best pair
        if tollerance is None:
            # Only return the best pair; if no pairs are found np.min crashes, therefore use 0 instead
            lowest_diff = np.min(diff_from_expectation) if len(diff_from_expectation) > 0 else 0
            candidate_pairs = peaks_wvl[np.argwhere(diff_from_expectation == lowest_diff)]
        else:
            candidate_pairs = peaks_wvl[np.argwhere(diff_from_expectation < tollerance)]

        # (Optional) Visualize the proces
        if axes is not None:
            unique_candidates = np.unique(candidate_pairs)

            axes[0].axvline(min_wvl - search_window[0], color='black')
            axes[0].axvline(max_wvl + search_window[1], color='black')
            axes[1].set_xlim(min_wvl - search_window[0], max_wvl + search_window[1])

            for ax, title in zip(axes, ('Full spectrum', 'Search window')):
                self.plot(ax, title)
                ax.plot(
                    unique_candidates, # Wavelength
                    self.flux[np.isin(self.wavelength, unique_candidates)],
                    '.', ms=10, color='green', label='Line detection'
                )

                if draw_expected:
                    ax.axvline(line1.wavelength, linestyle='--', color='orange', label=fr'{line1.label}: $\lambda=${line1.wavelength} Å')
                    ax.axvline(line2.wavelength, linestyle='--', color='red', label=fr'{line2.label}: $\lambda=${line2.wavelength} Å')

            axes[1].legend()

        # (Optional) Calculating the the difference from the expected value
        if output_difference:
            tollerances = np.abs(np.abs(candidate_pairs[:, 0] - candidate_pairs[:, 1]) - expected_peak_diff)
            candidate_pairs = np.column_stack((candidate_pairs, tollerances))

        # (Optional) Calculating the radial velocity
        if output_radial_velocity:
            # The choice for np.min or np.max (i.e. using the lower or higher wavelength) is abitrary
            # as long as it is consistent for the observed and expected wavelengths
            lower_identified_wvls = np.min(candidate_pairs[:, (0, 1)], axis=1)
            radial_velocities = (lower_identified_wvls - min_wvl) / min_wvl * 2.997e5
            candidate_pairs = np.column_stack((candidate_pairs, radial_velocities))

        return candidate_pairs
    
    def remove_outliers(self, ax: plt.Axes = None):
        # Remove points that are very low
        low_points = self.flux <= 0.1
        deleted_wvl_low = self.wavelength[low_points]
        deleted_flux_low = self.flux[low_points]

        self.wavelength = self.wavelength[~low_points]
        self.flux = self.flux[~low_points]

        # (Optional) Visualize the proces
        if ax is not None:
            ax.plot(self.wavelength, self.flux, '.', ms=2)
            ax.plot(deleted_wvl_low, deleted_flux_low, '.', color='red', ms=2)

        return deleted_wvl_low, deleted_flux_low

    def continuum(self, degree: int = None, max_degree = 6, ax: plt.Axes = None, wavelength = None, flux = None):
        wavelength = self.wavelength if wavelength is None else wavelength
        flux = self.flux if flux is None else flux

        # (Optional) Calculate the polynomial degree that fits the best on the spectrum
        if degree is None:
            # Ignore warnings for a degree that is too high, because that is what is being tested here
            warnings.simplefilter('ignore', np.exceptions.RankWarning)

            # Find a polynomial degree that fits the best
            degs = np.arange(max_degree + 1)
            rmse = [
                root_mean_squared_error(
                    flux, # Observed
                    np.poly1d( # Continuum fit
                        np.polyfit(wavelength, flux, deg) # Polynamial coefficients
                    )(wavelength)
                ) for deg in degs
            ]

            degree = degs[np.argmin(rmse)]

        continuum = np.poly1d(np.polyfit(wavelength, flux, deg=degree))(wavelength)

        # (Optional) Visualize the proces
        if ax is not None:
            ax.plot(wavelength, flux, '.', ms=2)
            ax.plot(wavelength, continuum, label=f'Continuum (degree {degree})')
            ax.legend()

        return continuum

    def normalize(self, degree: int = None, max_degree = 6, ax: plt.Axes = None):
        # Divide out the continuum
        self.flux /= self.continuum(degree, max_degree, ax)
        
    def select_dib(self, center_wavelength: float, search_window: tuple[float, float] = (10, 10), ax: plt.Axes = None):
        def select_window(range_mask):
            wavelength = self.wavelength[range_mask]
            flux = self.flux[range_mask]

            # Determine continuum
            mean_start_wvl, mean_start_flux = np.mean(wavelength[:3]), np.mean(flux[:3])
            mean_end_wvl, mean_end_flux = np.mean(wavelength[-2:]), np.mean(flux[-2:])
            slope = (mean_start_flux - mean_end_flux) / (mean_start_wvl - mean_end_wvl)
            start = mean_start_flux - slope * mean_start_wvl
            continuum = slope * wavelength + start

            return wavelength, flux, continuum

        def fit_n_gaussians(n_gaussians, init_centers = None):
            # Ignore overflow errors during fitting
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', OptimizeWarning)

            # Setup fit parameters and bounds
            init_width = 0.1
            init_amplitude = 0.1
            init_skew = 2
            init_params = np.repeat([[center_wavelength, init_width, init_amplitude, init_skew]], n_gaussians, axis=0)

            center_bound_range = 1
            lower_bounds = np.repeat([[center_wavelength - center_bound_range, 0, 0, -np.inf]], n_gaussians, axis=0)
            upper_bounds = np.repeat([[center_wavelength + center_bound_range, np.inf, np.inf, np.inf]], n_gaussians, axis=0)

            if init_centers is not None:
                assert len(init_centers) == n_gaussians, f'the length of init_centers ({len(init_centers)}) must be the same as n_gaussians ({n_gaussians})'

                init_params[:, 0] = init_centers
                lower_bounds[:, 0] = init_centers - center_bound_range
                upper_bounds[:, 0] = init_centers + center_bound_range

            def model(wvl, *params_list):
                return continuum + n_gaussian(wvl, *params_list)
            
            try:
                params, _ = curve_fit(
                    model, wavelength, flux, maxfev=10_000,
                    p0=init_params.flatten(), bounds=(lower_bounds.flatten(), upper_bounds.flatten())
                )

                return DibProfile(self.target, model, params)
            except:
                return None

        # Fit a single gaussian to narrow the window
        wavelength, flux, continuum = select_window(
            (self.wavelength > center_wavelength - search_window[0]) & (self.wavelength < center_wavelength + search_window[1])
        )

        single_gauss_profile = fit_n_gaussians(1)

        if single_gauss_profile is None:
            return None

        single_gauss_prediction = single_gauss_profile.predict(wavelength)
        single_gaussian_rmse = single_gauss_profile.rmse(wavelength, flux)

        # Ignore bad DIB detections
        if single_gaussian_rmse > 0.1:
            return None

        # Narrow the window: select 5 sigma around the lowest point
        width = single_gauss_profile.parameters[1] # parameters: [center, width, amplitude, skew]
        window_min = wavelength[np.argmin(single_gauss_prediction)] - 5 * width
        window_max = wavelength[np.argmin(single_gauss_prediction)] + 5 * width
        wavelength, flux, continuum = select_window(
            (self.wavelength > window_min) & (self.wavelength < window_max)
        )

        # Fit for different amount of gaussians
        amount_of_gaussians = np.array([1, 2, 3])
        dib_profiles = np.array([profile for n in amount_of_gaussians if (profile := fit_n_gaussians(n)) is not None], dtype=object)

        if len(dib_profiles) == 0:
            return None

        rmses = np.array([profile.rmse(wavelength, flux) for profile in dib_profiles])
        params = np.array([np.reshape(profile.parameters, (-1, 4)) for profile in dib_profiles], dtype=object)
        max_widths = np.array([np.max(param[:, 1], axis=0) for param in params])
        fwhms = 2 * np.sqrt(2 * np.log(2)) * max_widths
        ews = np.array([profile.equivalent_width(wavelength, continuum) for profile in dib_profiles])

        # (Optional) Visualize the proces
        if ax is not None:
            ax.set_title(rf'{self.target} | {self.format_obs_date()}')
            ax.set_xlabel('Wavelength [$\\AA$]')
            ax.set_ylabel('Normalized flux + Offset')

            for idx, (n, profile, rmse, fwhm, ew) in enumerate(zip(amount_of_gaussians, dib_profiles, rmses, fwhms, ews)):
                height_diff = continuum[np.argmin(flux)] - np.min(flux)
                offset = idx * height_diff

                ax.plot(wavelength, flux + offset, '.', color='C0', ms=5)
                ax.plot(wavelength, profile.predict(wavelength) + offset, color=f'C{idx + 1}', label=rf'RMSE={rmse:.4g}, FWHM={fwhm:.4g}, EW={ew:.4g}')
                ax.text(wavelength[0], flux[0] + offset - height_diff * 0.1, rf'{n}-Gaussian fit', color=f'C{idx + 1}')
                
            ax.legend()

        best_fit_mask = np.argmin(rmses)
        best_profile = dib_profiles[best_fit_mask]
        best_rmse = rmses[best_fit_mask]
        best_fwhm = fwhms[best_fit_mask]
        best_ews = ews[best_fit_mask]

        return np.reshape(best_profile.parameters, (-1, 4)), best_rmse, best_fwhm, best_ews, window_min, window_max
        

class FitsSpectrum(Spectrum):
    def __init__(self, fits_file: str):
        # Open fits file
        hdulist = fits.open(fits_file)
        header = hdulist[0].header

        # Load data
        starting_wvl = header['CRVAL1']
        stepsize_wvl = header['CDELT1']
        flux = hdulist[0].data
        wavelength = np.arange(0, len(flux), 1) * stepsize_wvl + starting_wvl

        # Create instance
        super().__init__(header['OBJECT'], wavelength, flux)

        self.obs_date = datetime.fromisoformat(header["DATE-OBS"])
        self.v_rad = header["HIERARCH ESO QC VRAD BARYCOR"] * u.km / u.s

    def plot(self, ax: plt.Axes, title: str = None, x_label: str = None, y_label: str = None):
        return super().plot(ax, title or f'{self.target} | {self.obs_date}', x_label, y_label)
    
    def format_obs_date(self):
        return self.obs_date.strftime(r'%d-%m-%Y, %H:%M:%S')
    
    def correct_shift(self):
        # TODO: determine if identify_atomic_lines should be used here

        self.wavelength += (self.v_rad / cst.c.to('km/s')) * self.wavelength
