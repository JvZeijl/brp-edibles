from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from astropy.io import fits
from astropy import constants as cst, units as u
from datetime import datetime
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.linear_model import LinearRegression
from itertools import groupby
from operator import itemgetter
import warnings

from dib_profile import DibProfile
from models import n_gaussian, GAUSSIAN_SIGMA
from lines import identify_atomic_line

class AtomicLine:
    def __init__(self, label: str, wavelength: float):
        self.label = label
        self.wavelength = wavelength

class Spectrum:
    def __init__(self, target: str, wavelength: np.ndarray, flux: np.ndarray):
        self.target = target
        self.wavelength = wavelength
        self.flux = flux
        self.error = self._find_error()

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
        tollerance = None,
        axes: tuple[plt.Axes, plt.Axes] = None,
        draw_expected = False,
        output_radial_velocity = False,
        output_difference = False
    ) -> np.ndarray:
        return identify_atomic_line(self.wavelength, self.flux, line1.wavelength, line2.wavelength, search_window, tollerance, axes, draw_expected, output_radial_velocity, output_difference)
    
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

    def _find_error(self, return_segment = False):
        gradient = np.gradient(self.flux)
        flat_mask = np.abs(gradient) < 2

        indices = np.where(flat_mask)[0]
        flat_regions = []
        region_lengths = []
        for _, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > 50:  # Filter small regions
                flat_regions.append((group[0], group[-1]))
                region_lengths.append(len(group))

        best_region = flat_regions[np.argmax(region_lengths)]
        wavelength = self.wavelength[best_region[0]:best_region[1]]
        flux = self.flux[best_region[0]:best_region[1]]
        wvl_matrix = wavelength.reshape(-1, 1)

        model = LinearRegression().fit(wvl_matrix, flux)
        pred = model.predict(wvl_matrix)
        rmse = root_mean_squared_error(flux, pred)

        # plt.plot(wavelength, flux, '.', ms=2)
        # plt.plot(wavelength, pred)
        # plt.xlabel('Wavelength [Å]')
        # plt.ylabel('Flux')
        # plt.title(f'Error estimate of {self.target} | RMSE={rmse:.4g}')
        # plt.show()
        # plt.close()

        return (rmse, wavelength[0], wavelength[-1]) if return_segment else rmse

    def select_dibs(self, window_size = 20, window_step = 1, sigma_size = 3):
        window_start = self.wavelength[0]
        found_peaks = {}

        while window_start + window_size <= self.wavelength[-1]:
            # Select window
            window_mask = (window_start < self.wavelength) & (self.wavelength < window_start + window_size)
            wavelength = self.wavelength[window_mask]
            flux = self.flux[window_mask]
            flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
            peaks, props = find_peaks(1-flux, width=10)

            # Width of the peak in units of index
            idx_fwhms = props['widths']
            idx_widths = sigma_size * idx_fwhms / GAUSSIAN_SIGMA

            for idx_peak, idx_width in zip(peaks, idx_widths):
                idx_left_bound = np.floor(idx_peak - idx_width).astype(int)
                idx_right_bound = np.ceil(idx_peak + idx_width).astype(int)

                # Ignore peaks that are not fully in the window
                if not 0 < idx_left_bound < len(wavelength) or not 0 < idx_right_bound < len(wavelength):
                    continue

                peak_center = wavelength[idx_peak]
                left_bound = wavelength[idx_left_bound]
                right_bound = wavelength[idx_right_bound]

                existing_peak = found_peaks.get(peak_center)

                # Only assign if not defined or window is larger
                if existing_peak is None or right_bound - left_bound > existing_peak[1] - existing_peak[0]:
                    found_peaks[peak_center] = (left_bound, right_bound)

            # Move window
            window_start += window_step

        return found_peaks
        
    def fit_gaussian(self, center_wavelength: float, bounds: tuple, ax: plt.Axes = None, show_error = False, max_gaussians = 5):
        """
        Parameters
        ----------
        center_wavelength : float
            Expected middle of the DIB

        bounds : tuple
            Start and end of the DIB (assumed to contain 3 sigma on each side of center_wavelength)

        ax : plt.Axes | None
            Optionally plot the resulting fit
        """
        sigma2 = (center_wavelength - bounds[0]) * 2 / 3

        def select_window(range_mask):
            wavelength = self.wavelength[range_mask]
            flux = self.flux[range_mask]

            def continuum(mode: Literal['upper', 'lower']):
                # The regions beyond 2-sigma are considered part of the continuum
                left_anchor_region = flux[wavelength < center_wavelength - sigma2]
                right_anchor_region = flux[wavelength > center_wavelength + sigma2]

                # Determine the anchor points for the given mode
                # [0] and [-1] select the most outward points if there are multiple values at the max/min
                mode_fn = np.max if mode == 'upper' else np.min
                left_anchor_idx = np.argwhere(flux == mode_fn(left_anchor_region))[0]
                right_anchor_idx = np.argwhere(flux == mode_fn(right_anchor_region))[-1]

                # Determine the line
                slope = (flux[left_anchor_idx] - flux[right_anchor_idx]) / (wavelength[left_anchor_idx] - wavelength[right_anchor_idx])
                intercept = flux[left_anchor_idx] - slope * wavelength[left_anchor_idx]

                return slope * wavelength + intercept

            # Min-max normalization and detrend by dividing out the upper continuum
            flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
            flux /= continuum('upper')

            continuum_lower = continuum('lower')
            continuum_upper = continuum('upper')

            return wavelength, flux, np.mean([continuum_lower, continuum_upper], axis=0), continuum_lower, continuum_upper

        def fit_n_gaussians(n_gaussians: int, init_centers = None):
            """
            Parameters
            ----------
            n_gaussians : int
                Amount of gaussians

            init_centers: list[float] | None
                Override the center_wavelength per peak, the length of init_centers must be the same as n_gaussians
            """
            # Ignore overflow errors during fitting
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', OptimizeWarning)

            # Setup fit parameters and bounds
            init_width = 0.1
            init_amplitude = 0.1
            init_skew = 0
            init_params = np.repeat([[center_wavelength, init_width, init_amplitude, init_skew]], n_gaussians, axis=0)

            center_bound_range = 0.1
            lower_bounds = np.repeat([[center_wavelength - center_bound_range, 0, 0, -2]], n_gaussians, axis=0)
            upper_bounds = np.repeat([[center_wavelength + center_bound_range, 10, 1, 2]], n_gaussians, axis=0)

            if init_centers is not None:
                if len(init_centers) > n_gaussians:
                    init_centers = init_centers[:n_gaussians]

                init_center_bound_range = np.full_like(init_centers, 0.02)

                if len(init_centers) < n_gaussians:
                    n_extra_gaussians = n_gaussians - len(init_centers)
                    init_centers = np.pad(init_centers, (0, n_extra_gaussians), 'constant', constant_values=center_wavelength)
                    init_center_bound_range = np.pad(init_center_bound_range, (0, n_extra_gaussians), 'constant', constant_values=center_bound_range)

                init_params[:, 0] = init_centers
                lower_bounds[:, 0] = init_centers - init_center_bound_range
                upper_bounds[:, 0] = init_centers + init_center_bound_range

            def model(wvl, *params_list):
                return continuum + n_gaussian(wvl, *params_list)
            
            params, _ = curve_fit(
                model, wavelength, flux, maxfev=10_000,
                p0=init_params.flatten(), bounds=(lower_bounds.flatten(), upper_bounds.flatten()),
            )

            return DibProfile(self.target, model, params)

        # Fit a single gaussian to narrow the window
        wavelength, flux, continuum, lower_continuum, upper_continuum = select_window(
            (self.wavelength > bounds[0]) & (self.wavelength < bounds[1])
        )

        # Find substructure locations
        substructure_peaks, props = find_peaks(1-flux, prominence=0.01, height=0.2, width=2)

        # Sort the peaks such that the ones closest to the center are always fitted
        substructure_peaks = substructure_peaks[np.argsort(np.abs(wavelength[substructure_peaks] - center_wavelength))]

        # Select the 5 most prominent peaks
        if len(substructure_peaks) > max_gaussians:
            substructure_peaks = substructure_peaks[np.argsort(props['prominences'])[-max_gaussians:]]

        # Fit for different amount of gaussians (try the amount of peaks found or at least a triple)
        amount_of_gaussians = np.arange(1, max(len(substructure_peaks), 3) + 1)
        dib_profiles = np.array([
            profile for n in amount_of_gaussians
                if (profile := fit_n_gaussians(n, init_centers=wavelength[substructure_peaks])) is not None
        ], dtype=object)

        if len(dib_profiles) == 0:
            return None

        rmses = np.array([profile.rmse(wavelength, flux) for profile in dib_profiles])
        r2s = np.array([profile.r2(wavelength, flux) for profile in dib_profiles])
        params = np.array([np.reshape(profile.parameters, (-1, 4)) for profile in dib_profiles], dtype=object)
        max_widths = np.array([np.max(param[:, 1], axis=0) for param in params])
        fwhms = 2 * np.sqrt(2 * np.log(2)) * max_widths
        ews = np.array([profile.equivalent_width(wavelength, continuum) for profile in dib_profiles])

        # Prepend 0 such that the single gaussian will be selected if the other gaussians do not change wrt the single gaussian
        rmse_change = np.abs(np.diff(rmses, prepend=0))
        best_fit_idx = np.argwhere(rmses == np.min(rmses[rmse_change > 0.01]))[0][0]
        best_profile = dib_profiles[best_fit_idx]
        best_rmse = rmses[best_fit_idx]
        best_r2 = r2s[best_fit_idx]
        best_fwhm = fwhms[best_fit_idx]
        best_ews = ews[best_fit_idx]

        # (Optional) Visualize the proces
        if ax is not None:
            if show_error:
                ax.set_title(rf'{self.target} | {self.format_obs_date()}; $\sigma={self.error:.4g}$')
            else:
                ax.set_title(rf'{self.target} | {self.format_obs_date()}')

            ax.set_xlabel('Wavelength [$\\AA$]')
            ax.set_ylabel('Normalized flux + Offset')

            for idx, (n, profile, rmse, fwhm, ew) in enumerate(zip(amount_of_gaussians, dib_profiles, rmses, fwhms, ews)):
                height_diff = continuum[np.argmin(flux)] - np.min(flux)
                offset = idx * (height_diff + 0.5)

                if show_error:
                    ax.errorbar(wavelength, flux + offset, self.error, fmt='.', color='C0', ms=5)
                else:
                    ax.plot(wavelength, flux + offset, '.', color='C0', ms=5)

                # ax.plot(wavelength[substructure_peaks], flux[substructure_peaks] + offset, 'x', color='black', ms=5, label='Substructure' if idx == 0 else '')

                ax.plot(wavelength, lower_continuum + offset, color='C6', linestyle='--', label='Lower Continuum' if idx == 0 else '')
                ax.plot(wavelength, upper_continuum + offset, color='C7', linestyle='--', label='Upper Continuum' if idx == 0 else '')
                ax.plot(wavelength, continuum + offset, color='C8', label='Mean Continuum' if idx == 0 else '')

                ax.plot(wavelength, profile.predict(wavelength) + offset, color=f'C{idx + 1}', label=rf'RMSE={rmse:.4g}, FWHM={fwhm:.4g}, EW={ew:.4g}')
                ax.text(wavelength[0], lower_continuum[0] + offset - height_diff * 0.1, rf'{n}-Gaussian fit {'(best)' if idx == best_fit_idx else ''}', color=f'C{idx + 1}')
                
            ax.legend()

        return np.reshape(best_profile.parameters, (-1, 4)), best_rmse, best_r2, best_fwhm, best_ews
        

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
        self.v_rad_bary = header["HIERARCH ESO QC VRAD BARYCOR"] * u.km / u.s
        self.v_rad_heli = header["HIERARCH ESO QC VRAD HELICOR"] * u.km / u.s

    def plot(self, ax: plt.Axes, title: str = None, x_label: str = None, y_label: str = None):
        return super().plot(ax, title or f'{self.target} | {self.obs_date}', x_label, y_label)
    
    def format_obs_date(self):
        return self.obs_date.strftime(r'%d-%m-%Y, %H:%M:%S')
    
    def correct_shift(self):
        self.wavelength += ((self.v_rad_bary + self.v_rad_heli) / cst.c.to('km/s')) * self.wavelength
