from typing import Self
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error
from astropy.io import fits
from astropy import constants as cst, units as u
import warnings
from scipy.optimize import curve_fit
from scipy.integrate import simpson

from models import skewed_gauss, double_gaussian
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

    def continuum(self, degree: int = None, max_degree = 6, ax: plt.Axes = None):
        # (Optional) Calculate the polynomial degree that fits the best on the spectrum
        if degree is None:
            # Ignore warnings for a degree that is too high, because that is what is being tested here
            warnings.simplefilter('ignore', np.exceptions.RankWarning)

            # Find a polynomial degree that fits the best
            degs = np.arange(max_degree + 1)
            rmse = [
                root_mean_squared_error(
                    self.flux, # Observed
                    np.poly1d( # Continuum fit
                        np.polyfit(self.wavelength, self.flux, deg) # Polynamial coefficients
                    )(self.wavelength)
                ) for deg in degs
            ]

            degree = degs[np.argmin(rmse)]

        continuum = np.poly1d(np.polyfit(self.wavelength, self.flux, deg=degree))(self.wavelength)

        # (Optional) Visualize the proces
        if ax is not None:
            ax.plot(self.wavelength, self.flux, '.', ms=2)
            ax.plot(self.wavelength, continuum, label=f'Continuum (degree {degree})')
            ax.legend()

        return continuum

    def normalize(self, degree: int = None, max_degree = 6, ax: plt.Axes = None):
        # Divide out the continuum
        self.flux /= self.continuum(degree, max_degree, ax)
        
    def select_dib(self, center_wavelength: float, search_window: tuple[float, float] = (10, 10), ax: plt.Axes = None):
        # Select window
        range_mask = (self.wavelength > center_wavelength - search_window[0]) & (self.wavelength < center_wavelength + search_window[1])
        wavelength = self.wavelength[range_mask]
        flux = self.flux[range_mask]

        def fit_single_gaussian(init_center = center_wavelength, init_width = 0.1, init_amplitude = 0.1, init_skew = 2, init_slope = (flux[0] - flux[-1]) / (wavelength[0] - wavelength[-1]), init_start = flux[0]):
            params = None

            try:
                params, _ = curve_fit(
                    skewed_gauss, wavelength, flux, p0=[init_center, init_width, init_amplitude, init_skew, init_slope, init_start], maxfev=10000, bounds=np.array((
                    [init_center-0.01,-np.inf,-np.inf,-np.inf, -np.inf, -np.inf],
                    [init_center+0.01, np.inf, np.inf, np.inf,  np.inf,  np.inf]
                )))

                center, width, amplitude, skew, slope, start = params
                prediction = skewed_gauss(wavelength, *params)
                continuum = slope * wavelength + start
                rmse = np.sqrt(np.sum((flux - prediction)**2) / flux.size)
                return params, prediction, continuum, rmse
            except RuntimeError:
                return None
            
        params, prediction, continuum, rmse = fit_single_gaussian()
        center, width, amplitude, skew, slope, start = params

        window_min = wavelength[np.argmin(prediction)] - 5 * width
        window_max = wavelength[np.argmin(prediction)] + 5 * width
        range_mask = (self.wavelength > window_min) & (self.wavelength < window_max)
        wavelength = self.wavelength[range_mask]
        flux = self.flux[range_mask]

        params, prediction, continuum, rmse = fit_single_gaussian(*params)
        center, width, amplitude, skew, slope, start = params

        fwhm = 2 * np.sqrt(2 * np.log(2)) * width
        ew = simpson(1 - prediction / continuum, wavelength)

        # (Optional) Visualize the proces
        if ax is not None:
            # print(params)
            ax.set_title(rf'$\lambda={center:.4g}$; RMSE={rmse:.4g}, FWHM={fwhm:.4g}, EW={ew:.4g}')
            ax.plot(wavelength, flux, '.', ms=2)
            ax.plot(wavelength, prediction, label=f'Gaussian fit')
            ax.axvline(window_min)
            ax.axvline(window_max)

        return prediction, params, rmse, fwhm, ew, False
        

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

        self.obs_date = header["DATE-OBS"]
        self.v_rad = header["HIERARCH ESO QC VRAD BARYCOR"] * u.km / u.s

    def plot(self, ax: plt.Axes, title: str = None, x_label: str = None, y_label: str = None):
        return super().plot(ax, title or f'{self.target} | {self.obs_date}', x_label, y_label)
    
    def correct_shift(self):
        # TODO: determine if identify_atomic_lines should be used here

        self.wavelength += (self.v_rad / cst.c.to('km/s')) * self.wavelength

class DibProfile(Spectrum):
    def __init__(self, target: str, wavelength: np.ndarray, flux: np.ndarray, center_wavelength: float):
        super().__init__(target, wavelength, flux)

        self.center_wavelength = center_wavelength
