import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from atomic_lines import AtomicLine

class Spectrum:
    def __init__(self, target, wavelength, flux):
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
