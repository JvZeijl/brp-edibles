import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def identify_atomic_line(
    wavelength, flux,
    line1: float,
    line2: float,
    search_window = (10, 10),
    tollerance = None,
    axes: tuple[plt.Axes, plt.Axes] = None,
    draw_expected = False,
    output_radial_velocity = False,
    output_difference = False
) -> np.ndarray:
    """
    Find atomic lines by pairs based on their expected difference

    Parameters
    ----------
    wavelength : list
        List of the wavelengths for an entire spectrum

    flux : list
        List of the fluxes for an entire spectrum

    line1, line2 : float
        Lines to search for in units of Angstrom

    search_window : (left, right)
        How much to search for to the left and right of line1 and line2

    tollerance : float
        Allowed deviation from the expected difference between line1 and line2. If tollerance is None then the best one is returned.

    axes : (ax_top, ax_bottom)
        Give the axes to draw on (optional)

    draw_expected : bool
        Draw the location of the expected lines (line1 and line2), default: False

    output_radial_velocity : bool
        Calculate and return the radial velocity that corresponds to the wavelength shift

    output_difference : bool
        Return the deviation from the expected difference between the lines
    """

    wavelength_original = wavelength
    flux_original = flux

    # Preparing expected wavelength information
    min_wvl, max_wvl = np.sort([line1, line2])
    expected_peak_diff = max_wvl - min_wvl

    # Finding the peaks in the selected range
    range_mask = (wavelength > min_wvl - search_window[0]) & (wavelength < max_wvl + search_window[1])
    wavelength = wavelength[range_mask]
    flux = flux[range_mask]

    # Search window is outside of the spectrum
    if len(flux) == 0:
        return None

    flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
    peaks, _ = find_peaks(1-flux, height=0.99, prominence=0.01)

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
        axes[0].plot(wavelength_original, flux_original, '.', ms=2)
        axes[0].plot(
            unique_candidates, # Wavelength
            flux_original[np.isin(wavelength_original, unique_candidates)],
            '.', ms=10, color='green', label='Line detection'
        )

        axes[1].set_xlim(min_wvl - search_window[0], max_wvl + search_window[1])
        axes[1].plot(wavelength, flux, '.', ms=2)
        axes[1].plot(
            unique_candidates, # Wavelength
            flux[np.isin(wavelength, unique_candidates)],
            '.', ms=10, color='green', label='Line detection'
        )

        for ax, title in zip(axes, ('Full spectrum', 'Search window')):
            ax.set_title(title)

            if draw_expected:
                ax.axvline(line1, linestyle='--', color='orange', label=fr'$\lambda=${line1} Å')
                ax.axvline(line2, linestyle='--', color='red', label=fr'$\lambda=${line2} Å')

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


# --------------------
# --- Example Code ---
# --------------------
if __name__ == "__main__":
    wavelength, flux = np.loadtxt('data/ascii/HD170740/HD170740_w564_n9_20160612_U.txt', unpack=True)
    na_line1 = 5889.95
    na_line2 = 5895.924
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

    na_line_pairs = identify_atomic_line(
        wavelength, flux,
        na_line1, na_line2,
        axes=axes, draw_expected=True,
        output_difference=True,
        output_radial_velocity=True
    )

    # na_line_pairs is a list of possible detections
    for index, pair in enumerate(na_line_pairs):
        print(f'---- Detection {index + 1} ----') # zero based index --> +1
        print('Line 1 (angstrom)', pair[0])
        print('Line 2 (angstrom)', pair[1])
        print('Deviation (angstrom)', pair[2])
        print('Radial velocity', pair[3])

    fig.suptitle(f'Na I atomic line detection for HD170740')
    fig.tight_layout()
    fig.savefig('na_lines.pdf')