from spectrum import Spectrum
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

type AtomicLine = tuple[str, float]

# -------------------------------------------------------------------------------------
# -------------------------------------- SOURCES --------------------------------------
# -------------------------------------------------------------------------------------
# Na I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/sodiumtable3.htm
# K  I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/potassiumtable3.htm
# Ca I:  https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable3.htm
# Ca II: https://physics.nist.gov/PhysRefData/Handbook/Tables/calciumtable4.htm

NA_I_LINES: list[AtomicLine] = [
    ('Na I', 5889.950),
    ('Na I', 5895.924)
]

K_I_LINES: list[AtomicLine] = [
    ('K I', 7664.8991),
    ('K I', 7698.9645)
]

CA_I_LINES: list[AtomicLine] = [
    # TODO
]

CA_II_LINES: list[AtomicLine] = [
    # TODO
]

def identify_atomic_line(spectrum: Spectrum, line1: AtomicLine, line2: AtomicLine, search_window=(10, 100), tollerance=1):
    wvl1, wvl2 = line1[1], line2[1]
    min_wvl = np.min([wvl1, wvl2])
    max_wvl = np.max([wvl1, wvl2])
    expected_wvl_diff = max_wvl - min_wvl

    wvl_range_mask = (spectrum.wavelength > min_wvl - search_window[0]) & (spectrum.wavelength < max_wvl + search_window[1])
    wavelength = spectrum.wavelength[wvl_range_mask]
    flux = spectrum.flux[wvl_range_mask]
    peaks, _ = find_peaks(-flux, height=-0.99, prominence=0.01)

    possible_peak_combinations = []
    for current_peak in peaks:
        following_peaks = peaks[peaks > current_peak]
        wvl_diffs = wavelength[following_peaks] - wavelength[current_peak]
        possible_diff_indices = following_peaks[np.abs(wvl_diffs - expected_wvl_diff) < tollerance]

        if len(possible_diff_indices) > 0:
            possible_peak_combinations.extend(np.column_stack((wavelength[possible_diff_indices], np.full(possible_diff_indices.shape, wavelength[current_peak]))))

    return np.array(possible_peak_combinations)

def find_optimal_tollerance(spectrum: Spectrum, line1: AtomicLine, line2: AtomicLine, search_window=(10, 100), stepsize=0.001, verbose=False):
    tollerance = 0
    identified_lines = []
    while len(identified_lines) != 1:
        identified_lines.extend(identify_atomic_line(spectrum, line1, line2, search_window, tollerance))
        n_lines = len(identified_lines)

        if n_lines == 1:
            break
        elif len(identified_lines) > 1:
            if tollerance == 0:
                break
            
            tollerance -= stepsize
        else:
            tollerance += stepsize

        if verbose:
            print(f'Found {len(identified_lines)} lines, adjusting tollerance to {tollerance}')
    return tollerance

def plot_atomic_lines(spectrum: Spectrum, line1: AtomicLine, line2: AtomicLine, search_window=(10, 100), tollerance=None, draw_expected=True, stepsize=0.001):
    if tollerance == None:
        tollerance = find_optimal_tollerance(spectrum, line1, line2, search_window, stepsize)
        print(f'Optimal tollerance {tollerance:4g}')

    identified_peaks = identify_atomic_line(spectrum, line1, line2, search_window, tollerance)
    peaks_mask = np.isin(spectrum.wavelength, identified_peaks.flatten())
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,6))

    # Show search window
    axes[0].axvline(line1[1] - search_window[0], color='black')
    axes[0].axvline(line2[1] + search_window[1], color='black')
    axes[1].set_xlim(line1[1] - search_window[0], line2[1] + search_window[1])

    for ax in axes:
        spectrum.plot(ax)
        
        # Identified peaks
        ax.plot(spectrum.wavelength[peaks_mask], spectrum.flux[peaks_mask], '.', ms=10, color='green')

        if draw_expected:
            ax.axvline(line1[1], linestyle='--', color='orange', label=fr'{line1[0]}: $\lambda=${line1[1]} Å')
            ax.axvline(line2[1], linestyle='--', color='red', label=fr'{line2[0]}: $\lambda=${line2[1]} Å')


    axes[1].legend()
    fig.tight_layout()

    return identified_peaks

def radial_velocity_km_s(identified_line: float, expected_line: float):
    return (identified_line - expected_line) / expected_line * 2.997e5

