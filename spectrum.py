import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, os.path as path

from dirs import DATA_FOLDER, OUTPUT_FOLDER

class Spectrum:
    def __init__(self, target, wavelength=None, flux=None):
        self.target = target

        if wavelength is None or flux is None:
            wavelength, flux = self._read_spectrum()
        
        self.wavelength = wavelength
        self.flux = flux

    def _read_spectrum(self):
        wavelength = []
        flux = []

        for dir_entry in tqdm(os.listdir(path.join(DATA_FOLDER, self.target)), f'Reading data for target {self.target}'):
            entry_path = path.join(DATA_FOLDER, self.target, dir_entry)

            # Ignore directories and hidden files
            if not path.isfile(entry_path) or dir_entry.startswith('.'):
                continue

            # Check for the right file format
            if not dir_entry.endswith('.ascii'):
                print(f'[WARNING]: not a valid data file {dir_entry}')
                continue

            spectrum = np.loadtxt(entry_path, unpack=True)
            wavelength.append(spectrum[0])
            flux.append(spectrum[1])

        # Sort based on the wavelength
        wavelength = np.concatenate(wavelength)
        flux = np.concatenate(flux)
        sorted_indices = np.argsort(wavelength)

        return wavelength[sorted_indices], flux[sorted_indices]
    
    def plot(self, ax: plt.Axes):
        ax.plot(self.wavelength, self.flux, '.', ms=2)
        ax.set_title(self.target)
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Normalized flux')
        