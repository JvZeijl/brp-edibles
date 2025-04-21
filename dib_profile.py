import numpy as np
from spectrum import Spectrum

class DibProfile(Spectrum):
    def __init__(self, target, wavelength, flux):
        super().__init__(target, wavelength, flux)

    def center(self, central_wavelength: float = None):
        central_wavelength = central_wavelength or self.wavelength[np.argmin(self.flux)]
        return DibProfile(self.target, self.wavelength - central_wavelength, self.flux, central_wavelength)