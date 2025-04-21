import numpy as np
from scipy.integrate import simpson

class DibProfile:
    def __init__(self, target, model, parameters):
        self.target = target
        self.model = model
        self.parameters = parameters

    def predict(self, wavelength):
        return self.model(wavelength, *self.parameters)
    
    def rmse(self, wavelength, flux_true):
        return np.sqrt(np.sum((flux_true - self.predict(wavelength))**2) / flux_true.size)
    
    def equivalent_width(self, wavelength, continuum):
        return simpson(1 - self.predict(wavelength) / continuum, wavelength)
