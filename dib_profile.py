from scipy.integrate import simpson
from sklearn.metrics import r2_score, root_mean_squared_error

class DibProfile:
    def __init__(self, target, model, parameters, start=0, end=0):
        self.target = target
        self.model = model
        self.parameters = parameters
        self.start = start
        self.end = end

    def predict(self, wavelength):
        return self.model(wavelength, *self.parameters)
    
    def rmse(self, wavelength, flux_true):
        return root_mean_squared_error(flux_true, self.predict(wavelength))
    
    def equivalent_width(self, wavelength, continuum):
        return simpson(1 - self.predict(wavelength) / continuum, wavelength)
    
    def r2(self, wavelength, flux_true):
        return r2_score(flux_true, self.predict(wavelength))
