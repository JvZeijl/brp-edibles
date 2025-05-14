import numpy as np
from scipy.optimize import curve_fit

def skewed_gauss(wavelength,centre,width,amplitude,skew,slope=0,c=1):
    exp_1=np.exp(-(((wavelength-centre)/(2*width))**2))
    exp_2=np.exp(-skew*np.arctan((wavelength-centre)/(width))) 
    continuum=((slope*wavelength)+c)
    skewed_gauss=amplitude*exp_1*exp_2
    return  continuum-skewed_gauss

def n_gaussian(wavelength, *params_list):
    assert len(params_list) % 4 == 0, f'The number of parameters ({len(params_list)}) must be a multiple of 4 parameters, in order: center, width, amplitude, skew'

    gaussian_params = np.reshape(params_list, (-1, 4))
    gaussians = [skewed_gauss(wavelength, *params, c=0) for params in gaussian_params]
    return np.sum(gaussians, axis=0)

def fit_n_gaussians(wavelength, flux, n_gaussians: int, center_wavelength: float, continuum = 1, init_centers: list | None = None):
    """
    Parameters
    ----------
    n_gaussians : int
        Amount of gaussians

    center_wavelength : float
        Expected middle of the DIB, is overriden if init_centers is provided

    init_centers: list[float] | None
        Override the center_wavelength per peak, the length of init_centers must be the same as n_gaussians
    """
    # Setup fit parameters and bounds
    init_width = 0.1
    init_amplitude = 0.1
    init_skew = 0
    center_bound_range = 0.1
    init_params = np.repeat([[center_wavelength, init_width, init_amplitude, init_skew]], n_gaussians, axis=0)

    lower_bounds = np.repeat([[center_wavelength - center_bound_range, 0, 0, -2]], n_gaussians, axis=0)
    upper_bounds = np.repeat([[center_wavelength + center_bound_range, 2, 2, 2]], n_gaussians, axis=0)

    # Optionally set centers per gaussian
    if init_centers is not None:
        assert len(init_centers) == n_gaussians, f'the length of init_centers ({len(init_centers)}) must be the same as n_gaussians ({n_gaussians})'

        init_params[:, 0] = init_centers
        lower_bounds[:, 0] = init_centers - center_bound_range
        upper_bounds[:, 0] = init_centers + center_bound_range

    # Fitting
    def model(wvl, *params_list):
        return continuum + n_gaussian(wvl, *params_list)
    
    params, _ = curve_fit(
        model, wavelength, flux, maxfev=10_000,
        p0=init_params.flatten(), bounds=(lower_bounds.flatten(), upper_bounds.flatten()),
    )

    return params