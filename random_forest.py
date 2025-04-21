from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from spectrum import Spectrum
from common import tqdm

def train_rf(rf: RandomForestRegressor, spectra: list[Spectrum]) -> RandomForestRegressor:
    wavelength = np.concatenate([spectrum.wavelength for spectrum in spectra])[:, np.newaxis]
    flux = np.concatenate([spectrum.flux for spectrum in spectra])

    rf.fit(wavelength, flux)

def plot_rf_prediction(rf: RandomForestRegressor, test_spectrum: Spectrum):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)

    predicted_flux = rf.predict(test_spectrum.wavelength[:, np.newaxis])
    rmse = root_mean_squared_error(test_spectrum.flux, predicted_flux)

    test_spectrum.plot(axes[0])
    axes[1].plot(test_spectrum.wavelength, predicted_flux, '.', ms=2)
    axes[2].plot(test_spectrum.wavelength, test_spectrum.flux - predicted_flux, '.', ms=2)

    axes[0].set_title('Observed spectrum')
    axes[1].set_title('Predicted spectrum')
    axes[2].set_title(f'Difference (RMSE={rmse:.4g})')

    fig.suptitle(f'RF prediction for {test_spectrum.target}')
    fig.tight_layout()


def cross_validation(spectra: list[Spectrum], n_estimators=100):
    # Start with 0 estimators because it gets increased in the for-loop straightaway
    rf = RandomForestRegressor(n_estimators=0, warm_start=True, random_state=27)
    spectra_tqdm = tqdm(spectra)

    for spectrum in spectra_tqdm:
        training_spectra = [spec for spec in spectra if spec is not spectrum]
        rf.n_estimators += n_estimators

        spectra_tqdm.set_description(f'Training on {len(training_spectra)} spectra: {', '.join([ts.target for ts in training_spectra])}')
        train_rf(rf, training_spectra)
        plot_rf_prediction(rf, spectrum)