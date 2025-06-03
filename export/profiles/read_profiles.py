import numpy as np

def read_profiles(path):
    profiles_file = open(path, 'r')
    obs_date_list = []
    start_list = []
    end_list = []
    rmse_list = []
    r2_list = []
    fwhm_list = []
    ew_list = []
    centers_list = []
    widths_list = []
    amplitudes_list = []
    skews_list = []

    for line in profiles_file.readlines():
        # Ignore comments
        if line.startswith('#'):
            continue

        obs_date, start, end, centers, widths, amplitudes, skews, rmse, r2, fwhm, ew = line.replace('\n', '').split('\t')

        obs_date_list.append(obs_date)
        start_list.append(float(start))
        end_list.append(float(end))
        rmse_list.append(float(rmse))
        r2_list.append(float(r2))
        fwhm_list.append(float(fwhm))
        ew_list.append(float(ew))
        centers_list.append(np.fromstring(centers.strip('[]'), sep=','))
        widths_list.append(np.fromstring(widths.strip('[]'), sep=','))
        amplitudes_list.append(np.fromstring(amplitudes.strip('[]'), sep=','))
        skews_list.append(np.fromstring(skews.strip('[]'), sep=','))

    return obs_date_list, start_list, end_list, rmse_list, r2_list, fwhm_list, ew_list, centers_list, widths_list, amplitudes_list, skews_list