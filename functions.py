# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:00:49 2025

@author: mseeg
"""
# Packages
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from spectrum import Spectrum
#Loading files 
# Functions

def load_spectrum(path,target_name):
    file_names=os.listdir(path)
    data_list=[]

    for file in file_names:
        filepath=os.path.join(path,file) 
        data=np.loadtxt(filepath)
        data_list.append(data)

    data_arr=np.concatenate(data_list)
    wavelength_arr=data_arr[:,0]
    spectrum_arr=data_arr[:,1]
    #target_name = datafile.split('_')[0]

    return Spectrum(target_name, wavelength_arr, spectrum_arr)

def fileloader(path):

    file_names=os.listdir(path)
    data_list = []

    for file in file_names:
        if file.startswith('.'):
            continue

        filepath=os.path.join(path,file) 
        data = np.loadtxt(filepath)
        data_list.append(data)

    data_arr=np.concatenate(data_list)
    wavelength_arr= data_arr[:,0]
    spectrum_arr=data_arr[:,1]

    return wavelength_arr, spectrum_arr

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def ScaleData(data):
    return (data-np.mean(data))/np.sqrt(np.var(data))
def residual(spectrum,expected_spectrum):
    return spectrum-expected_spectrum
def query(wavelength,spectrum,center,query_range):

    query=np.where((((center-query_range)<wavelength) & (wavelength<(center+query_range))))

    selected_wavelength=wavelength[query]
    selected_spectrum=spectrum[query]
    return selected_wavelength,selected_spectrum

# We have to mix the gaussian with a linear continuum
def skewed_gauss(wavelength,centre,width,amplitude,skew,slope,c):
    exp_1=np.exp(-(((wavelength-centre)/(2*width))**2))
    exp_2=np.exp(-skew*np.arctan((wavelength-centre)/(width))) 
    continuum=((slope*wavelength)+c)
    skewed_gauss=amplitude*exp_1*exp_2
    return  continuum-skewed_gauss

def gaussian(wavelength,centre,width,amplitude,continuum,slope,c):
     exp_1=np.exp(-(((wavelength-centre)/(2*width))**2))
     continuum=((slope*wavelength)+c)
     return continuum-(amplitude*exp_1)
# Make a double 'mixing' gaussian    
def double_gaussian(wavelength,centre1,centre2,width1,width2,amplitude1,amplitude2,slope,c):
    exp_1=amplitude1*np.exp(-(((wavelength-centre1)/(2*width1))**2))
    exp_2=amplitude2*np.exp(-(((wavelength-centre2)/(2*width2))**2))
    gaussian=(exp_1+exp_2)
    continuum=((slope*wavelength)+c)
    return continuum-gaussian
def FWHM(fit_std):
    return 2 * np.sqrt(2 * np.log(2)) * fit_std
    
def dib_finder(wavelength,spectrum,locs,wave_range,threshold=None,promin=0.03,index_search=False,ref_wavelengths=None):
    dib_wavelengths=[]
    dib_spectra=[]
    if threshold!=None:
        
        dib_locs,_=find_peaks(-spectrum,height=(threshold-1),prominence=promin)
        for dib in range(dib_locs.size):
            dib_wavelength,dib_spectrum=query(wavelength,spectrum,locs[dib],wave_range)
            dib_wavelengths.append(dib_wavelength)
            dib_spectra.append(dib_spectrum)
            return dib_wavelengths,dib_spectra,dib_locs
    else:
        if index_search!=False:
            centra_list=[]
            for dib in range(len(locs)):
                dib_wavelength,dib_spectrum=query(wavelength,spectrum,ref_wavelengths[locs[dib]],wave_range)
                dib_wavelengths.append(dib_wavelength)
                dib_spectra.append(dib_spectrum)
                centra_list.append(wavelength[locs[dib]])
            return dib_wavelengths,dib_spectra,centra_list
        else:
            for dib in range(len(locs)):
                dib_wavelength,dib_spectrum=query(wavelength,spectrum,locs[dib],wave_range)
                dib_wavelengths.append(dib_wavelength)
                dib_spectra.append(dib_spectrum)
            return dib_wavelengths,dib_spectra
    if index_search!=False:
        centra_list=[]
        for dib in range(len(locs)):
            dib_wavelength,dib_spectrum=query(wavelength,spectrum,wavelength[locs[dib]],wave_range)
            dib_wavelengths.append(dib_wavelength)
            dib_spectra.append(dib_spectrum)
            centra_list.append(wavelength[locs[dib]])
        return dib_wavelengths,dib_spectra,centra_list
        

def fitter(wavelengths,spectra,centra,model,p0):         
        #p0.insert(0,centra[i])
    params,covariance=curve_fit(model,wavelengths,spectra,p0=p0,maxfev=10000)
    
    return params
def fitter_plotter(wavelengths,spectra,model,centra_list,p0_list,bounds_list,residues=False,error_model=False):
    param_list=[]
    prediction_list=[]
    for i in range(len(wavelengths)):
        slope=lin_continuum(wavelengths[i],spectra[i])
        c=wavelengths[i][0]
        p0_list[i].append(slope)
        p0_list[i].append(c)
        params,_=curve_fit(model,wavelengths[i],spectra[i],p0=p0_list[i],maxfev=10000,bounds=bounds_list[i])
        param_list.append(params)
        prediction=model(wavelengths[i],*param_list[i])
        prediction_list.append(prediction)
    if residues==True:
        residu_list=[]
        residu_stat_list=[]
        for i in range(len(wavelengths)):
            residu=residual(spectra[i],prediction_list[i])
            residu_list.append(residu)
            residu_mean=np.mean(residu)
            residu_variance=np.var(residu)
            residu_sigma=np.sqrt(residu_variance)
            residu_stat_list.append([residu_mean,residu_variance,residu_sigma,rmse(residu,wavelengths[i])])
        residu_arr=np.array(residu_list)
        if error_model!=False:
            stat_list=[]
            for i in range(len(wavelengths)):
                stat=error_model(spectra[i],prediction_list[i])
                stat_list.append(stat)
            return param_list,prediction_list,residu_arr,residu_stat_list,stat_list
        else:
            return param_list,prediction_list,residu_arr,residu_stat_list
    else:
        if error_model!=False:
            stat_list=[]
            for i in range(len(wavelengths)):
                stat=error_model(spectra[i],prediction_list[i])
                stat_list.append(stat)
            return param_list,prediction_list,stat_list
        else:
        
            return param_list,prediction_list
def subplotter(wavelengths,cols):
    rows=len(wavelengths)//cols
    if len(wavelengths) %cols!=0:
        rows+=1
    position=range(1,len(wavelengths)+1)

    return rows,position

def lin_continuum(wavelength,spectrum,intercept=False):
    start=spectrum[0]
    end=spectrum[-1]
    slope=(end-start)/wavelength.size
    if intercept!=False:
        c=spectrum[0]
        return slope,c
    else:
        
        return slope
def line(wavelength,spectrum):
    slope,c=lin_continuum(wavelength,spectrum,intercept=True)
    # Hacky solution
    wavelength_indices=np.arange(len(wavelength))
    continuum=(slope*wavelength_indices)+c
    return continuum
def detrender(wavelength,spectrum):
    norm_spectrum=spectrum/line(wavelength,spectrum)
    return norm_spectrum
def min_finder(wavelengths,spectrum):
    argmin=np.argmin(spectrum)
    min_wavelength=wavelengths[argmin]
    min_spectrum=spectrum[argmin]
    return min_wavelength
def rmse(residuals,wavelengths):
    tot=np.sum(residuals**2)
    return np.sqrt(tot/wavelengths.size)

def quickplot(wavelength,spectrum,cols,s=2,title='none',fit=False):
    fig=plt.figure()
    rows,position=subplotter(wavelength,cols)
    # Supply fits as a list
    if fit!=False:
        for i in range(len(wavelength)):
            ax=fig.add_subplot(rows,cols,position[i])
            ax.scatter(wavelength[i],spectrum[i],s=s)
            ax.plot(wavelength[i],fit[i],color='red')
            ax.set_xlabel('Wavelength ($\\AA$)')
            ax.set_ylabel('Norm. Intensity')    
            ax.grid()
        fig.suptitle(t=title)
        fig.tight_layout()
        fig.show()
        return fig
    else:
        for i in range(len(wavelength)):
            ax=fig.add_subplot(rows,cols,position[i])
            ax.scatter(wavelength[i],spectrum[i],s=s)
            ax.set_xlabel('Wavelength ($\\AA$)')
            ax.set_ylabel('Norm. Intensity')    
            ax.grid()
        fig.suptitle(t=title)
        fig.tight_layout()
        fig.show()
        return fig


def remove_continuum(wavelength, spectrum, ax = None, spline_order = None):
    if spline_order is None:
        params, _ = curve_fit(lambda wvl, order: np.poly1d(np.polyfit(wvl, spectrum, order))(wvl), wavelength, spectrum, p0=[1])
        spline_order = params[0]
        print(f'Spline order: {spline_order}')

    continuum = np.poly1d(np.polyfit(wavelength, spectrum, deg=spline_order))(wavelength)
    
    if ax is not None:
        ax.plot(wavelength, spectrum, '.', ms=2)
        ax.plot(wavelength, continuum, label='Continuum')

    return spectrum / continuum

# Machine Learning related functions

def train_rf(rf: RandomForestRegressor, spectra:List[Spectrum]) -> RandomForestRegressor:
    wavelength = np.concatenate([spectrum.wavelength for spectrum in spectra])[:, np.newaxis]
    flux = np.concatenate([spectrum.flux for spectrum in spectra])

    rf.fit(wavelength, flux)

def plot_rf_prediction(rf: RandomForestRegressor, test_spectrum: Spectrum):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)

    predicted_flux = rf.predict(test_spectrum.wavelength[:, np.newaxis])
    rmse = mean_squared_error(test_spectrum.flux, predicted_flux,squared=False)

    test_spectrum.plot(axes[0])
    axes[1].plot(test_spectrum.wavelength, predicted_flux, '.', ms=2)
    axes[2].plot(test_spectrum.wavelength, test_spectrum.flux - predicted_flux, '.', ms=2)

    axes[0].set_title('Observed spectrum')
    axes[1].set_title('Predicted spectrum')
    axes[2].set_title(f'Difference (RMSE={rmse:.4g})')

    fig.suptitle(f'RF prediction for {test_spectrum.target}')
    fig.tight_layout()
    return fig,rmse

def cross_validation(spectra: List[Spectrum], n_estimators=100):
    # Start with 0 estimators because it gets increased in the for-loop straightaway
    rf = RandomForestRegressor(n_estimators=0, warm_start=True, random_state=27)
    #spectra_tqdm = tqdm(spectra)

    for spectrum in spectra:
        training_spectra = [spec for spec in spectra if spec is not spectrum]
        rf.n_estimators += n_estimators

        #spectra_tqdm.set_description(f'Training on {len(training_spectra)} spectra: {', '.join([ts.target for ts in training_spectra])}')
        train_rf(rf, training_spectra)
        fig,rmse=plot_rf_prediction(rf, spectrum)
        return fig,rmse



    