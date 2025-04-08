import numpy as np

def skewed_gauss(wavelength,centre,width,amplitude,skew,slope=0,c=1):
    exp_1=np.exp(-(((wavelength-centre)/(2*width))**2))
    exp_2=np.exp(-skew*np.arctan((wavelength-centre)/(width))) 
    continuum=((slope*wavelength)+c)
    skewed_gauss=amplitude*exp_1*exp_2
    return  continuum-skewed_gauss

def double_gaussian(wavelength,centre1,width1,amplitude1,skew1,centre2,width2,amplitude2,skew2,slope,c):
    exp_1=skewed_gauss(wavelength, centre1,width1,amplitude1,skew1)
    exp_2=skewed_gauss(wavelength, centre2,width2,amplitude2,skew2)
    gaussian=(exp_1+exp_2)
    continuum=((slope*wavelength)+c)
    return continuum-gaussian