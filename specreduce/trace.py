import numpy as np
from scipy.signal import medfilt
from astropy.modeling import fitting, models

from matplotlib import pyplot as plt

def find_spectrum(image, disp_axis=1):
    '''
    Locate the brightest object in a 2D spectrum

    Parameters
    ----------
    image : 2D array-like object
        2D spectrum overwhich to search for an object
    Returns
    -------
    spec_loc_indx : int
        the index of the approximate spectrum location
    TODO:  
    '''
    image_array = np.array(image)
    cross_disp_median = np.median(image_array, axis=disp_axis)
    smooth_cross_disp = medfilt(cross_disp_median, kernel_size=7)
    spec_loc_indx = np.argmax(smooth_cross_disp)
    if len(smooth_cross_disp[smooth_cross_disp==smooth_cross_disp[spec_loc_indx]])>1:
        spec_loc_indx = np.mean(np.where(smooth_cross_disp==smooth_cross_disp[spec_loc_indx]))
    spec_loc_indx = int(spec_loc_indx)
    return spec_loc_indx
    

    
    
    
    #Centroid - weighted mean
    #photutils does this in a 2D way
    #Future: use WCS



# Use a user defined aperture
# Collapse full spectrum to get approximate spectrum location
    #options: tell it a range where to look for the spectrum
    #Filter
    # find peak
    # Fit Gauss
# Fit gaussian to get center
# Collapse spectrum in bins
    # create a sampling number that combines number of pixels in dispersion and fit order
# Fit gaussian for each bin
# Fit smooth function to the peak of each bin
    #Change order of fit