import numpy as np
from scipy.signal import medfilt
from astropy.modeling import fitting, models

from matplotlib import pyplot as plt

def find_non_disp_dimensions(disp_axis, ndim):
    '''
    Find the axis(es) that are not the dispersion axis
    
    Parameters
    ----------
    disp_axis : int
        dispersion axis number
    ndim : int
        number of dimensions of the input image
        
    Returns
    -------
    not_disp_dim : list
        list of non-dispersion dimensions
    '''
    not_disp_dim = []
    for dim in range(ndim):
        if dim == disp_axis:
            continue
        else:
            not_disp_dim.append(dim)
    return not_disp_dim

def find_spectrum(image, disp_axis=1):
    '''
    Locate the brightest object in a 2D spectrum

    Parameters
    ----------
    image : 2D or 3D array-like object
        2D or 3D spectrum overwhich to search for an object
    Returns
    -------
    spec_loc_indx : tup
        the index of the approximate spectrum location, if image is 3D this array is in 
        the same order as the dimensions of the spectrum, excluding the dispersion axis
    TODO: integrate ability to look at WCS attribute
    TODO: generalize to higher dimensions cubes
    '''
    disp_axis = int(disp_axis)
    image_array = np.array(image)
    cross_disp_median = np.median(image_array, axis=disp_axis)
    smooth_cross_disp = medfilt(cross_disp_median, kernel_size=5)
    spec_loc_indx = []
    if len(smooth_cross_disp.shape) == 1:
        mid_max_pt = np.median(np.where(smooth_cross_disp==smooth_cross_disp.max())[0])
        spec_loc_indx.append(mid_max_pt)
    else:
        marginal_1 = np.sum(smooth_cross_disp, axis=1)
        marginal_2 = np.sum(smooth_cross_disp, axis=0)
        #Median smoothing leads to the same (lower) value when near maximum when points 
        #are symmetric, take the lower one
        mid_max_pt1 = np.median(np.where(marginal_1==marginal_1.max())[0])
        mid_max_pt2 = np.median(np.where(marginal_2==marginal_2.max())[0])
        spec_loc_indx.append(mid_max_pt1)
        spec_loc_indx.append(mid_max_pt2)
    return tuple(spec_loc_indx)

def find_centroid(location, flux, axis=0):
    '''
    Find the centroid along the axis not specified of 2D array by finding a weighted
    sum of the location array weighted by the flux array for each point along the dispersion
    axis (sum over the axis specified)
    
    Parameters
    ----------
    location : 2D array-like
        the array of the location values to be centroided
    flux : 2D array-like
        the values of the flux to be used as weights in the centroiding
    axis: int
        axis to sum over (collapse over)
    
    Returns
    ----------
    centroid : 1D array 
        array of centroid positions along the dispersion axis
        
    '''
    location=np.array(location)
    flux = np.array(flux)
    centroid = np.sum(location*flux, axis=axis)/np.sum(flux, axis=axis)
    return centroid

    
def find_trace_pts(image, disp_axis=1, nsum_pts=11, nstep=11, aperture=20):
    '''
    Find discrete points along the trace of a spectrum by centroiding the trace
    In its current form, this function does not handle fractional pixels 
    
    Parameters
    ----------
    image : array-like, 2 or 3D
    disp_axis : int
        dispersion axis number (0, 1, or 2)
    nsum_pts: int, odd
        the number of points over which to calculate the median to determine the trace
    nstep: int
        the spacing of each trace point
    aperture : int
        the width used in the cross-dispersion direction over which the centroid is calculated.
        The aperture width is centered on approximate spectrum location. 
    
    Returns
    ----------
        
    
    TODO: support sampling differently along differnt axes?
    TODO: accept non-integer nsum_pts, nstep, aperture?
    TODO: require nstep > nsum_pts? so you are always sampling over more than your kernel size
    '''
    spec_loc_indx = find_spectrum(image, disp_axis=disp_axis)
    #median filter in the dispersion direction
    #find the median of a bin by median filtering with a kernel of nsum_pts and 
    #then choosing every nstep points
    kernel = np.int_(np.ones(len(image.shape)))
    kernel[disp_axis] = nsum_pts
    med_filt_image = medfilt(image, kernel_size=kernel)
    #Select the cross-dispersion centroid every nstep points in the dispersion direction
    #sample every nstep pts
    indx_start = int(np.ceil(nsum_pts/2))
    indx_end = int(np.ceil(image.shape[disp_axis]-nsum_pts/2))
    indx = np.int_(np.arange(indx_start, indx_end, nstep))
    fit_pts = np.take(med_filt_image, indices=indx, axis=disp_axis) 
    non_disp_axes = find_non_disp_dimensions(disp_axis, len(image.shape))
    if len(non_disp_axes) == 1:
        #spec_loc_indx and non_disp_axes are array-like objects but for this case should
        #both only have 1 element
        location = np.int_(np.arange(spec_loc_indx[0]-aperture/2., spec_loc_indx[0]+aperture/2.))
        flux = np.take(image, axis=non_disp_axes[0], indices=location)
        centroid = find_centroid(location, flux, axis=non_disp_axes[0])[indx]
    elif len(non_disp_axes) > 1: #current implementation limited to 2 non-dispersion axes
        #sum over first (index=0) non-dispersion direction, corresponds to calculating centroid in
        #second (index=1) non-dispersion direction
        image2D_1 = np.sum(image, axis=non_disp_axes[0]) 
        #sum over second non-dispersion direction, corresponds to calculating centroid in
        #first non-dispersion direction
        image2D_2 = np.sum(image, axis=non_disp_axes[1])
        # since collapsing to 2D, need to get non-disp dimension again
        non_disp_axis = find_non_disp_dimensions(disp_axis, len(image2D_1.shape))
        location1 = np.int_(np.arange(spec_loc_indx[1]-aperture/2., spec_loc_indx[1]+aperture/2.))
        location2 = np.int_(np.arange(spec_loc_indx[0]-aperture/2., spec_loc_indx[0]+aperture/2.))
        flux1 = np.take(image2D_1, axis=non_disp_axis[0], indices=location1)
        flux2 = np.take(image2D_2, axis=non_disp_axis[0], indices=location2)
        centroid1 = find_centroid(location1, flux1, axis=non_disp_axis[0])[indx]
        centroid2 = find_centroid(location2, flux2, axis=non_disp_axis[0])[indx]
        centroid = np.vstack((centroid2, centroid1))
    return indx, centroid

def find_trace_fit(indx, centroid, fit='legendre', order=2):
    '''
    Fit a model to points along the trace
    
    Parameters
    -----------
    indx: 1D array
        an array of pixels along the dipserion axis at which the spectrum location was calculated
    centroid: 1D or 2D array 
        an array of pixels that correspond to the location of the spectrum in 1D if there is
        one cross-disperion direction and 2D if there are two (a spectral cube)
    fit : str
        function to fit to centroid points. Must be 'legendre' or 'polynomial'
    order : int
        order of the function to be fit
    
    Returns
    -----------
    trace_fit: astropy LevMarLSQFitter object
        a fit in 1D or 2D (depending on dimensionality of centroid) to the spectrum location (trace)
        
    TODO: enable other fitting methods?
    '''
    if (fit == 'legendre') and (len(centroid) == 1):
        fit_model = models.polynomial.Legendre1D(degree=order)
    elif (fit == 'legendre') and (len(centroid) == 2):
        fit_model = models.polynomial.Legendre2D(degree=order)
    elif (fit == 'polynomial') and (len(centroid) == 1):
        fit_model = models.polynomial.Polynomial1D(degree=order)
    elif (fit == 'polynomial') and (len(centroid) == 2):
        fit_model = models.polynomial.Polynomial2D(degree=order)
    else:
        raise NotImplementedError('fit={} is not implemented. Fit must be '.format(fit) +
                                  '"legendre" or "polynomial"' )
    
    fit_method = fitting.LevMarLSQFitter()
    if len(centroid) == 1:
        trace_fit = fit_method(fit_model, indx, centroid[0])
    elif len(centroid) == 2:
        trace_fit = fit_method(fit_model, indx, centroid[0], centroid[1])
    return trace_fit    
