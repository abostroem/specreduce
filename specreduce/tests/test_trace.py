import pytest
import numpy as np
from astropy.modeling import models
from .. import trace

def make_2d_spec_image(amplitude=1.0, mean=None, stddev=None, disp_size=1024, xdisp_size=1024,
                       seed=2, SNR=100):
    random_state = np.random.RandomState(seed=seed)
    if mean is None:
        mean = xdisp_size//2.
    if stddev is None:
        stddev = mean*0.2
    gauss = models.Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev)
    profile = gauss(np.arange(xdisp_size))
    image = np.tile(profile, (disp_size, 1)) + random_state.randn(disp_size, xdisp_size)*amplitude/SNR
    return image
    
def make_3d_spec_cube(amplitude=1.0, mean_1=None, stddev_1=None, 
                      mean_2=None, stddev_2=None,
                      disp_size=1024, xdisp_size_1=1024, xdisp_size_2=1024, 
                      seed=2, SNR=100):
    random_state = np.random.RandomState(seed=seed)
    if mean_1 is None:
        mean_1 = xdisp_size_1//2.
    if stddev_1 is None:
        stddev_1 = mean_1*0.2
    if mean_2 is None:
        mean_2 = xdisp_size_2//2.
    if stddev_2 is None:
        stddev_2 = mean_2*0.2
    disp_indx = np.arange(disp_size)
    xdisp_indx1 = np.arange(xdisp_size_1)
    xdisp_indx2 = np.arange(xdisp_size_2)
    cube =  np.zeros((disp_size, xdisp_size_1, xdisp_size_2), dtype=np.float64)
    gauss2D = models.Gaussian2D(amplitude=amplitude, x_mean=mean_1, x_stddev=stddev_1,
                               y_mean=mean_2, y_stddev=stddev_2)
    
    xdisp_1_pixs, xdisp_2_pixs = np.indices((xdisp_size_1, xdisp_size_2))
    image = gauss2D(xdisp_1_pixs, xdisp_2_pixs)
    cube = np.tile(image, (disp_size, 1, 1)) + random_state.randn(disp_size, xdisp_size_1, xdisp_size_2)*amplitude/SNR 
    return cube

########################
# TESTS
########################
def test_find_non_disp_dim_2D():
    '''
    Test basic functionality for an even number of pixels in 2D, disp axis=0
    '''
    spec2d = np.ones((10,10))
    disp_axis = 1
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 2)
    assert cross_disp_axis == [0]

def test_find_non_disp_dim_2D_second():
    '''
    Test basic functionality for an odd number of pixels in 2D, disp axis = 0
    '''
    spec2d = np.ones((10,10))
    disp_axis = 0
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 2)
    assert cross_disp_axis == [1]

def test_find_non_disp_dim_3D():
    '''
    Test basic functionality for an even number of pixels in 3D, disp_axis = 1
    '''
    spec2d = np.ones((10,10, 10))
    disp_axis = 1
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 3)
    assert cross_disp_axis == [0,2]
    
def test_find_non_disp_dim_3D_second():
    '''
    Test basic functionality for an even number of pixels, disp_axis = 0
    '''
    spec2d = np.ones((10,10, 10))
    disp_axis = 0
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 3)
    assert cross_disp_axis == [1,2]
    
def test_find_spectrum_axis0_2D():
    '''
    Test that spectrum is found for 2D, even number of pixels, disp_axis=0
    '''
    xdisp_size=50
    spec2d = make_2d_spec_image(disp_size=100, xdisp_size=xdisp_size)
    spec_loc_indx = trace.find_spectrum(spec2d, disp_axis=0)
    assert spec_loc_indx==(float(xdisp_size//2),)
    
def test_find_spectrum_axis0_2D_2():
    '''
    Test that spectrum is found for 2D, odd number of pixels, disp_axis=0
    '''
    xdisp_size=51
    spec2d = make_2d_spec_image(disp_size=100, xdisp_size=xdisp_size)
    spec_loc_indx = trace.find_spectrum(spec2d, disp_axis=0)
    assert spec_loc_indx==(float(xdisp_size//2),)
    
def test_find_spectrum_axis1_2D():
    '''
    Test that spectrum is found for 2D, even number of pixels, disp_axis=1
    '''
    xdisp_size=50
    spec2d = make_2d_spec_image(disp_size=100, xdisp_size=xdisp_size).T
    spec_loc_indx = trace.find_spectrum(spec2d, disp_axis=1)
    assert spec_loc_indx==(float(xdisp_size//2),)

def test_find_spectrum_axis0_3D():
    '''
    Test that spectrum is found for 3D, even number of pixels, disp_axis=0
    '''
    xdisp_size_1=50
    xdisp_size_2=50
    spec2d = make_3d_spec_cube(disp_size=100, xdisp_size_1=xdisp_size_1, xdisp_size_2=xdisp_size_2)
    spec_loc_indx = trace.find_spectrum(spec2d, disp_axis=0)
    assert spec_loc_indx==(float(xdisp_size_1//2),float(xdisp_size_2//2))

def test_find_spectrum_axis0_3D_2():
    '''
    Test that spectrum is found for 3D, even number of pixels, disp_axis=0, different spatial dims
    '''
    xdisp_size_1=50
    xdisp_size_2=60
    spec2d = make_3d_spec_cube(disp_size=100, xdisp_size_1=xdisp_size_1, xdisp_size_2=xdisp_size_2)
    spec_loc_indx = trace.find_spectrum(spec2d, disp_axis=0)
    assert spec_loc_indx==(float(xdisp_size_1//2),float(xdisp_size_2//2))

def test_find_spectrum_axis2_3D():
    '''
    Test that spectrum is found for 3D, even number of pixels, disp_axis=2
    '''
    xdisp_size_1=50
    xdisp_size_2=60
    spec2d = np.transpose(make_3d_spec_cube(disp_size=100, xdisp_size_1=xdisp_size_1, xdisp_size_2=xdisp_size_2))
    spec_loc_indx = trace.find_spectrum(spec2d, disp_axis=2)
    assert spec_loc_indx==(float(xdisp_size_2//2),float(xdisp_size_1//2))

def test_centroid():
    '''
    Test that centroid is found for 2D, even number of pixeks disp_axis=0
    '''
    xdisp_size=50
    disp_size=100
    spec2d = make_2d_spec_image(xdisp_size=xdisp_size, disp_size=disp_size)
    location = np.tile(np.arange(xdisp_size), (disp_size, 1))
    centroid = trace.find_centroid(location, spec2d, axis=1)
    assert np.isclose(centroid, np.ones(centroid.shape)*xdisp_size//2, rtol=1.0).all()
    
def test_centroid_2():
    '''
    Test that centroid is found for 2D, even number of pixeks disp_axis=1
    '''
    xdisp_size=50
    disp_size=100
    spec2d = make_2d_spec_image(xdisp_size=xdisp_size, disp_size=disp_size).T
    location = np.tile(np.arange(xdisp_size), (disp_size, 1)).T
    centroid = trace.find_centroid(location, spec2d, axis=0)
    assert np.isclose(centroid, np.ones(centroid.shape)*xdisp_size//2, rtol=1.0).all()
    
