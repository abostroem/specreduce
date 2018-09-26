import pytest
import numpy as np
from astropy.modeling import models
from ..trace import find_spectrum

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
    
def make_2d_spec_cube(amplitude=1.0, mean_1=None, stddev_1=None, 
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
    spec2d = np.ones((10,10))
    disp_axis = 1
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 2)
    assert cross_disp_axis == [0]

def test_find_non_disp_dim_2D_second():
    spec2d = np.ones((10,10))
    disp_axis = 0
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 2)
    assert cross_disp_axis == [1]

def test_find_non_disp_dim_3D():
    spec2d = np.ones((10,10, 10))
    disp_axis = 1
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 3)
    assert cross_disp_axis == [0,2]
    
def test_find_non_disp_dim_3D_second():
    spec2d = np.ones((10,10, 10))
    disp_axis = 0
    cross_disp_axis = trace.find_non_disp_dimensions(disp_axis, 3)
    assert cross_disp_axis == [1,2]



 

def test_find_spectrum():
    '''
    '''
    indx = 512
    image = make_2d_spec_image(amplitude=1.0, mean=indx, stddev=None, disp_size=1024, xdisp_size=1024,
                       seed=2, SNR=100)
    
    spec_indx = find_spectrum(image)
    
