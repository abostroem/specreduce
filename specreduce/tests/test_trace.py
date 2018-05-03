import pytest
import numpy as np
from astropy.modeling import models
from ..trace import find_spectrum

def test_find_spectrum():
    '''
    '''
    indx = 512
    gauss = models.Gaussian1D(amplitude=10.0, mean=0, stddev=5)
    profile = gauss(np.arange(-indx, indx))
    image = np.tile(profile, (1024, 1)).T
    spec_indx = find_spectrum(image)
    assert spec_indx == indx
