# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:08:39 2019

@author: Anton
"""

import numpy as np
from DiffOperators import div2D, grad, euclidian_norm2D

def kernel_ga(xi, alpha):
    # This kernel used for estimaiton of edge influeces
    return np.exp(-xi**2 / alpha**2)

def kernel_pm(xi, alpha):
    return 1 / np.sqrt(1 + (xi / alpha) ** 2)

def PeronaMalik(f, delta_t, K, alpha):
    """
    Denoising method 
    
    Parameters:
    ===========
        f -- gray image as [M x N] matrix 
        delta_t -- size of time step in numerical scheme
        K -- number of iterations
        alpha -- parameter of diffusion
    Return:
    =======
        -- [M x N] matrix
    """
    f_in = f.copy()
    for k in range(K):
        diverg = div2D(grad(f_in) * kernel_pm(f_in, alpha)**2)
        f_in = f_in + delta_t * diverg
        
    return f_in