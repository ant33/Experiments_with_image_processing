# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:37:26 2019

@author: Anton
"""

import numpy as np
from DiffOperators import grad, laplacian

def euclidian_norm2D(v):
    # Here v = [v0, v1], v0, v1 -- matrixs
    return (v[0]**2+v[1]**2)**.5

def grad_edge(f, eta):
    """
    Contours in image 
    Parameters:
    ==========
    f -- gray image as matrix
    eta -- treshold, that is nonegative
    Returns:
    ========
        matrix with elements in {0, 1}
    """
    grads = euclidian_norm2D(grad(f))
    return 1-(grads < eta).astype(int)

def laplace_edge(f_in):
    """
    Function is finding coordinates (i, j) pixels where is of laplace extremum.
    Parameters:
    ==========
    f_in -- gray picture as matrix with size (M x N)
    Return:
    =======
         -- matrix with size (M x N) with elements in {0, 1}
    """ 
    f = laplacian(f_in.copy())
    
    edges = np.zeros(f.shape)
    mask_x = np.zeros(f.shape)
    edges[:,:-1] = f[:,:-1]*f[:,1:] 
    mask_x[:,:-1] = (edges < 0)[:,:-1]
    
    edges = np.zeros(f.shape)
    mask_y = np.zeros(f.shape)
    edges[:-1,:] = f[:-1,:]*f[1:,:] 
    mask_y[:-1,:] = (edges < 0)[:-1, :]
    
    return (mask_x.astype(bool) + mask_y.astype(bool)).astype(int)

def MarrHildreth(f_in,  eta):
    """
    Edge detector based on using grad_laplace and grad_edge
    Parameters:
    ==========
    f -- gray image as matrix
    eta -- treshold, that is nonegative
    Returns:
    ========
        matrix with elements in {0, 1}
    """
    f = f_in.copy()
    return laplace_edge(f)*grad_edge(f, eta)
    
    