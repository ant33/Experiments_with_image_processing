# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:17:06 2019

@author: Anton
"""

import numpy as np
from DiffOperators import div2D, grad, euclidian_norm2D, laplacian, heat_equation

def kernel_pm(xi, epsilon):
    return 1 / np.sqrt(epsilon + (xi) ** 2)

def DenoisingTikhonov(f, delta_t, K, lmbd):
    """
    Denoising method 
    
    Parameters:
    ===========
        f       -- gray image as [M x N] matrix 
        delta_t -- size of time step in numerical scheme
        K       -- number of iterations. 
                   If K==0, the function works until 
                   convergence of error to zero or 
                   until reach max iteration (max num of iter. = 6000 ))
                   Stop criteria: ||u^{k+1} - u^{k}|| / ||u^{k}||  < 10^{-5}
        lmbd   -- parameter of regularization
        
    Return:
    =======
        -- [M x N] matrix
    """
    f_in = f.copy()
    f_0 = f.copy()
    if K != 0: # fixed number of iteration
        for k in range(K):
            diverg = div2D(grad(f_in))
            f_in = f_in + delta_t * (lmbd * (f_0 - f_in)+diverg)   
        f_new = f_in
    else: # works until error reach zero
        lim_steps = 0
        diverg = div2D(grad(f_in))
        f_new = f_in + delta_t * (lmbd * (f_0 - f_in)+diverg)   
        while True:
            error = (np.sum(np.abs(f_in - f_new)**2) ** (1/2)) /\
                    np.sum(np.abs(f_in)**2) ** (1/2) 
            lim_steps += 1
            if error > 10**(-5) and lim_steps < 6000:
                f_in = f_new
                diverg = div2D(grad(f_in))
                f_new = f_in + delta_t * (lmbd * (f_0 - f_in)+diverg)
            else:
                break
    return f_new

def Denoise_TV(f, delta_t, K, lmbd, epsilon):
    """
    Denoising method 
    
    Parameters:
    ===========
        f -- gray image as [M x N] matrix 
        delta_t -- size of time step in numerical scheme
        K       -- number of iterations. 
                   If K==0, the function works until 
                   convergence of error to zero or 
                   until reach max iteration (max num of iter. = 6000 ))
                   Stop criteria: ||u^{k+1} - u^{k}|| / ||u^{k}||  < 10^{-5}
        lambd -- parameter of regularization
        epsilon -- parameter of kernel
        
    Return:
    =======
        -- [M x N] matrix
    """
    f_in = f.copy()
    f_0 = f.copy()
    if K != 0: # fixed number of iteration
        for k in range(K):
            kernel = kernel_pm(euclidian_norm2D(grad(f_in)), epsilon)
            diverg = div2D(grad(f_in) * kernel)
            f_in = f_in + delta_t * (lmbd * (f_0 - f_in)+diverg)       
        f_new = f_in
    else: # works until error reach zero
        lim_steps = 0
        kernel = kernel_pm(euclidian_norm2D(grad(f_in)), epsilon)
        diverg = div2D(grad(f_in) * kernel)
        f_new = f_in + delta_t * (lmbd * (f_0 - f_in)+diverg)   
        while True:
            error = (np.sum(np.abs(f_in - f_new)**2) ** (1/2)) /\
                    np.sum(np.abs(f_in)**2) ** (1/2) 
            lim_steps += 1
            if error > 10**(-5) and lim_steps < 6000:
                f_in = f_new
                kernel = kernel_pm(euclidian_norm2D(grad(f_in)), epsilon)
                diverg = div2D(grad(f_in) * kernel)
                f_new = f_in + delta_t * (lmbd * (f_0 - f_in)+diverg)
            else:
                break
            
    return f_new