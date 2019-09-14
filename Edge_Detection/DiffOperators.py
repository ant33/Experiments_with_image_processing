# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:24:24 2019

@author: Anton Platov
Here discribed numerical diff operators on 2d matrices
"""

import numpy as np


def grad_x(matrix_in): 
    # Gradient on X (forward difference)
    matrix = matrix_in.copy()
    matrix[:-1,:] = - matrix[:-1, :] + matrix[1:,:]
    matrix[-1,:] = np.zeros(matrix[-1,:].shape[0])
    return matrix


def grad_y(matrix_in):
    # Gradient on Y (forward difference)
    matrix = matrix_in.copy()
    matrix[:,:-1] = - matrix[:,:-1] + matrix[:,1:]
    matrix[:,-1] = np.zeros(matrix[:,-1].shape[0])
    return matrix


def partial_x(matrix_in):
    # partial differential on X (forward difference)
    matrix = matrix_in.copy()
    matrix[1:,:] = matrix[1:,:]-matrix[:-1,:]
    matrix[0,:] = matrix_in[0,:]
    matrix[-1,:] = -matrix_in[-2,:]
    return matrix
    

def partial_y(matrix_in):
    # partial differential on X (forward difference)
    matrix = matrix_in.copy()
    matrix[:, 1:] = matrix[:, 1:]-matrix[:, :-1]
    matrix[:, 0] = matrix_in[:, 0]
    matrix[:, -1] = -matrix_in[:, -2]
    return matrix


def grad(P):
    # returns 2d vector
    fx = grad_x(P)
    fy = grad_y(P)
    return [fx, fy]


def div2D(p):
    """
    Divergence of 2D vector p
    return
        -- dp_1/dx + dp_2/dy
    """
    # p = [p_1, p_2]
    return partial_x(p[0]) + partial_y(p[1])


def laplacian(matrix):
    # Return laplace of 2D matrix.
    return div2D(grad(matrix))


def heat_equation(f, delta_t, K):
    """
    f -- gray image as 2d matrix;
    delta_t -- time step;
    K -- number of iterations;
    Used explicit Euler scheme of order 1 in time.
    """
    new_image = f.copy()
    for k in range(K):
        new_image += delta_t * laplacian(new_image)
    return new_image


