# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:02:22 2021

@author: Esther Cruz Rico, Burak Mete
"""

from pennylane.optimize import NesterovMomentumOptimizer
import pennylane as qml
import pandas as pd
import numpy as np
from scipy.linalg import expm

def param_Hermitian(dim, params): 
    upper = np.zeros((dim, dim), dtype = np.complex64)

    for i in range(dim):
        for j in range(i+1, dim):
            upper[i,j] = params[i,j]+1j*params[j,i]

    hermitian = np.diag(np.diag(params))+upper+np.conj(upper.T)
    return hermitian

def exp_hermitian(H): 
    unitary = expm(1j*H)
    return unitary

def param_unitary(dim, params): 
    H = param_Hermitian(dim, params)
    U = exp_hermitian(H)
    return U

def compute_indices(loc, Nlayer):
    
    assert loc%2 == 0 # locality must be even
    N = loc*2**(Nlayer-1) # Number of qubits
    layers = []
    first_layer = [list(range(j,j+loc)) for j in range(0,N,loc)]
    layers.append(first_layer)
    prev_layer = first_layer
    for i in range(1,Nlayer):
        next_layer = []
        for j in range(0,len(prev_layer),2):
            next_layer.append(prev_layer[j][:loc//2] + prev_layer[j+1][:loc//2])
        layers.append(next_layer)
        prev_layer = next_layer
    return layers

def compute_indices_2d(loc, Nlayer):
    """Compute the wires in which the gates act in a 2D lattice
    
    loc: side of the sublattice in which the operator act
    Nlayer: depth of the circuit"""
    
    assert loc%2 == 0 # locality must be even
    layers = []
    qubit_side = loc* 2**(Nlayer-1)
    Nq = qubit_side**2 # number of qubits
    
    # first layer
    qubits = range(Nq)
    # arrange them in 2d array
    qubits = np.array(qubits).reshape((loc* 2**(Nlayer-1), loc* 2**(Nlayer-1)))
    
    first_layer = [qubits[i:i+loc, j:j+loc] for i in range(0, qubit_side, loc)                                             for j in range(0,qubit_side,loc)]
    layers.append(first_layer)
    
    # iterate to depth
    prev_layer = first_layer
    for i in range(1, Nlayer):
        layer_Nside = 2**(Nlayer-i)
        prev_layer_idx = np.array(range(len(prev_layer))).reshape(layer_Nside, layer_Nside)
        next_layer_idx = [prev_layer_idx[i:i+2, j:j+2] for i in range(0, layer_Nside, 2) for j in range(0,layer_Nside,2)]
        next_layer = []
        for idx in next_layer_idx:
            i1, i2, i3, i4 = idx.ravel()
            gate_qubits = np.zeros((loc, loc), dtype = np.int)
            gate_qubits[:loc//2, :loc//2] = prev_layer[i1][:loc//2,:loc//2] 
            gate_qubits[:loc//2, loc//2:] = prev_layer[i2][:loc//2,:loc//2] 
            gate_qubits[loc//2:, :loc//2] = prev_layer[i3][:loc//2,:loc//2] 
            gate_qubits[loc//2:, loc//2:] = prev_layer[i4][:loc//2,:loc//2] 
            next_layer.append(gate_qubits)
        layers.append(next_layer)
        prev_layer = next_layer
    
    return layers

def compute_indices_MPS(loc, n_qubits):
    
    assert loc%2 == 0 # locality must be even
    layers = [list(range(i,loc+i)) for i in range(0, n_qubits-loc+1)]
    return layers

    