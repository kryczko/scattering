#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def step(x, center=5.0, width=1.0, max_h=1.0):
    V = np.zeros_like(x)
    indices = np.abs(x - 5.0) < width
    V[indices] += (np.ones(x.shape[0]) * max_h)[indices]
    return V


# along x
N = 128
# along y
M = 128

L = 10.0
x = np.linspace(0, L, N+2)
dx = x[1] - x[0]

hbar = 1.0
m = 1.0

t = hbar**2 / 2.0 / m / dx**2
V_lead = 0.5
h = 2.0 * t 
tol = 1e-4

def getModes(E):
    H = np.zeros((N, N), dtype=np.complex)

    H += np.diag(2.0 * t * np.ones(N) + V_lead, 0) + np.diag(-t * np.ones(N-1), 1) + np.diag(-t * np.ones(N-1), -1)

    V = np.diag(t * np.ones(N))

    I = np.diag(np.ones(N))
    Z = np.zeros((N, N))

    A = np.concatenate((np.concatenate((Z, I), axis=0), np.concatenate((-V, E*I - H), axis=0)), axis=1)

    B = np.concatenate((np.concatenate((I, Z), axis=0), np.concatenate((Z, V.T), axis=0)), axis=1)
    eigs, modes = linalg.eig(A, B)
    return eigs, modes

def sortModes(eigs, modes):
    prop_right = []
    prop_left = []
    evan = []

    for i in range(eigs.shape[0]):
        eig = eigs[i]
        mode = modes[i, :]
        val = np.absolute(eig)
        if val < 1. or val > 1.:
            evan.append((eig, mode))
        elif val - 1.0 < tol:
            curr = 2 * (np.vdot(mode.T * eig, np.dot(np.diag(-t * np.ones(mode.shape[0])), mode)) ).imag
            if curr < 0.0:
                prop_left.append((eig, mode))
            else:
                prop_right.append((eig, mode))
        else:
            print('Found mode that is somehow not propagating nor evanescent... val =', val)

    return np.array(prop_right), np.array(prop_left), np.array(evan)

def normalizeAndTruncateModes(modes):
    new_modes = []
    for i in range(modes.shape[1]):
        new_modes.append(modes[:N, i] / np.sqrt(np.sum(np.absolute(modes[:N, i])**2)))
    return np.array(new_modes)

def findDualModes(modes):
    duals = []
    for i in range(modes.shape[0]):
        b = np.zeros(modes.shape[0])
        b[i] = 1.
        sol, _, _, _ = linalg.lstsq(modes, b)
        duals.append(sol)
    return np.array(duals)

def computeBlochMatrix(eigs, modes, dual_modes, index=1):
    outer = np.zeros((modes.shape[0], modes.shape[0]))
    for i in range(eigs.shape[0])
        eig = eigs[i]
        mode = modes[i, :]
        dual_mode = dual_modes[i, :]

        outer += eig**index * np.outer(mode, dual_mode)
    return outer 

for E in np.linspace(0, 2, 10):
    eigs, modes = getModes(E)
    modes = normalizeAndTruncateModes(modes)
    dual_modes = findDualModes(modes)
    prop_right, prop_left, evan = sortModes(eigs, modes)


    F_1 = computeBlochMatrix(eigs, modes, dual_modes)




