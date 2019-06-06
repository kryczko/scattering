#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def step(x, center=5.0, width=1.0, max_h=1.0):
    V = np.zeros_like(x)
    indices = np.abs(x - 5.0) < width
    V[indices] += (np.ones(x.shape[0]) * max_h)[indices]
    return V

def gaussian(x, mu=5.0, sig=0.1, max_h=1.0):
    return max_h * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def calc_MM(E):
    V_L = 0.0
    V_R = 0.0

    N = 512
    L = 10.0

    A = 1.0 / np.sqrt(L)
    x = np.linspace(0, L, N+2)
    dx = x[1] - x[0]


    hbar = 1.0
    m = 1.0

    t = hbar**2 / 2. / m / dx**2

    k_L = np.sqrt( 2. * m * (E - V_L) / hbar**2 )
    k_R = np.sqrt( 2. * m * (E - V_R) / hbar**2 )

    # plt.plot(x - L, np.ones(N+2) * V_L)
    # plt.plot(x, step(x))
    # plt.plot(x + L, np.ones(N+2) * V_R)
    # plt.show()
    H = np.zeros((N+2, N+2), dtype=np.complex)

    H += np.diag(-2.0 * t * np.ones(N+2) - gaussian(x), 0) + np.diag(t * np.ones(N+1), 1) + np.diag(t * np.ones(N+1), -1)
    H[0][0] += t * np.exp(1j * k_L * dx)
    H[-1][-1] += t * np.exp(1j * k_R * dx)
    # plt.imshow(H.real)
    # plt.colorbar()
    # plt.show()

    # this is the main matrix
    M = E * np.diag(np.ones(N + 2)) + H

    # this is the source
    q = np.zeros(N + 2, dtype=np.complex)
    q[0] = t * A * (np.exp(1j * k_L * dx ) - np.exp(-1j * k_L * dx))

    psi = linalg.solve(M, q)
    # psi /= np.sqrt(np.sum(np.absolute(psi)**2))
    # plt.plot(x, np.absolute(psi)**2)
    # plt.show()

    v_in = hbar * k_L / m 
    v_out = hbar * k_R / m 

    trans = np.sqrt(v_out / v_in) * psi[-1] / A
    refle = (psi[0] - A) / A

    return np.vdot(trans, trans), np.vdot(refle, refle)

def calc_GF(E):
    V_L = 0.0
    V_R = 0.0

    N = 512
    L = 10.0

    A = 1.0 / np.sqrt(L)
    x = np.linspace(0, L, N+2)
    dx = x[1] - x[0]


    hbar = 1.0
    m = 1.0

    t = hbar**2 / 2. / m / dx**2

    k_L = np.sqrt( 2. * m * (E - V_L) / hbar**2 )
    k_R = np.sqrt( 2. * m * (E - V_R) / hbar**2 )

    H = np.zeros((N+2, N+2), dtype=np.complex)

    H += np.diag(2.0 * t * np.ones(N+2) + gaussian(x), 0) + np.diag(-t * np.ones(N+1), 1) + np.diag(-t * np.ones(N+1), -1)
    
    # tack on the self energies
    H[0][0] += t * np.exp(1j * k_L * dx)
    H[-1][-1] += t * np.exp(1j * k_R * dx)


    # this is the main matrix
    M = E * np.diag(np.ones(N + 2)) + H
    # this is the source
    q = np.zeros(N + 2, dtype=np.complex)
    q[0] = t * A * (np.exp(1j * k_L * dx ) - np.exp(-1j * k_L * dx))

    G_R = linalg.inv(M)

    print(G_R[0,-1], G_R[-1,0])

    v_in = hbar * k_L / m 
    v_out = hbar * k_R / m 

    trans = 1j * hbar * np.sqrt(v_out * v_in) * G_R[-1][-1] / dx
    T = np.vdot(trans, trans)

    return T, 1 - T


energies = np.linspace(0.0, 2.0, 32)
coeffs = [calc_MM(E) for E in energies]
coeffs = np.array(coeffs)
plt.plot(energies, coeffs[:,0], energies, coeffs[:,1])
plt.legend(['transmission', 'reflection'])
plt.show()