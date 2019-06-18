#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def step(x, center=5.0, width=1.0, max_h=1.0):
    V = np.zeros_like(x)
    indices = np.abs(x - 5.0) < width
    V[indices] += (np.ones(x.shape[0]) * max_h)[indices]
    return V

N = 1024
L = 10.0
x = np.linspace(0, L, N+2)
dx = x[1] - x[0]

hbar = 1.0
m = 1.0

t = hbar**2 / 2.0 / m / dx**2
V_lead = 0.5
h = 2.0 * t 

def roots(E):
    roots = np.roots([-t, (E - (h + V_lead)), -t])
    return roots

Es = np.linspace(0, 1., 128)

def sort(roots):
    prop_left = []
    prop_right = []
    evan = []

    for r in roots:
        part_cur = (2. * np.conj(r) * t * (r**-1.)).imag
        if part_cur == 0:
            evan.append(r)
        elif part_cur > 0:
            prop_right.append(r)
        elif part_cur < 0:
            prop_left.append(r)
    if len(prop_right) == 0:
        print('No propagating modes were found!!')
        return None, None
    return prop_right[0], prop_left[0]

Es = np.linspace(0, 1, 10)

for E in Es:
    rs = roots(E)
    mode_r, mode_l = sort(rs)

    if mode_r is not None and mode_l is not None:

        H = np.zeros((N+2, N+2), dtype=np.complex)

        H += np.diag((h + V_lead) * np.ones(N+2), 0) + np.diag(-t * np.ones(N+1), 1) + np.diag(-t * np.ones(N+1), -1)
        H[0][0] += h - t / mode_l
        H[-1][-1] += h - t * mode_r 

        M = E * np.diag(np.ones(N + 2)) + H


        q = np.zeros(N + 2, dtype=np.complex)
        q[0] = -t * (1 / mode_l - 1 / mode_r) 

        psi = linalg.solve(M, q)

        t = psi[-1]
        print(np.absolute(t)**2)
