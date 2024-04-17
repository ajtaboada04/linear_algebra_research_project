# represent shrodinger's equation in python

import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0
m = 1.0
omega = 1.0

# Grid
x = np.linspace(-5, 5, 1000)
dx = x[1] - x[0]

# Potential
V = 0.5 * m * omega**2 * x**2

# Kinetic energy operator
T = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        if i == j:
            T[i, j] = -2
        elif i == j - 1 or i == j + 1:
            T[i, j] = 1
T = -0.5 * (hbar**2 / m) * T / dx**2

# Hamiltonian
H = T + np.diag(V)

# Eigenvalues and eigenvectors
E, psi = np.linalg.eigh(H)

# Plot
plt.figure()
plt.plot(x, V, label='V(x)')

for i in range(5):
    plt.plot(x, E[i] + psi[:, i], label=f'E_{i}')
plt.legend()
plt.show()
