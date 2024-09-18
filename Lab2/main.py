import numpy as np

# Матрица I
I = np.array([
    [0.4],
    [0.2],
    [0.6],
    [0.5]
])

# Матрица W12
W12 = np.array([
    [0.2, 0.6, 0.5, 0.3],
    [0.1, 0.8, 0.7, 0.2],
    [0.5, 0.3, 0.9, 0.7],
    [0.4, 0.6, 0.1, 0.8],
    [0.7, 0.9, 0.5, 0.6],
    [0.6, 0.3, 0.7, 0.2]
])

# Матрица W23
W23 = np.array([
    [0.6, 0.3, 0.7, 0.4, 0.2, 0.5],
    [0.7, 0.5, 0.8, 0.1, 0.9, 0.4],
    [0.3, 0.6, 0.2, 0.7, 0.8, 0.3],
    [0.4, 0.9, 0.5, 0.2, 0.1, 0.7],
    [0.5, 0.7, 0.8, 0.6, 0.5, 0.2]
])

# Матрица W34
W34 = np.array([
    [0.7, 0.6, 0.1, 0.5, 0.4],
    [0.8, 0.2, 0.3, 0.9, 0.1],
    [0.4, 0.9, 0.7, 0.5, 0.2]
])

sigmoida = lambda x: 1 / (1 + np.exp(-x))

X2 = np.dot(W12, I)

O2 = sigmoida(X2)

X3 = np.dot(W23, O2)

O3 = sigmoida(X3)

X4 = np.dot(W34, O3)

O4 = sigmoida(X4)

print(np.round(O4, 4))
