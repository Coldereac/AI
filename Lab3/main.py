import numpy as np

T = np.array([[0.6], [0.2], [0.4]])
O = np.array([[0.2], [0.5], [0.3]])
W12 = np.array([[0.3, 0.7, 0.4, 0.2, 0.1, 0.8],
                [0.9, 0.1, 0.6, 0.5, 0.2, 0.7],
                [0.2, 0.3, 0.8, 0.4, 0.3, 0.5],
                [0.7, 0.6, 0.9, 0.1, 0.8, 0.6],
                [0.1, 0.5, 0.4, 0.3, 0.7, 0.5]])
W23 = np.array([[0.5, 0.3, 0.2, 0.7, 0.9],
                [0.7, 0.1, 0.8, 0.2, 0.6],
                [0.3, 0.8, 0.7, 0.1, 0.5],
                [0.9, 0.2, 0.6, 0.7, 0.3]])
W34 = np.array([[0.4, 0.1, 0.6, 0.7],
                [0.3, 0.5, 0.8, 0.9],
                [0.6, 0.4, 0.1, 0.2]])

E4 = T - O
E3 = np.dot(W34.T, E4)
E2 = np.dot(W23.T, E3)

print("Помилки 34: \n", E4)
print("Помилки 23: \n", E3)
print("Помилки 12: \n", E2)
