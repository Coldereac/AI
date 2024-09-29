# import numpy as np
#
#
# class NeuralNetwork:
#
#     def __init__(self, lr=0.5):
#         self.W12 = np.array([[0.3, 0.8, 0.7, 0.2],
#                              [0.6, 0.9, 0.1, 0.4],
#                              [0.4, 0.5, 0.6, 0.9],
#                              [0.8, 0.2, 0.3, 0.7],
#                              [0.5, 0.7, 0.8, 0.1]])
#
#         self.W23 = np.array([[0.7, 0.3, 0.5, 0.4, 0.9],
#                              [0.5, 0.2, 0.8, 0.3, 0.1],
#                              [0.8, 0.6, 0.3, 0.5, 0.2],
#                              [0.3, 0.7, 0.4, 0.1, 0.9],
#                              [0.4, 0.8, 0.6, 0.2, 0.7],
#                              [0.9, 0.4, 0.2, 0.6, 0.1]])
#
#         self.W34 = np.array([[0.5, 0.2, 0.7, 0.1, 0.8, 0.4],
#                              [0.7, 0.3, 0.8, 0.9, 0.1, 0.6],
#                              [0.4, 0.5, 0.2, 0.7, 0.9, 0.3]])
#
#         self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
#         self.lr = lr
#
#     def calculateLayer(self, W: np.array, I: np.array):
#         return self.sigmoid(np.dot(W, I))
#
#     def calculateAllLayers(self, I: np.array):
#         O2 = self.calculateLayer(self.W12, I)
#         O3 = self.calculateLayer(self.W23, O2)
#         O4 = self.calculateLayer(self.W34, O3)
#         return O2, O3, O4
#
#     def calculate(self, I: np.array):
#         O2 = self.calculateLayer(self.W12, I)
#         O3 = self.calculateLayer(self.W23, O2)
#         O4 = self.calculateLayer(self.W34, O3)
#         return O4
#
#     def train(self, T: np.array, I: np.array):
#         O2, O3, O4 = self.calculateAllLayers(I)
#
#         # Помилки на кожному рівні
#         E4 = T - O4
#         E3 = np.dot(self.W34.T, E4) * O3 * (1 - O3)
#         E2 = np.dot(self.W23.T, E3) * O2 * (1 - O2)
#
#         # Корекція ваг
#         dW34 = self.lr * np.dot(E4 * O4 * (1 - O4), O3.T)
#         dW23 = self.lr * np.dot(E3 * O3 * (1 - O3), O2.T)
#         dW12 = self.lr * np.dot(E2 * O2 * (1 - O2), I.T)
#
#         self.W34 += dW34
#         self.W23 += dW23
#         self.W12 += dW12
import numpy as np


class NeuralNetwork:

    def __init__(self, lr=0.5):
        self.W12 = np.array([[0.3, 0.8, 0.7, 0.2],
                             [0.6, 0.9, 0.1, 0.4],
                             [0.4, 0.5, 0.6, 0.9],
                             [0.8, 0.2, 0.3, 0.7],
                             [0.5, 0.7, 0.8, 0.1]])

        self.W23 = np.array([[0.7, 0.3, 0.5, 0.4, 0.9],
                             [0.5, 0.2, 0.8, 0.3, 0.1],
                             [0.8, 0.6, 0.3, 0.5, 0.2],
                             [0.3, 0.7, 0.4, 0.1, 0.9],
                             [0.4, 0.8, 0.6, 0.2, 0.7],
                             [0.9, 0.4, 0.2, 0.6, 0.1]])

        self.W34 = np.array([[0.5, 0.2, 0.7, 0.1, 0.8, 0.4],
                             [0.7, 0.3, 0.8, 0.9, 0.1, 0.6],
                             [0.4, 0.5, 0.2, 0.7, 0.9, 0.3]])

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.lr = lr

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def calculateLayer(self, W, I):
        return self.sigmoid(np.dot(W, I))

    def calculate(self, I):
        O2 = self.calculateLayer(self.W12, I)
        O3 = self.calculateLayer(self.W23, O2)
        O4 = self.calculateLayer(self.W34, O3)
        return O4

    def train(self, T, I):
        O2 = self.calculateLayer(self.W12, I)
        O3 = self.calculateLayer(self.W23, O2)
        O4 = self.calculateLayer(self.W34, O3)

        E4 = T - O4
        delta4 = E4 * self.sigmoid_derivative(O4)

        E3 = np.dot(self.W34.T, delta4)
        delta3 = E3 * self.sigmoid_derivative(O3)

        E2 = np.dot(self.W23.T, delta3)
        delta2 = E2 * self.sigmoid_derivative(O2)

        self.W34 += self.lr * np.dot(delta4, O3.T)
        self.W23 += self.lr * np.dot(delta3, O2.T)
        self.W12 += self.lr * np.dot(delta2, I.T)
