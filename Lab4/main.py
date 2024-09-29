import numpy as np

import NeuralNetwork as nn

I = np.array([[0.6], [0.2], [0.7], [0.4]])
T = np.array([[0.1], [0.9], [0.5]])
neunet = nn.NeuralNetwork(0.3)
print("Results before training: \t ", np.round(neunet.calculate(I), 5).tolist())
neunet.train(T, I)
print("Results after one training:  ", np.round(neunet.calculate(I), 5).tolist())
for i in range(100):
    neunet.train(T, I)
print("Results after 100 trainings: ", np.round(neunet.calculate(I), 2).tolist())
print("Ideal: \t\t\t\t\t\t ", T.tolist())
