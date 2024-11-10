import NeuralNetwork as net
import numpy as np


def train_fcn(network):
    training_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_list = training_file.readlines()
    training_file.close()
    epochs = 5
    print("Neural network training...")
    for e in range(epochs):
        print("Epoch", e + 1)
        # перебираємо всі записи в тренувальному наборі даних
        for record in training_list:
            # отримуємо список значень,
            # використовуючи символ, в якості роздільника
            all_values = record.split(',')
            # формуємо тренувальні вхідні сигнали в діапазоні від 0.01 до 1.0
            inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] маркерне значення
            targets[int(all_values[0])] = 0.99
            # навчаємо нейронну мережу
            network.train(inputs, targets)


def test_fcn(network):
    test_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_list = test_file.readlines()
    test_file.close()
    # список для зберігання результатів
    # розпізнавання рукописних цифр, спочатку порожній
    scorecard = []
    print("Neural network testing...")
    # перебираємо всі записи в тестовому наборі даних
    for record in test_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        # опитування нейронної мережі
        outputs = network.query(inputs)
        # отримуємо індекс найбільшого значення вихідного сигналу,
        # який є маркерним значенням
        label = np.argmax(outputs)
        # якщо відповідь нейронної мережі збігається з маркерним значенням
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
        # розраховуємо показник ефективності у вигляді частки правильних відповідей
    scorecard_array = np.asarray(scorecard)
    print("Efficiency:", scorecard_array.sum() / scorecard_array.size * 100, "%")


# кількість вузлів у вхідному шарі
input_nodes = 784
# кількість вузлів у прихованому шарі
hidden_nodes = 200
# кількість вузлів у вихідному шарі
output_nodes = 10
# коефіцієнт швидкості навчання
learning_rate = 0.1
# створюємо екземпляр нейронної мережі
n = net.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# навчаємо нейронну мережу
train_fcn(n)
# тестуємо нейронну мережу
test_fcn(n)
# зберігаємо вагові коефіцієнти зв'язків у файли
n.save_weights()
