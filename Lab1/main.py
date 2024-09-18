import LinearClassifier as lc

# Тренувальні дані
data = [
    (2, 12),
    (9, 2),
    (3, 10),
    (10, 3),
    (4, 11),
    (11, 4),
    (6, 12),
    (12, 5),
    (3, 8),
    (11, 2),
    (4, 9),
    (12, 2),
    (5, 10),
    (12, 3)
]
weight = 0.05
learning_rate = 0.01
# Створення класифікатора
classifier = lc.LinearClassifier(weight, learning_rate)

# Класифікація до навчання
print("Класифікація жуків до навчання:")
for i, (width, length) in enumerate(data):
    print(f"Приклад {i + 1}: {classifier.query(length / width)} ")

# Навчання класифікатора
for i in range(1000):
    for input in data:
        classifier.train(input[0], input[1])

# Класифікація після навчання
print("\nКласифікація жуків після навчання:")
for i, (width, length) in enumerate(data):
    print(f"Приклад {i + 1}: {classifier.query(length / width)} ")
