import numpy as np

# Функция активации ReLU
def ReLU(x):
    return np.maximum(0, x)

# Сигмоидная функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Веса и порог для классификации XOR с двухслойной нейронной сетью
w1 = np.array([[1, -1], [-1, 1]])  # Веса для первого слоя
w2 = np.array([1, 1])  # Веса для второго слоя
b1 = np.array([0, 0])  # Порог для первого слоя
b2 = 0  # Порог для второго слоя

# Функция классификации точек с использованием двухслойной нейронной сети
def classify_point(x1, x2):
    hidden_layer = ReLU(np.dot(np.array([x1, x2]), w1) + b1)
    output = sigmoid(np.dot(hidden_layer, w2) + b2)
    return int(output > 0.5)

# Функция XOR
def XOR(x1, x2):
    return int(x1 != x2)

# Входные данные XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Классификация точек
for i, point in enumerate(X):
    classification = classify_point(point[0], point[1])
    xor_result = XOR(point[0], point[1])
    print(f"Point: {point}, Classification: {classification}, XOR Result: {xor_result}")