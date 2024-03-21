import random
import numpy as np


class Point:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2


class Neuron:
    def step_function(self, x):
        return max(0, x)


class Perceptron(Neuron):
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0

    def train(self, X_train, y_train, num_epochs=1000):
        for _ in range(num_epochs):
            for i in range(len(X_train)):
                prediction = self.step_function(np.dot(X_train[i], self.weights) + self.bias)
                if prediction != y_train[i]:
                    self.weights += y_train[i] * X_train[i]
                    self.bias += y_train[i]

    def test(self, X_test, y_test):
        correct = 0
        for i in range(len(X_test)):
            prediction = self.step_function(np.dot(X_test[i], self.weights) + self.bias)
            if prediction == y_test[i]:
                correct += 1
        return correct / len(X_test)


class Adaline(Neuron):
    def __init__(self, num_features, learning_rate=0.1):
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.learning_rate = learning_rate

    def train(self, X_train, y_train, num_epochs=1000):
        for _ in range(num_epochs):
            for i in range(len(X_train)):
                prediction = np.dot(X_train[i], self.weights) + self.bias
                error = y_train[i] - prediction
                self.weights += self.learning_rate * error * X_train[i]
                self.bias += self.learning_rate * error

    def test(self, X_test, y_test):
        correct = 0
        for i in range(len(X_test)):
            prediction = self.step_function(np.dot(X_test[i], self.weights) + self.bias)
            if prediction == y_test[i]:
                correct += 1
        return correct / len(X_test)


class NeuralNetwork:
    def __init__(self, num_features):
        self.num_features = num_features
        self.perceptron = Perceptron(num_features)
        self.adaline = Adaline(num_features)

    def train(self, X_train, y_train):
        self.perceptron.train(X_train, y_train)
        self.adaline.train(X_train, y_train)

    def test(self, X_test, y_test):
        predictions_perceptron = []
        predictions_adaline = []
        for i in range(len(X_test)):
            prediction_perceptron = self.perceptron.step_function(
                np.dot(X_test[i], self.perceptron.weights) + self.perceptron.bias)
            prediction_adaline = self.adaline.step_function(np.dot(X_test[i], self.adaline.weights) + self.adaline.bias)
            predictions_perceptron.append(prediction_perceptron)
            predictions_adaline.append(prediction_adaline)
            print(
                f"Point {i + 1}: Coordinates: ({X_test[i][0]}, {X_test[i][1]}), Perceptron Prediction: {prediction_perceptron}, Adaline Prediction: {prediction_adaline}, True Label: {y_test[i]}")
        accuracy_perceptron = sum(1 for i in range(len(X_test)) if predictions_perceptron[i] == y_test[i]) / len(X_test)
        accuracy_adaline = sum(1 for i in range(len(X_test)) if predictions_adaline[i] == y_test[i]) / len(X_test)
        return accuracy_perceptron, accuracy_adaline


# Генерация тренировочных и тестовых данных
def generate_data(num_points):
    X_data = []
    y_data = []
    for _ in range(num_points):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        X_data.append([x1, x2])
        y_data.append(1 if x1 > x2 else -1)
    return np.array(X_data), np.array(y_data)


# Создание объектов классов и обучение модели
X_train, y_train = generate_data(20)
X_test, y_test = generate_data(1000)
neural_network = NeuralNetwork(num_features=2)
neural_network.train(X_train, y_train)

# Тестирование и вывод результатов
accuracy_perceptron, accuracy_adaline = neural_network.test(X_test, y_test)

print("Perceptron Results:")
print(f"Perceptron Accuracy on Test Data: {accuracy_perceptron}")
print(f"Adaline Accuracy on Test Data: {accuracy_adaline}")
