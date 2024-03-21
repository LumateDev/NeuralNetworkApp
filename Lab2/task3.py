import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(0)
        self.weights = np.random.uniform(-1, 1, (4, 9))

    def relu(self, x):
        return np.maximum(0, x)

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def forward(self, x):
        return self.relu(np.dot(self.weights, x.T))

    def backward(self, x, y_true, y_pred, lr):
        error = y_pred - y_true
        gradient = np.dot(error.reshape(-1, 1), x.reshape(1, -1)) / len(x)
        self.weights -= lr * gradient

def test(network, x):
    for i in range(len(x)):
        neurons_out = network.forward(x[i])
        answers = [1 if neuron_out >= 0.5 else 0 for neuron_out in neurons_out]
        print(f"Для {i + 1}-ой буквы ответ: {answers}")

neural_network = NeuralNetwork()

lr = 0.3
epochs = 1000
x_train = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 1, 0, 0, 1, 1, 1]])
y_train = np.array([[0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0]])

# Обучение
for epoch in range(epochs):
    for i in range(len(x_train)):
        neurons_out = neural_network.forward(x_train[i])
        neural_network.backward(x_train[i], y_train[i], neurons_out, lr)
    if epoch % 50 == 0:
        error = neural_network.mse_loss(y_train, neural_network.forward(x_train))
        print(f"Эпоха обучения: {epoch}/{epochs}, ошибка: {error}")

# Проверка на обычных буквах
print("\n--------------------Проверка на обычных буквах--------------------")
test(neural_network, x_train)

# Проверка на "шумных" буквах
print("\n--------------------Проверка на <шумных> буквах--------------------")
x_test_noisy = np.array([[1, 1, 1, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 1, 0, 0, 1, 1, 0, 1, 0],
                         [1, 0, 0, 1, 0, 0, 1, 1, 0]])
test(neural_network, x_test_noisy)