import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    center = 0

    def __init__(self, center):
        self.center = center

    def get_center(self):
        return self.center


class NeuralNetwork:

    def gauss_fun(self, x, center, radius):

        # Функция Гаусса  многомерный случай
        # Args: x - входное значение; center - центр соответствующего нейрона; radius - радиус

        norm_of_vector = np.linalg.norm(x - center)
        a = 1 / (2 * np.power(radius, 2))
        return np.exp(-a * norm_of_vector)

    def calculate_characteristics_matrix(self, neurons_quantity, entrances_x, neurons, radius):

        #Функция, которая считает характеристическую матрицу значений радиально-симметричных элементов
        #Args: neurons_quantity - кол-во нейронов; entrances_x - входные значения; neurons - список нейронов; radius - радиус
        #Return: характеристическая матрица

        h = np.zeros((len(entrances_x), neurons_quantity))
        for i in range(len(entrances_x)):
            for j in range(neurons_quantity):
                h[i][j] = self.gauss_fun(entrances_x[i][0], neurons[j].get_center(), radius)
        return h

    def calculate_matrix_weight_coefficients(self, matrix_h, matrix_y):

        #Функция, которая считает матрицу весовых коэффициентов
        #Args: matrix_h - характеристическая матрица; matrix_y - список правильных ответов
        #Return: матрица весовых коэффициентов


        return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(matrix_h), matrix_h)),
                                   np.transpose(matrix_h)), matrix_y)

    def calculate_total_outputs(self, neurons, matrix_w, entrances_x, neurons_quantity, radius):

        #Функция, которая считает итоговые выходы на основе матрицы весовых коэффициентов
        #Args: neurons - список нейронов; matrix_w - матрица весовых коэффициентов; entrances_x - входные значения; neurons_quantity - кол-во нейронов; radius - радиус
        #Return: список значений

        total_list = np.matmul(self.calculate_characteristics_matrix(neurons_quantity, entrances_x, neurons, radius),
                               matrix_w)
        return total_list


neurons_quantity = 5
entrances_x = [[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0]]
outputs_y = [[-0.48], [-0.78], [-0.83], [-0.67], [-0.2], [0.7], [1.48], [1.17], [0.2]]
# Взяли центры как 1, 3, 5, 7, 9 входной параметр
centers = [-2.0, -1.0, 0.0, 1.0, 2.0]
# Сказано, что r можно просто взять = 1.5 для всех нейронов
radius = 1.5

neurons = [Neuron(centers[i]) for i in range(len(centers))]
neural_network = NeuralNetwork()

print("Характеристическая матрица:")
h = neural_network.calculate_characteristics_matrix(neurons_quantity, entrances_x, neurons, radius)
print(h)

print("Матрица (вектор) весовых коэффициентов:")
w = neural_network.calculate_matrix_weight_coefficients(h, outputs_y)
print(w)

print("Вектор полученных значений:")
total_outputs = neural_network.calculate_total_outputs(neurons, w, entrances_x, neurons_quantity, radius)
print(total_outputs)

n = len(entrances_x)
s = 0
for i in range(n):
    s += np.abs(1 - (outputs_y[i][0] / total_outputs[i]))
print(f"Средняя относительная ошибка аппроксимации: {s / n * 100}%")

plt.scatter(entrances_x, outputs_y, c="blue", label="Исходные точки")
plt.plot(entrances_x, total_outputs, label="Полученная аппроксимирующая зависимость")
plt.legend()
plt.grid(True)
plt.show()
