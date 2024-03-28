
#Ну всё, сейчас мы так запрограммируем эту нейросеть, что всем завидно будет
#Главная задача - не использовать numpy и прочую халяву, сами сделаем

import math as mt
import matplotlib.pyplot as plt

#Аналог функции transpose из numpy
def matrix_transposition(matrix):
    """
    Функция, которая транспонирует матрицу
    Args: matrix - исходная матрица
    Return: транспонированная матрица
    """
    row_len = len(matrix)
    column_len = len(matrix[0])
    result_matrix = [[0 for i in range(row_len)] for j in range(column_len)]
    for i in range(row_len):
        for j in range(column_len):
            result_matrix[j][i] = matrix[i][j]
    return result_matrix

#Аналог функции dot из numpy
def matrix_multiplication(matrix_1, matrix_2):
    """
    Функция, которая умножает 2 матрицы
    Args: matrix_1 - первая матрица; matrix_2 - вторая матрица
    Return: результирующая матрица
    """
    result_matrix = [[0 for i in range(len(matrix_2[0]))] for j in range(len(matrix_1))]
    for i in range(len(matrix_1)):
        for j in range(len(matrix_2[0])):
            for k in range(len(matrix_2)):
                result_matrix[i][j] += matrix_1[i][k] * matrix_2[k][j]
    return result_matrix

#Аналог функции inv из numpy.linalg
#Самая хитрая функция из всех, пусть будет с комментариями
def rising_matrix_to_minusOne(matrix):
    """
    Функция, которая возводит матрицу в степень -1 (находит обратную матрицу методом Гаусса-Жордана)
    Args: matrix - матрица
    Return: матрица, возведённая в степень -1 (обратная)
    """
    n = len(matrix)
    augmented_matrix = [row + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]

    #Приведение к ступенчатому виду
    for i in range(n):
        #Нормализация текущей строки
        pivot = augmented_matrix[i][i]
        for j in range(n * 2):
            augmented_matrix[i][j] /= pivot

        #Обнуление остальных строк
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(n * 2):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    #Извлечение обратной матрицы из расширенной
    inverse = [row[n:] for row in augmented_matrix]

    return inverse

class Neuron:

    center = 0

    def __init__(self, center):
        """
        Функция, которая инициализирует нейрон с центром
        Args: center - центр
        Return: отсутствует
        """
        self.center = center

    def get_center(self):
        """
        Функция-гетер центра нейрона
        Args: отсутствуют
        Return: центр нейрона
        """
        return self.center

class Neural_Network:

    def gauss_fun(self, x, center, radius):
        """
        Функция Гаусса (как бы многомерный случай, но не совсем)
        Args: x - входное значение; center - центр соответствующего нейрона; radius - радиус
        Return: значение функции
        """
        #В многомерном случае в norm_of_vector корень из суммы квадратов по разным центрам
        norm_of_vector = mt.pow(x - center, 2)
        a = 1 / (2 * mt.pow(radius, 2))
        return mt.exp(-a * norm_of_vector)

    def calculate_characteristics_matrix(self, neurons_quantity, entrances_x, neurons, radius):
        """
        Функция, которая считает характеристическую матрицу значений радиально-симметричных элементов
        Args: neurons_quantity - кол-во нейронов; entrances_x - входные значения; neurons - список нейронов; radius - радиус
        Return: характеристическая матрица
        """
        h = [[0 for j in range(neurons_quantity)] for i in range(len(entrances_x))]
        for i in range(len(entrances_x)):
            for j in range(neurons_quantity):
                h[i][j] += self.gauss_fun(entrances_x[i][0], neurons[j].get_center(), radius)
        return h

    def calculate_matrix_weight_coefficients(self, matrix_h, matrix_y):
        """
        Функция, которая считает матрицу весовых коэффициентов
        Args: matrix_h - характеристическая матрица; matrix_y - список правильных ответов
        Return: матрица весовых коэффициентов
        """
        #А это наглядный минус длинных названий функций)
        #return matrix_multiplication(matrix_multiplication(rising_matrix_to_minusOne(matrix_multiplication(matrix_transposition(matrix_h), matrix_h)), matrix_transposition(matrix_h)), matrix_y)
        h_T_h = matrix_multiplication(matrix_transposition(matrix_h), matrix_h)
        h_minusOne = rising_matrix_to_minusOne(h_T_h)
        h_T_y = matrix_multiplication(matrix_transposition(matrix_h), matrix_y)
        result = matrix_multiplication(h_minusOne, h_T_y)
        return result

    def calculate_total_outputs(self, neurons, matrix_w, entrances_x, neurons_quantity, radius):
        """
        Функция, которая считает итоговые выходы на основе матрицы весовых коэффициентов
        Args: neurons - список нейронов; matrix_w - матрица весовых коэффициентов; entrances_x - входные значения; neurons_quantity - кол-во нейронов; radius - радиус
        Return: список значений
        """
        total_list = []
        for i in range(len(entrances_x)):
            tmp = 0
            for j in range(neurons_quantity):
                tmp += self.gauss_fun(entrances_x[i][0], neurons[j].get_center(), radius) * matrix_w[j][0]
            total_list.append(tmp)
        return total_list
        #return [sum([self.gauss_fun(entrances_x[j], neurons[i].get_center(), radius) * matrix_w[i][0] for i in range(neurons_quantity)] for j in range(len(entrances_x)))]

neurons_quantity = 5
entrances_x = [[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0]]
outputs_y = [[-0.48], [-0.78], [-0.83], [-0.67], [-0.2], [0.7], [1.48], [1.17], [0.2]]
#Взяли центры как 1, 3, 5, 7, 9 входной параметр
centers = [-2.0, -1.0, 0.0, 1.0, 2.0]
#Сказано, что r можно просто взять = 1.5 для всех нейронов
radius = 1.5

neurons = [Neuron(centers[i]) for i in range(len(centers))]
neural_network = Neural_Network()

print()
print("Характеристическая матрица:")
h = neural_network.calculate_characteristics_matrix(neurons_quantity, entrances_x, neurons, radius)
for i in range(len(h)):
    s = ""
    for j in range(len(h[0])):
        s += str(h[i][j]) + " "
    print(s)

print()
print("Матрица (вектор) весовых коэффициентов:")
w = neural_network.calculate_matrix_weight_coefficients(h, outputs_y)
for i in range(len(w)):
    s = ""
    for j in range(len(w[0])):
        s += str(w[i][j]) + " "
    print(s)

print()
print("Вектор полученных значений:")
total_outputs = neural_network.calculate_total_outputs(neurons, w, entrances_x, neurons_quantity, radius)
for i in range(len(total_outputs)):
    print(total_outputs[i])

n = len(entrances_x)
s = 0
for i in range(n):
    s += mt.fabs(1 - (outputs_y[i][0] / total_outputs[i]))
print(f"Средняя относительная ошибка аппроксимации: {s / n * 100}%")

plt.scatter(entrances_x, outputs_y, c="blue", label="Исходные точки")
plt.plot(entrances_x, total_outputs, label="Полученная аппроксимирующая зависимость")
plt.legend()
plt.grid(True)
plt.show()
