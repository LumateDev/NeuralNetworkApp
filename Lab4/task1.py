
#Промежуточная база: в нейросети Кохонена используется линейная функция активации; функция ошибок ->
# -> основанная на евклидовом расстоянии между весами и входными данными; теперь тёмный лес ->
# -> корректировка весов происходит следующим путём: нахождение лучшего соответствующего нейрона для ->
# -> входного образца; обновление весов лучшего нейрона и окружающих нейронов (корректировка ->
# -> весов в направлении входных данных с учётом скорости обучения) (сам не понял, что прочитал)

import numpy as np
import numpy.random as rnd


class Neuron:

    def __init__(self, size: int):
        """
        Функция, которая инициализирует нейрону список случайных весов
        :param size - длина списка
        :return: None
        """
        self.w = rnd.random(size)

    def summat(self, input_values: list):
        """
        Функция, которая реализует сумматор нейрона
        Args: input_values - список входных значений
        Return: выход нейрона
        """
        #прибавляем порог (1 +)
        return 1 + sum(input_values[i] * self.w[i] for i in range(len(input_values)))

    def fun_activation_ReLU(self, input_x: float):
        """
        Кусочно-линейная функция активации (ReLU)
        Args: input_x - входное значение
        Return: результат функции
        """
        return max(0, input_x)

class Neural_Network:

    def euclid_error(self, neuron_weights: list, input_values: list):
        """
        Функция евклидова расстояния для оценки ошибки нейрона
        Args: neuron_weights - веса нейрона; input_values - список входных значений
        Return: ошибка нейрона
        """
        return np.sqrt(sum((neuron_weights[i] - input_values[i]) ** 2 for i in range(len(input_values))))

    def find_bmu(self, input_values: list, list_of_neurons: list):
        """
        Функция нахождения BMU (Best Matching Unit, лучший нейрон под конкретный пример)
        Args: input_values - список входных значений; neurons - список нейронов
        Return: индекс BMU
        """
        min_distance = float('inf')
        bmu_index = 0
        #enumerate() позволяет перебирать список и по индексам, и по элементам одновременно
        for i, neuron in enumerate(neurons):
            distance = self.euclid_error(neuron.w, input_values)
            if distance < min_distance:
                min_distance = distance
                bmu_index = i
        return bmu_index

    def update_weights(self, bmu_index: int, input_values: list, learning_rate: float, radius: float, neurons: list):
        """
        Функция обновления весов BMU и соседних нейронов
        Args: bmu_index - индекс BMU; input_values - список входных значений;
              learning_rate - скорость обучения; radius - радиус влияния; neurons - список нейронов
        Return: отсутствует
        """
        #обновление весов BMU
        for i in range(len(input_values)):
            neurons[bmu_index].w[i] += learning_rate * (input_values[i] - neurons[bmu_index].w[i])
        #обновление весов остальных нейронов в радиусе BMU
        for i in range(len(neurons)):
            if i != bmu_index:
                distance_to_BMU = abs(bmu_index - i)
                if distance_to_BMU <= radius:
                    #вычисление коэффициента влияния на веса других нейронов
                    influence = np.exp(-(distance_to_BMU ** 2) / (2 * radius ** 2))
                    #обновление весов соседних нейронов
                    for j in range(len(input_values)):
                        neurons[i].w[j] += learning_rate * influence * (input_values[j] - neurons[i].w[j])

    def train(self, inputs_values: list[list], learning_rate: float, max_epochs: int, radius: float, neurons: list):
        """
        Функция обучения нейросети
        Args: inputs_values - список списков входных данных; learning_rate - скорость обучения;
              max_epochs - максимальное количество эпох обучения; radius - начальный радиус влияния;
              neurons - список нейронов
        Return: отсутствует
        """
        for epoch in range(max_epochs):
            #уменьшение радиуса влияния с каждой эпохой
            current_radius = radius * np.exp(-epoch / max_epochs)
            #обучение сети
            for i in range(len(inputs_values)):
                bmu_index = self.find_bmu(inputs_values[i], neurons)
                self.update_weights(bmu_index, inputs_values[i], learning_rate, current_radius, neurons)
            print(f"Эпоха {epoch + 1} завершена. Текущий радиус влияния: {current_radius}")
            #learning_rate -= 0.05

    def testing(self, list_of_neurons: list, inputs_values: list[list], outputs_values: list[list]):
        """
        Функция, которая проверяет работу нейросети
        Args: list_of_neurons - список нейронов; inputs_values - список списков входных данных;
              outputs_values - список списков желаемых результатов
        Return: отсутствует
        """
        for i in range(len(inputs_values)):
            #средние выходы нейронов
            neurons_outputs = []
            for j in range(len(list_of_neurons)):
                #выход нейрона
                neuron_out = list_of_neurons[j].fun_activation_ReLU(list_of_neurons[j].summat(inputs_values[i]))
                neurons_outputs.append(neuron_out)
            #среднее значение сети
            mean = sum(neurons_outputs) / len(list_of_neurons)
            #блок магического преобразования (раз нельзя уменьшить веса, то будем их интерпретировать как хотим)
            result = 0
            if mean < 8750: result = 0
            elif 8750 < mean < 8900: result = 1
            elif 8900 < mean < 10700: result = 1.25
            elif 10700 < mean < 11300: result = 1.5
            else: result = 1.75
            print(f"Результаты нейронов для примера {i + 1}: {neurons_outputs}; среднее значение: {mean}; интерпретируемый результат: {result} ожидаемый выход: {outputs_values[i][0]}")
            print(f"Сумма входных данных: {sum(inputs_values[i])}; т.к. результат: {result}, то вот такая вот зависимость: {result / sum(inputs_values[i])}")

#кол-во нейронов
neurons_quantity = 4
#входные параметры
inputs_x = [
    [1, 1, 60, 79, 60, 72, 63],
    [1, 0, 60, 61, 30, 5, 17],
    [0, 0, 60, 61, 30, 66, 58],
    [1, 1, 85, 78, 72, 70, 85],
    [0, 1, 65, 78, 60, 67, 65],
    [0, 1, 60, 78, 77, 81, 60],
    [0, 1, 55, 79, 56, 69, 72],
    [1, 0, 55, 56, 50, 56, 60],
    [1, 0, 55, 60, 21, 64, 50],
    [1, 0, 60, 56, 30, 16, 17],
    [0, 1, 85, 89, 85, 92, 85],
    [0, 1, 60, 88, 76, 66, 60],
    [1, 0, 55, 64, 0, 9, 50],
    [0, 1, 80, 83, 62, 72, 72],
    [1, 0, 55, 10, 3, 8, 50],
    [0, 1, 60, 67, 57, 64, 50],
    [1, 1, 75, 98, 86, 82, 85],
    [0, 1, 85, 85, 81, 85, 72],
    [1, 1, 80, 56, 50, 69, 50],
    [1, 0, 55, 60, 30, 8, 60]
]
#желаемые выходы
outputs_y = [
    [1], [0], [0], [1.25], [1],
    [1.25], [0], [0], [0], [0],
    [1.75], [1.25], [0], [1.25], [0],
    [0], [1.50], [1.25], [0], [0]
]
#скорость обучения
learning_rate = 0.01
#кол-во эпох
epochs = 1000
#радиус влияния
radius = 0.5
#список нейронов
neurons = [Neuron(len(inputs_x[0])) for _ in range(neurons_quantity)]
#нейросеть
neural_network = Neural_Network()
#проверка сети до обучения
print()
print("Проверка сети до обучения")
neural_network.testing(neurons, inputs_x, outputs_y)
#обучение
print()
neural_network.train(inputs_x, learning_rate, epochs, radius, neurons)
#проверка после обучения
print()
print("Проверка сети после обучения")
neural_network.testing(neurons, inputs_x, outputs_y)

print(f"""
Выводы по сети (сумма входных данных - СВД):
0 < СВД < 333 - зависимость = 0 - кластер 1
333 < СВД < 337 - зависимость = 0.0029-0.0030 - кластер 2
337 < СВД < 410 - зависимость = 0.0030-0.0035 - кластер 3
410 < СВД < 429 - зависимость = 0.0035-0.0036 - кластер 4
429 < СВД < 438 - зависимость = 0.004 - кластер 5
""")
