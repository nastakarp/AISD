import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

# Функция heapify для создания кучи
def heapify(arr, n, i):
    largest = i  # Инициализируем наибольший элемент как корень
    left = 2 * i + 1  # Левый дочерний элемент
    right = 2 * i + 2  # Правый дочерний элемент

    # Если левый дочерний элемент больше корня
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Если правый дочерний элемент больше текущего наибольшего
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Если наибольший элемент не корень
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Меняем местами

        # Рекурсивно перестраиваем поддерево
        heapify(arr, n, largest)

# Основная функция пирамидальной сортировки
def heap_sort(arr):
    n = len(arr)

    # Построение макс-кучи
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Извлечение элементов из кучи по одному
    for i in range(n - 1, 0, -1):
        # Меняем текущий корень с последним элементом
        arr[i], arr[0] = arr[0], arr[i]

        # Восстанавливаем кучу для оставшегося подмножества элементов
        heapify(arr, i, 0)

# Функция для замера времени выполнения Heap Sort
def measure_time(arr):
    start_time = time.time()
    heap_sort(arr)
    end_time = time.time()
    return end_time - start_time

# Генерация массивов
def generate_arrays(n):
    arr_sorted = list(range(n))
    arr_almost_sorted = list(range(int(n * 0.9))) + [random.randint(0, n) for _ in range(int(n * 0.1))]
    arr_desc_sorted = list(range(n, 0, -1))
    arr_random = [random.randint(0, n) for _ in range(n)]
    return arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random

# Размеры массивов для тестирования
sizes = [i for i in range(1, 100001, 10000)]
# Замена ключей на русские слова
results = {'Размер': [], 'Отсортированный': [], 'Почти отсортированный': [], 'Убывающий': [], 'Случайный': []}

for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Размер'].append(n)
    results['Отсортированный'].append(measure_time(arr_sorted))
    results['Почти отсортированный'].append(measure_time(arr_almost_sorted))
    results['Убывающий'].append(measure_time(arr_desc_sorted))
    results['Случайный'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

# Преобразуем размеры массивов в n * log(n) для всех случаев
X_nlogn = np.array([n * math.log(n) if n > 0 else 0 for n in results['Размер']]).reshape(-1, 1)

# Модели линейной регрессии для каждого типа массива
model_sorted = LinearRegression()
model_almost_sorted = LinearRegression()
model_desc_sorted = LinearRegression()
model_random = LinearRegression()

# Обучение моделей на данных n * log(n)
model_sorted.fit(X_nlogn, np.array(results['Отсортированный']))
model_almost_sorted.fit(X_nlogn, np.array(results['Почти отсортированный']))
model_desc_sorted.fit(X_nlogn, np.array(results['Убывающий']))
model_random.fit(X_nlogn, np.array(results['Случайный']))

# Получаем предсказанные значения для регрессионных линий
y_pred_sorted = model_sorted.predict(X_nlogn)
y_pred_almost_sorted = model_almost_sorted.predict(X_nlogn)
y_pred_desc_sorted = model_desc_sorted.predict(X_nlogn)
y_pred_random = model_random.predict(X_nlogn)

# Построение графиков для каждого типа массива

# 1. График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный массив', linestyle='None')  # Точки для данных
plt.plot(results['Размер'], y_pred_sorted, label='Регрессия (Отсортированный)', linestyle='-')  # Линейная регрессия
plt.title('Производительность Heap Sort для отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный массив', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted, label='Регрессия (Почти отсортированный)', linestyle='-')
plt.title('Производительность Heap Sort для почти отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий массив', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted, label='Регрессия (Убывающий)', linestyle='-')
plt.title('Производительность Heap Sort для убывающего массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный массив', linestyle='None')
plt.plot(results['Размер'], y_pred_random, label='Регрессия (Случайный)', linestyle='-')
plt.title('Производительность Heap Sort для случайного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Общий график для всех случаев на одном графике
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный', linestyle='None')
plt.plot(results['Размер'], y_pred_sorted, label='Регрессия (Отсортированный)', linestyle='-')
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted, label='Регрессия (Почти отсортированный)', linestyle='-')
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted, label='Регрессия (Убывающий)', linestyle='-')
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный', linestyle='None')
plt.plot(results['Размер'], y_pred_random, label='Регрессия (Случайный)', linestyle='-')

plt.title('Производительность Heap Sort: Все случаи')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод уравнений регрессий
print("Уравнение регрессии для отсортированного массива: y =", model_sorted.coef_[0], "* n*log(n) +", model_sorted.intercept_)
print("Уравнение регрессии для почти отсортированного массива: y =", model_almost_sorted.coef_[0], "* n*log(n) +", model_almost_sorted.intercept_)
print("Уравнение регрессии для убывающего массива: y =", model_desc_sorted.coef_[0], "* n*log(n) +", model_desc_sorted.intercept_)
print("Уравнение регрессии для случайного массива: y =", model_random.coef_[0], "* n*log(n) +", model_random.intercept_)
