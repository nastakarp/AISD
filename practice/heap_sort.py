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
sizes = [i for i in range(1000, 100001, 10000)]
results = {'Size': [], 'Sorted': [], 'Almost Sorted': [], 'Descending': [], 'Random': []}

# Измерение времени для каждого типа массива и их размеров
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Size'].append(n)
    results['Sorted'].append(measure_time(arr_sorted))
    results['Almost Sorted'].append(measure_time(arr_almost_sorted))
    results['Descending'].append(measure_time(arr_desc_sorted))
    results['Random'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

# Преобразуем размеры массивов в n * log(n) для всех случаев
X_nlogn = np.array([n * math.log(n) for n in results['Size']]).reshape(-1, 1)

# Модели линейной регрессии для каждого типа массива
model_sorted = LinearRegression()
model_almost_sorted = LinearRegression()
model_desc_sorted = LinearRegression()
model_random = LinearRegression()

# Обучение моделей на данных n * log(n)
model_sorted.fit(X_nlogn, np.array(results['Sorted']))
model_almost_sorted.fit(X_nlogn, np.array(results['Almost Sorted']))
model_desc_sorted.fit(X_nlogn, np.array(results['Descending']))
model_random.fit(X_nlogn, np.array(results['Random']))

# Получаем предсказанные значения для регрессионных линий
y_pred_sorted = model_sorted.predict(X_nlogn)
y_pred_almost_sorted = model_almost_sorted.predict(X_nlogn)
y_pred_desc_sorted = model_desc_sorted.predict(X_nlogn)
y_pred_random = model_random.predict(X_nlogn)

# Построение графиков для каждого типа массива

# 1. График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], 'o', label='Sorted Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_sorted, label='Linear Regression (Sorted)', linestyle='-')  # Линейная регрессия
plt.title('Heap Sort Performance on Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Almost Sorted'], 'o', label='Almost Sorted Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_almost_sorted, label='Linear Regression (Almost Sorted)', linestyle='-')  # Линейная регрессия
plt.title('Heap Sort Performance on Almost Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Descending'], 'o', label='Descending Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_desc_sorted, label='Linear Regression (Descending)', linestyle='-')  # Линейная регрессия
plt.title('Heap Sort Performance on Descending Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Random'], 'o', label='Random Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_random, label='Linear Regression (Random)', linestyle='-')  # Линейная регрессия
plt.title('Heap Sort Performance on Random Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# Общий график для всех случаев на одном графике
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], 'o', label='Sorted', linestyle='None')
plt.plot(results['Size'], y_pred_sorted, label='Regression (Sorted)', linestyle='-')
plt.plot(results['Size'], results['Almost Sorted'], 'o', label='Almost Sorted', linestyle='None')
plt.plot(results['Size'], y_pred_almost_sorted, label='Regression (Almost Sorted)', linestyle='-')
plt.plot(results['Size'], results['Descending'], 'o', label='Descending', linestyle='None')
plt.plot(results['Size'], y_pred_desc_sorted, label='Regression (Descending)', linestyle='-')
plt.plot(results['Size'], results['Random'], 'o', label='Random', linestyle='None')
plt.plot(results['Size'], y_pred_random, label='Regression (Random)', linestyle='-')

plt.title('Heap Sort Performance: All Cases')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()
