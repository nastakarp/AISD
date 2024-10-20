import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

# Реализация Shell Sort с последовательностью Хиббарда
def shell_sort_hibbard(arr):
    n = len(arr)
    # Формируем последовательность Хиббарда
    gap = 1
    gaps = []
    while gap < n:
        gaps.append(gap)
        gap = 2 * gap + 1  # Последовательность Хиббарда

    # Сортируем с шагами по Хиббарду (в обратном порядке)
    for gap in reversed(gaps):
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp

    return arr

# Функция замера времени выполнения сортировки
def measure_time(arr):
    start_time = time.time()
    shell_sort_hibbard(arr)
    end_time = time.time()
    return end_time - start_time

# Генерация тестовых массивов
def generate_arrays(n):
    arr_sorted = list(range(n))
    arr_almost_sorted = list(range(int(n * 0.9))) + [random.randint(0, n) for _ in range(int(n * 0.1))]
    arr_desc_sorted = list(range(n, 0, -1))
    arr_random = [random.randint(0, n) for _ in range(n)]
    return arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random

# Размеры массивов для тестирования
sizes = [i for i in range(1000, 100001, 10000)]
results = {'Size': [], 'Sorted': [], 'Almost Sorted': [], 'Descending': [], 'Random': []}

# Измерение времени выполнения для каждого типа массива
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Size'].append(n)
    results['Sorted'].append(measure_time(arr_sorted))
    results['Almost Sorted'].append(measure_time(arr_almost_sorted))
    results['Descending'].append(measure_time(arr_desc_sorted))
    results['Random'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

# Преобразование размеров массивов для каждой регрессии
X = np.array(results['Size']).reshape(-1, 1)

# 1. Линейная регрессия для отсортированного массива (лучший случай, O(n))
model_sorted = LinearRegression()
model_sorted.fit(X, np.array(results['Sorted']))

# 2. Логарифмическая регрессия для почти отсортированного массива (средний случай O(n log n))
X_nlogn = np.array([n * math.log(n) for n in results['Size']]).reshape(-1, 1)
model_almost_sorted = LinearRegression()
model_random = LinearRegression()
model_almost_sorted.fit(X_nlogn, np.array(results['Almost Sorted']))
model_random.fit(X_nlogn, np.array(results['Random']))

# 3. Полиномиальная регрессия для убывающего массива (худший случай O(n^(3/2)))
X_n32 = np.array([n ** 1.5 for n in results['Size']]).reshape(-1, 1)
model_desc_sorted = LinearRegression()
model_desc_sorted.fit(X_n32, np.array(results['Descending']))

# Предсказания для каждого случая
y_pred_sorted = model_sorted.predict(X)
y_pred_almost_sorted = model_almost_sorted.predict(X_nlogn)
y_pred_desc_sorted = model_desc_sorted.predict(X_n32)
y_pred_random = model_random.predict(X_nlogn)

# Построение графиков

# 1. График для отсортированного массива (лучший случай O(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], 'o', label='Sorted Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_sorted, label='Linear Regression (Sorted)', linestyle='-')  # Линейная регрессия
plt.title('Shell Sort Hibbard Performance on Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива (средний случай O(n log n))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Almost Sorted'], 'o', label='Almost Sorted Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_almost_sorted, label='Logarithmic Regression (Almost Sorted)', linestyle='-')  # Логарифмическая регрессия
plt.title('Shell Sort Hibbard Performance on Almost Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для убывающего массива (худший случай O(n^(3/2)))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Descending'], 'o', label='Descending Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_desc_sorted, label='Polynomial Regression (Descending)', linestyle='-')  # Полиномиальная регрессия
plt.title('Shell Sort Hibbard Performance on Descending Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для случайного массива (средний случай O(n log n))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Random'], 'o', label='Random Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_random, label='Logarithmic Regression (Random)', linestyle='-')  # Логарифмическая регрессия
plt.title('Shell Sort Hibbard Performance on Random Array')
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

plt.title('Shell Sort Hibbard Performance: All Cases')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()
