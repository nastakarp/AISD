import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Функция сортировки вставками
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Функция для замера времени выполнения сортировки
def measure_time(arr):
    start_time = time.time()
    insertion_sort(arr)
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
sizes = [i for i in range(1000, 10001, 1000)]
results = {'Size': [], 'Sorted': [], 'Almost Sorted': [], 'Descending': [], 'Random': []}

# Замер времени выполнения
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Size'].append(n)
    results['Sorted'].append(measure_time(arr_sorted))
    results['Almost Sorted'].append(measure_time(arr_almost_sorted))
    results['Descending'].append(measure_time(arr_desc_sorted))
    results['Random'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

# Преобразуем размеры массивов в массив X для регрессионного анализа
X = np.array(results['Size']).reshape(-1, 1)

# 1. Линейная регрессия для отсортированного массива (лучший случай)
model_sorted = LinearRegression()
model_sorted.fit(X, np.array(results['Sorted']))

# 2. Полиномиальные признаки для остальных массивов (квадратичная зависимость)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model_almost_sorted = LinearRegression()
model_desc_sorted = LinearRegression()
model_random = LinearRegression()

model_almost_sorted.fit(X_poly, np.array(results['Almost Sorted']))
model_desc_sorted.fit(X_poly, np.array(results['Descending']))
model_random.fit(X_poly, np.array(results['Random']))

# Предсказания для каждого массива
y_pred_sorted = model_sorted.predict(X)  # Линейная регрессия
y_pred_almost_sorted = model_almost_sorted.predict(X_poly)  # Квадратичная регрессия
y_pred_desc_sorted = model_desc_sorted.predict(X_poly)  # Квадратичная регрессия
y_pred_random = model_random.predict(X_poly)  # Квадратичная регрессия

# Построение графиков

# 1. График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], 'o', label='Sorted Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_sorted, label='Linear Regression (Sorted)', linestyle='-')  # Линейная регрессия
plt.title('Insertion Sort Performance on Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))
plt.show()

# 2. График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Almost Sorted'], 'o', label='Almost Sorted Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_almost_sorted, label='Quadratic Regression (Almost Sorted)', linestyle='-')  # Квадратичная регрессия
plt.title('Insertion Sort Performance on Almost Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))
plt.show()

# 3. График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Descending'], 'o', label='Descending Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_desc_sorted, label='Quadratic Regression (Descending)', linestyle='-')  # Квадратичная регрессия
plt.title('Insertion Sort Performance on Descending Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))
plt.show()

# 4. График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Random'], 'o', label='Random Array', linestyle='None')  # Точки для данных
plt.plot(results['Size'], y_pred_random, label='Quadratic Regression (Random)', linestyle='-')  # Квадратичная регрессия
plt.title('Insertion Sort Performance on Random Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))
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

plt.title('Insertion Sort Performance: All Cases')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()