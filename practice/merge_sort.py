import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

# Реализация Merge Sort
def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid

    L = [0] * n1
    R = [0] * n2

    for i in range(0, n1):
        L[i] = arr[left + i]
    for j in range(0, n2):
        R[j] = arr[mid + 1 + j]

    i = 0
    j = 0
    k = left

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2

        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)

        merge(arr, left, mid, right)

def measure_time(arr):
    start_time = time.time()
    merge_sort(arr, 0, len(arr) - 1)
    end_time = time.time()
    return end_time - start_time

def generate_arrays(n):
    arr_sorted = list(range(n))
    arr_almost_sorted = list(range(int(n * 0.9))) + [random.randint(0, n) for _ in range(int(n * 0.1))]
    arr_desc_sorted = list(range(n, 0, -1))
    arr_random = [random.randint(0, n) for _ in range(n)]
    return arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random

sizes = [i for i in range (1, 110000, 10000)]

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

# Построение регрессионных моделей
X_nlogn = np.array([n * math.log(n) if n > 0 else 0 for n in results['Размер']]).reshape(-1, 1)

model_sorted_nlogn = LinearRegression()
model_almost_sorted_nlogn = LinearRegression()
model_desc_sorted_nlogn = LinearRegression()
model_random_nlogn = LinearRegression()

model_sorted_nlogn.fit(X_nlogn, np.array(results['Отсортированный']))
model_almost_sorted_nlogn.fit(X_nlogn, np.array(results['Почти отсортированный']))
model_desc_sorted_nlogn.fit(X_nlogn, np.array(results['Убывающий']))
model_random_nlogn.fit(X_nlogn, np.array(results['Случайный']))

y_pred_sorted_nlogn = model_sorted_nlogn.predict(X_nlogn)
y_pred_almost_sorted_nlogn = model_almost_sorted_nlogn.predict(X_nlogn)
y_pred_desc_sorted_nlogn = model_desc_sorted_nlogn.predict(X_nlogn)
y_pred_random_nlogn = model_random_nlogn.predict(X_nlogn)

# График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], label='Отсортированный массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_sorted_nlogn, label='Регрессия (Отсортированный)', linestyle='-')
plt.title('Производительность Merge Sort для отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Почти отсортированный'], label='Почти отсортированный массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted_nlogn, label='Регрессия (Почти отсортированный)', linestyle='-')
plt.title('Производительность Merge Sort для почти отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Случайный'], label='Случайный массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_random_nlogn, label='Регрессия (Случайный)', linestyle='-')
plt.title('Производительность Merge Sort для случайного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Убывающий'], label='Убывающий массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted_nlogn, label='Регрессия (Убывающий)', linestyle='-')
plt.title('Производительность Merge Sort для убывающего массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Общий график для всех случаев на одном графике
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный', linestyle='None')
plt.plot(results['Размер'], y_pred_sorted_nlogn, label='Регрессия (Отсортированный)', linestyle='-')
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted_nlogn, label='Регрессия (Почти отсортированный)', linestyle='-')
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted_nlogn, label='Регрессия (Убывающий)', linestyle='-')
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный', linestyle='None')
plt.plot(results['Размер'], y_pred_random_nlogn, label='Регрессия (Случайный)', linestyle='-')

plt.title('Производительность Merge Sort: Все случаи')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод уравнений регрессий
print("Уравнение регрессии для отсортированного массива: y =", model_sorted_nlogn.coef_[0], "* n*log(n) +", model_sorted_nlogn.intercept_)
print("Уравнение регрессии для почти отсортированного массива: y =", model_almost_sorted_nlogn.coef_[0], "* n*log(n) +", model_almost_sorted_nlogn.intercept_)
print("Уравнение регрессии для убывающего массива: y =", model_desc_sorted_nlogn.coef_[0], "* n*log(n) +", model_desc_sorted_nlogn.intercept_)
print("Уравнение регрессии для случайного массива: y =", model_random_nlogn.coef_[0], "* n*log(n) +", model_random_nlogn.intercept_)
