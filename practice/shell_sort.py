import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

# Реализация Shell Sort
def shell_sort(arr):
    n = len(arr)
    gap = n // 2  # Инициализируем шаг

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

    return arr

# Функция замера времени выполнения сортировки
def measure_time(arr):
    start_time = time.time()
    shell_sort(arr)
    end_time = time.time()
    return end_time - start_time

# Генерация тестовых массивов
def generate_arrays(n):
    arr_sorted = list(range(n))
    arr_almost_sorted = list(range(int(n * 0.9))) + [random.randint(0, n) for _ in range(int(n * 0.1))]
    arr_desc_sorted = list(range(n, 0, -1))
    arr_random = [random.randint(0, n) for _ in range(n)]
    return arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random

sizes = [i for i in range(1000, 100000, 10000)]
results = {'Размер': [], 'Отсортированный': [], 'Почти отсортированный': [], 'Убывающий': [], 'Случайный': []}

# Измерение времени выполнения для каждого типа массива
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Размер'].append(n)
    results['Отсортированный'].append(measure_time(arr_sorted))
    results['Почти отсортированный'].append(measure_time(arr_almost_sorted))
    results['Убывающий'].append(measure_time(arr_desc_sorted))
    results['Случайный'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

# Преобразование размеров массивов для каждой регрессии
X = np.array(results['Размер']).reshape(-1, 1)

# 1. Линейная регрессия для отсортированного массива (лучший случай, O(n log n))
X_nlogn = np.array([n * math.log(n) for n in results['Размер']]).reshape(-1, 1)
model_sorted = LinearRegression()
model_sorted.fit(X_nlogn, np.array(results['Отсортированный']))

# 2. Полиномиальные признаки для среднего случая O(n^(3/2))
X_n32 = np.array([n ** 1.5 for n in results['Размер']]).reshape(-1, 1)
model_almost_sorted = LinearRegression()
model_random = LinearRegression()
model_almost_sorted.fit(X_n32, np.array(results['Почти отсортированный']))
model_random.fit(X_n32, np.array(results['Случайный']))

# 3. Полиномиальные признаки для худшего случая O(n^2)
X_n2 = np.array([n ** 2 for n in results['Размер']]).reshape(-1, 1)
model_desc_sorted = LinearRegression()
model_desc_sorted.fit(X_n2, np.array(results['Убывающий']))

# Предсказания для каждого случая
y_pred_sorted = model_sorted.predict(X_nlogn)
y_pred_almost_sorted = model_almost_sorted.predict(X_n32)
y_pred_desc_sorted = model_desc_sorted.predict(X_n2)
y_pred_random = model_random.predict(X_n32)

# Построение графиков

# 1. График для отсортированного массива (лучший случай O(n log n))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный массив', linestyle='None')
plt.plot(results['Размер'], y_pred_sorted, label='Регрессия (Отсортированный)', linestyle='-')
plt.title('Производительность Shell Sort для отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива (средний случай O(n^(3/2)))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный массив', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted, label='Полиномиальная регрессия (Почти отсортированный)', linestyle='-')
plt.title('Производительность Shell Sort для почти отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для убывающего массива (худший случай O(n^2))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий массив', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted, label='Полиномиальная регрессия (Убывающий)', linestyle='-')
plt.title('Производительность Shell Sort для убывающего массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для случайного массива (средний случай O(n^(3/2)))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный массив', linestyle='None')
plt.plot(results['Размер'], y_pred_random, label='Полиномиальная регрессия (Случайный)', linestyle='-')
plt.title('Производительность Shell Sort для случайного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Общий график для всех случаев на одном графике
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный (O(n log n))', linestyle='None')
plt.plot(results['Размер'], y_pred_sorted, label='Регрессия (Отсортированный)', linestyle='-')
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный (O(n^(3/2)))', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted, label='Регрессия (Почти отсортированный)', linestyle='-')
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий (O(n^2))', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted, label='Регрессия (Убывающий)', linestyle='-')
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный (O(n^(3/2)))', linestyle='None')
plt.plot(results['Размер'], y_pred_random, label='Регрессия (Случайный)', linestyle='-')

plt.title('Производительность Shell Sort: Все случаи')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод уравнений регрессий
print("Уравнение регрессии для отсортированного массива: y =", model_sorted.coef_[0], "* n*log(n) +", model_sorted.intercept_)
print("Уравнение регрессии для почти отсортированного массива: y =", model_almost_sorted.coef_[0], "* n^(3/2) +", model_almost_sorted.intercept_)
print("Уравнение регрессии для случайного массива: y =", model_random.coef_[0], "* n^(3/2) +", model_random.intercept_)
print("Уравнение регрессии для убывающего массива: y =", model_desc_sorted.coef_[0], "* n^2 +", model_desc_sorted.intercept_)
