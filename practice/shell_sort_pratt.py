import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

# Реализация Shell Sort с последовательностью Пратта
def shell_sort_pratt(arr):
    n = len(arr)
    # Формируем последовательность Пратта
    gaps = []
    i = j = 1
    while 2**i - 1 < n or 3**j - 1 < n:
        if 2**i - 1 < n:
            gaps.append(2**i - 1)
            i += 1
        if 3**j - 1 < n:
            gaps.append(3**j - 1)
            j += 1

    gaps = sorted(set(gaps), reverse=True)  # Уникальные шаги, сортированные по убыванию

    # Сортировка с шагами по Пратту
    for gap in gaps:
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
    shell_sort_pratt(arr)
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
sizes = [i for i in range(1, 100001, 10000)]
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

# 1. Линейная регрессия для отсортированного массива (лучший случай, O(n))
model_sorted = LinearRegression()
model_sorted.fit(X, np.array(results['Отсортированный']))

# 2. Регрессия для среднего и худшего случаев O(n (log n)^2)
X_nlogn2 = np.array([n * (math.log(n)**2) for n in results['Размер']]).reshape(-1, 1)
model_almost_sorted = LinearRegression()
model_random = LinearRegression()
model_desc_sorted = LinearRegression()

# Обучение моделей
model_almost_sorted.fit(X_nlogn2, np.array(results['Почти отсортированный']))
model_random.fit(X_nlogn2, np.array(results['Случайный']))
model_desc_sorted.fit(X_nlogn2, np.array(results['Убывающий']))

# Предсказания для каждого случая
y_pred_sorted = model_sorted.predict(X)
y_pred_almost_sorted = model_almost_sorted.predict(X_nlogn2)
y_pred_desc_sorted = model_desc_sorted.predict(X_nlogn2)
y_pred_random = model_random.predict(X_nlogn2)

# Построение графиков

# 1. График для отсортированного массива (лучший случай O(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный массив', linestyle='None')  # Точки для данных
plt.plot(results['Размер'], y_pred_sorted, label='Регрессия (Отсортированный)', linestyle='-')  # Линейная регрессия
plt.title('Производительность Shell Sort Pratt для отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива (средний случай O(n (log n)^2))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный массив', linestyle='None')  # Точки для данных
plt.plot(results['Размер'], y_pred_almost_sorted, label='Регрессия (Почти отсортированный)', linestyle='-')  # Полиномиальная регрессия
plt.title('Производительность Shell Sort Pratt для почти отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для убывающего массива (худший случай O(n (log n)^2))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий массив', linestyle='None')  # Точки для данных
plt.plot(results['Размер'], y_pred_desc_sorted, label='Регрессия (Убывающий)', linestyle='-')  # Полиномиальная регрессия
plt.title('Производительность Shell Sort Pratt для убывающего массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для случайного массива (средний случай O(n (log n)^2))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный массив', linestyle='None')  # Точки для данных
plt.plot(results['Размер'], y_pred_random, label='Регрессия (Случайный)', linestyle='-')  # Полиномиальная регрессия
plt.title('Производительность Shell Sort Pratt для случайного массива')
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

plt.title('Производительность Shell Sort Pratt: Все случаи')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод уравнений регрессии
print("Уравнение регрессии для отсортированного массива: y =", model_sorted.coef_[0], "* n +", model_sorted.intercept_)
print("Уравнение регрессии для почти отсортированного массива: y =", model_almost_sorted.coef_[0], "* n*(log(n)^2) +", model_almost_sorted.intercept_)
print("Уравнение регрессии для случайного массива: y =", model_random.coef_[0], "* n*(log(n)^2) +", model_random.intercept_)
print("Уравнение регрессии для убывающего массива: y =", model_desc_sorted.coef_[0], "* n*(log(n)^2) +", model_desc_sorted.intercept_)
