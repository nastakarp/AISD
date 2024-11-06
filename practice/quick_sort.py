import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
# Реализация Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        less = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]
        return quick_sort(less) + equal + quick_sort(greater)

# Функция для замера времени выполнения
def measure_time(arr):
    start_time = time.time()
    quick_sort(arr)
    end_time = time.time()
    return end_time - start_time

# Генерация массивов в зависимости от размера n
def generate_arrays(n):
    arr_sorted = list(range(n))
    arr_almost_sorted = list(range(int(n * 0.9))) + [random.randint(0, n) for _ in range(int(n * 0.1))]
    arr_desc_sorted = list(range(n, 0, -1))
    arr_random = [random.randint(0, n) for _ in range(n)]
    return arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random

# Размеры массивов для тестирования
sizes = [i for i in range (1,100001, 10000)]

# Словарь для хранения времени выполнения
# Замена ключей на русские слова
results = {'Размер': [], 'Отсортированный': [], 'Почти отсортированный': [], 'Убывающий': [], 'Случайный': []}

# Замер времени выполнения для каждого типа массивов и разных размеров
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Размер'].append(n)
    results['Отсортированный'].append(measure_time(arr_sorted))
    results['Почти отсортированный'].append(measure_time(arr_almost_sorted))
    results['Убывающий'].append(measure_time(arr_desc_sorted))
    results['Случайный'].append(measure_time(arr_random))

# Вывод таблицы с результатами перед построением графика
df_results = pd.DataFrame(results)
print(df_results)

# Преобразуем размеры массивов в n * log(n) для лучшего, среднего и случайного случая
X_nlogn = np.array([n * math.log(n) for n in results['Размер']]).reshape(-1, 1)

# Преобразуем размеры массивов в n^2 для худшего случая
X_n2 = np.array([n ** 2 for n in results['Размер']]).reshape(-1, 1)

# Модели линейной регрессии
# Для n*log(n)
model_sorted_nlogn = LinearRegression()
model_almost_sorted_nlogn = LinearRegression()
model_random_nlogn = LinearRegression()

# Для n^2
model_desc_sorted_n2 = LinearRegression()

# Обучение моделей для n*log(n)
model_sorted_nlogn.fit(X_nlogn, np.array(results['Отсортированный']))
model_almost_sorted_nlogn.fit(X_nlogn, np.array(results['Почти отсортированный']))
model_random_nlogn.fit(X_nlogn, np.array(results['Случайный']))

# Обучение модели для n^2 (худший случай)
model_desc_sorted_n2.fit(X_n2, np.array(results['Убывающий']))

# Получаем предсказанные значения для регрессионных линий
y_pred_sorted_nlogn = model_sorted_nlogn.predict(X_nlogn)
y_pred_almost_sorted_nlogn = model_almost_sorted_nlogn.predict(X_nlogn)
y_pred_random_nlogn = model_random_nlogn.predict(X_nlogn)
y_pred_desc_sorted_n2 = model_desc_sorted_n2.predict(X_n2)

# Построение графиков для каждого типа массива на отдельных графиках

# 1. График для отсортированного массива (лучший случай, n*log(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], label='Отсортированный массив', marker='o', linestyle='None')  # Точки
plt.plot(results['Размер'], y_pred_sorted_nlogn, label='Регрессия (Отсортированный)', linestyle='-')  # Сплошная линия
plt.title('Производительность Quick Sort для отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива (средний случай, n*log(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Почти отсортированный'], label='Почти отсортированный массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_almost_sorted_nlogn, label='Регрессия (Почти отсортированный)', linestyle='-')
plt.title('Производительность Quick Sort для почти отсортированного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для случайного массива (средний случай, n*log(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Случайный'], label='Случайный массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_random_nlogn, label='Регрессия (Случайный)', linestyle='-')
plt.title('Производительность Quick Sort для случайного массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для убывающего массива (худший случай, n^2)
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Убывающий'], label='Убывающий массив', marker='o', linestyle='None')
plt.plot(results['Размер'], y_pred_desc_sorted_n2, label='Регрессия (Убывающий)', linestyle='-')
plt.title('Производительность Quick Sort для убывающего массива')
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
plt.plot(results['Размер'], y_pred_desc_sorted_n2, label='Регрессия (Убывающий)', linestyle='-')
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный', linestyle='None')
plt.plot(results['Размер'], y_pred_random_nlogn, label='Регрессия (Случайный)', linestyle='-')

plt.title('Производительность Quick Sort: Все случаи')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод уравнений регрессии
print("Уравнение регрессии для отсортированного массива: y =", model_sorted_nlogn.coef_[0], "* n*log(n) +", model_sorted_nlogn.intercept_)
print("Уравнение регрессии для почти отсортированного массива: y =", model_almost_sorted_nlogn.coef_[0], "* n*log(n) +", model_almost_sorted_nlogn.intercept_)
print("Уравнение регрессии для случайного массива: y =", model_random_nlogn.coef_[0], "* n*log(n) +", model_random_nlogn.intercept_)
print("Уравнение регрессии для убывающего массива: y =", model_desc_sorted_n2.coef_[0], "* n^2 +", model_desc_sorted_n2.intercept_)
