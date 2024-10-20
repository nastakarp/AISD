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
sizes = [i for i in range (1000,100001, 10000)]

# Словарь для хранения времени выполнения
results = {'Size': [], 'Sorted': [], 'Almost Sorted': [], 'Descending': [], 'Random': []}

# Замер времени выполнения для каждого типа массивов и разных размеров
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Size'].append(n)
    results['Sorted'].append(measure_time(arr_sorted))
    results['Almost Sorted'].append(measure_time(arr_almost_sorted))
    results['Descending'].append(measure_time(arr_desc_sorted))
    results['Random'].append(measure_time(arr_random))
# Вывод таблицы с результатами перед построением графика
df_results = pd.DataFrame(results)
print(df_results)  # Явный вывод таблицы


# Преобразуем размеры массивов в n * log(n) для лучшего, среднего и случайного случая
X_nlogn = np.array([n * math.log(n) for n in results['Size']]).reshape(-1, 1)

# Преобразуем размеры массивов в n^2 для худшего случая
X_n2 = np.array([n ** 2 for n in results['Size']]).reshape(-1, 1)

# Модели линейной регрессии
# Для n*log(n)
model_sorted_nlogn = LinearRegression()
model_almost_sorted_nlogn = LinearRegression()
model_random_nlogn = LinearRegression()

# Для n^2
model_desc_sorted_n2 = LinearRegression()

# Фитинг моделей для n*log(n)
model_sorted_nlogn.fit(X_nlogn, np.array(results['Sorted']))
model_almost_sorted_nlogn.fit(X_nlogn, np.array(results['Almost Sorted']))
model_random_nlogn.fit(X_nlogn, np.array(results['Random']))

# Фитинг модели для n^2 (худший случай)
model_desc_sorted_n2.fit(X_n2, np.array(results['Descending']))

# Получаем предсказанные значения для регрессионных линий
y_pred_sorted_nlogn = model_sorted_nlogn.predict(X_nlogn)
y_pred_almost_sorted_nlogn = model_almost_sorted_nlogn.predict(X_nlogn)
y_pred_random_nlogn = model_random_nlogn.predict(X_nlogn)
y_pred_desc_sorted_n2 = model_desc_sorted_n2.predict(X_n2)

# Построение графиков для каждого типа массива на отдельных графиках

# 1. График для отсортированного массива (лучший случай, n*log(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], label='Sorted Array', marker='o', linestyle='None')  # Точки
plt.plot(results['Size'], y_pred_sorted_nlogn, label='Regression (Sorted)', linestyle='-')  # Сплошная линия
plt.title('Quick Sort Performance on Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 2. График для почти отсортированного массива (средний случай, n*log(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Almost Sorted'], label='Almost Sorted Array', marker='o', linestyle='None')  # Точки
plt.plot(results['Size'], y_pred_almost_sorted_nlogn, label='Regression (Almost Sorted)', linestyle='-')  # Сплошная линия
plt.title('Quick Sort Performance on Almost Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 3. График для случайного массива (средний случай, n*log(n))
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Random'], label='Random Array', marker='o', linestyle='None')  # Точки
plt.plot(results['Size'], y_pred_random_nlogn, label='Regression (Random)', linestyle='-')  # Сплошная линия
plt.title('Quick Sort Performance on Random Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# 4. График для убывающего массива (худший случай, n^2)
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Descending'], label='Descending Array', marker='o', linestyle='None')  # Точки
plt.plot(results['Size'], y_pred_desc_sorted_n2, label='Regression (Descending)', linestyle='-')  # Сплошная линия
plt.title('Quick Sort Performance on Descending Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()
# Общий график для всех случаев на одном графике
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], 'o', label='Sorted', linestyle='None')
plt.plot(results['Size'], y_pred_sorted_nlogn, label='Regression (Sorted)', linestyle='-')
plt.plot(results['Size'], results['Almost Sorted'], 'o', label='Almost Sorted', linestyle='None')
plt.plot(results['Size'], y_pred_almost_sorted_nlogn, label='Regression (Almost Sorted)', linestyle='-')
plt.plot(results['Size'], results['Descending'], 'o', label='Descending', linestyle='None')
plt.plot(results['Size'], y_pred_desc_sorted_n2, label='Regression (Descending)', linestyle='-')
plt.plot(results['Size'], results['Random'], 'o', label='Random', linestyle='None')
plt.plot(results['Size'], y_pred_random_nlogn, label='Regression (Random)', linestyle='-')

plt.title('Quick Sort Performance: All Cases')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()