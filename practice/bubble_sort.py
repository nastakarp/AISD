import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Функция сортировки пузырьком (bubble sort)
def bubble_sort(arr):
    n = len(arr)
    # Проходим по массиву
    for i in range(n):
        # Последние i элементов уже на своих местах
        for j in range(0, n-i-1):
            # Меняем элементы местами, если они стоят в неправильном порядке
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def measure_time(arr):
    start_time = time.time()
    bubble_sort(arr)  # Используем сортировку пузырьком
    end_time = time.time()
    return end_time - start_time

def generate_arrays(n):
    arr_sorted = list(range(n))
    arr_almost_sorted = list(range(int(n * 0.9))) + [random.randint(0, n) for _ in range(int(n * 0.1))]
    arr_desc_sorted = list(range(n, 0, -1))
    arr_random = [random.randint(0, n) for _ in range(n)]
    return arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random

sizes = [i for i in range (1000, 10001, 1000)]
results = {'Size': [], 'Sorted': [], 'Almost Sorted': [], 'Descending': [], 'Random': []}

for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Size'].append(n)
    results['Sorted'].append(measure_time(arr_sorted))
    results['Almost Sorted'].append(measure_time(arr_almost_sorted))
    results['Descending'].append(measure_time(arr_desc_sorted))
    results['Random'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

X = np.array(results['Size']).reshape(-1, 1)

# Применяем полиномиальные признаки (второго порядка)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Модели для каждого типа массива
model_sorted = LinearRegression()
model_almost_sorted = LinearRegression()
model_desc_sorted = LinearRegression()
model_random = LinearRegression()

# Обучаем модели на квадратичных признаках
model_sorted.fit(X_poly, np.array(results['Sorted']))
model_almost_sorted.fit(X_poly, np.array(results['Almost Sorted']))
model_desc_sorted.fit(X_poly, np.array(results['Descending']))
model_random.fit(X_poly, np.array(results['Random']))

# Получаем предсказанные значения для регрессионных линий
y_pred_sorted = model_sorted.predict(X_poly)
y_pred_almost_sorted = model_almost_sorted.predict(X_poly)
y_pred_desc_sorted = model_desc_sorted.predict(X_poly)
y_pred_random = model_random.predict(X_poly)

# Построение графиков для каждого типа массива

# 1. График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], 'o', label='Sorted Array', linestyle='None')  # Только точки для данных
plt.plot(results['Size'], y_pred_sorted, label='Quadratic Regression (Sorted)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Bubble Sort Performance on Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
plt.show()

# 2. График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Almost Sorted'], 'o', label='Almost Sorted Array', linestyle='None')  # Только точки для данных
plt.plot(results['Size'], y_pred_almost_sorted, label='Quadratic Regression (Almost Sorted)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Bubble Sort Performance on Almost Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
plt.show()

# 3. График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Descending'], 'o', label='Descending Array', linestyle='None')  # Только точки для данных
plt.plot(results['Size'], y_pred_desc_sorted, label='Quadratic Regression (Descending)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Bubble Sort Performance on Descending Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
plt.show()

# 4. График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Random'], 'o', label='Random Array', linestyle='None')  # Только точки для данных
plt.plot(results['Size'], y_pred_random, label='Quadratic Regression (Random)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Bubble Sort Performance on Random Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
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

plt.title('Bubble Sort Performance: All Cases')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()