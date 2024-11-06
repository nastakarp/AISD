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

sizes = [i for i in range (1000, 100001, 10000)]
results = {'Размер': [], 'Отсортированный': [], 'Почти отсортированный': [], 'Убывающий': [], 'Случайный': []}

# Измерение времени выполнения для каждого массива
for n in sizes:
    arr_sorted, arr_almost_sorted, arr_desc_sorted, arr_random = generate_arrays(n)
    results['Размер'].append(n)
    results['Отсортированный'].append(measure_time(arr_sorted))
    results['Почти отсортированный'].append(measure_time(arr_almost_sorted))
    results['Убывающий'].append(measure_time(arr_desc_sorted))
    results['Случайный'].append(measure_time(arr_random))

df_results = pd.DataFrame(results)
print(df_results)

X = np.array(results['Размер']).reshape(-1, 1)

# Применяем полиномиальные признаки (второго порядка)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Модели для каждого типа массива
model_sorted = LinearRegression()
model_almost_sorted = LinearRegression()
model_desc_sorted = LinearRegression()
model_random = LinearRegression()

# Обучаем модели на квадратичных признаках
model_sorted.fit(X_poly, np.array(results['Отсортированный']))
model_almost_sorted.fit(X_poly, np.array(results['Почти отсортированный']))
model_desc_sorted.fit(X_poly, np.array(results['Убывающий']))
model_random.fit(X_poly, np.array(results['Случайный']))

# Получаем предсказанные значения для регрессионных линий
y_pred_sorted = model_sorted.predict(X_poly)
y_pred_almost_sorted = model_almost_sorted.predict(X_poly)
y_pred_desc_sorted = model_desc_sorted.predict(X_poly)
y_pred_random = model_random.predict(X_poly)

# Построение графиков для каждого типа массива

# 1. График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Отсортированный'], 'o', label='Отсортированный массив', linestyle='None')  # Только точки для данных
plt.plot(results['Размер'], y_pred_sorted, label='Квадратичная регрессия (Отсортированный)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Производительность Bubble Sort на отсортированном массиве')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
plt.show()

# 2. График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Почти отсортированный'], 'o', label='Почти отсортированный массив', linestyle='None')  # Только точки для данных
plt.plot(results['Размер'], y_pred_almost_sorted, label='Квадратичная регрессия (Почти отсортированный)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Производительность Bubble Sort на почти отсортированном массиве')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
plt.show()

# 3. График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Убывающий'], 'o', label='Убывающий массив', linestyle='None')  # Только точки для данных
plt.plot(results['Размер'], y_pred_desc_sorted, label='Квадратичная регрессия (Убывающий)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Производительность Bubble Sort на убывающем массиве')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
plt.show()

# 4. График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Размер'], results['Случайный'], 'o', label='Случайный массив', linestyle='None')  # Только точки для данных
plt.plot(results['Размер'], y_pred_random, label='Квадратичная регрессия (Случайный)', linestyle='-')  # Сплошная линия для регрессии
plt.title('Производительность Bubble Sort на случайном массиве')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.xlim(min(sizes), max(sizes))  # Устанавливаем одинаковые пределы по оси X
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

plt.title('Производительность Bubble Sort: Все случаи')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (секунды)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод уравнений регрессий
print("Уравнение регрессии для отсортированного массива: y =", model_sorted.coef_[1], "* n +", model_sorted.coef_[2], "* n^2 +", model_sorted.intercept_)
print("Уравнение регрессии для почти отсортированного массива: y =", model_almost_sorted.coef_[1], "* n +", model_almost_sorted.coef_[2], "* n^2 +", model_almost_sorted.intercept_)
print("Уравнение регрессии для убывающего массива: y =", model_desc_sorted.coef_[1], "* n +", model_desc_sorted.coef_[2], "* n^2 +", model_desc_sorted.intercept_)
print("Уравнение регрессии для случайного массива: y =", model_random.coef_[1], "* n +", model_random.coef_[2], "* n^2 +", model_random.intercept_)
