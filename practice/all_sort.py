import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Сортировка пузырьком
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# Сортировка вставками
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# Сортировка выбором
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Быстрая сортировка
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Пирамидальная сортировка
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[i] < arr[left]:
        largest = left
    if right < n and arr[largest] < arr[right]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n//2, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

# Сортировка слиянием
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

# Сортировка Шелла
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
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


# Функции сортировки остаются такими же

def time_sort(sort_func, arr):
    start_time = time.time()
    sort_func(arr.copy())
    return time.time() - start_time


# Генерация массивов для тестирования
sizes = np.array([i for i in range(1000, 10001, 1000)]).reshape(-1, 1)
results = {'Selection Sort': [], 'Insertion Sort': [], 'Bubble Sort': [], 'Quick Sort': [], 'Heap Sort': [],
           'Merge Sort': [], 'Shell Sort': [], 'Shell Sort Hibbard': [], 'Shell Sort Pratt': []}

for size in sizes.flatten():
    arr_random = np.random.randint(0, 100000, size)
    results['Selection Sort'].append(time_sort(selection_sort, arr_random))
    results['Insertion Sort'].append(time_sort(insertion_sort, arr_random))
    results['Bubble Sort'].append(time_sort(bubble_sort, arr_random))
    results['Quick Sort'].append(time_sort(quick_sort, arr_random))
    results['Heap Sort'].append(time_sort(heap_sort, arr_random))
    results['Merge Sort'].append(time_sort(merge_sort, arr_random))
    results['Shell Sort'].append(time_sort(shell_sort, arr_random))
    results['Shell Sort Hibbard'].append(time_sort(shell_sort_hibbard, arr_random))
    results['Shell Sort Pratt'].append(time_sort(shell_sort_pratt, arr_random))

# Первый график для медленных алгоритмов с квадратичной регрессией
plt.figure(figsize=(12, 8))

# Квадратичная регрессия для медленных алгоритмов
poly = PolynomialFeatures(degree=2)

for sort_name in ['Bubble Sort', 'Insertion Sort', 'Selection Sort']:
    times = np.array(results[sort_name]).reshape(-1, 1)

    # Трансформация для квадратичной регрессии
    sizes_poly = poly.fit_transform(sizes)

    # Модель квадратичной регрессии
    model = LinearRegression()
    model.fit(sizes_poly, times)

    # Предсказания
    predicted_times = model.predict(sizes_poly)

    # Построение графика
    plt.plot(sizes, predicted_times, label=f'{sort_name} (Quadratic Regression)')
    plt.scatter(sizes, times, label=f'{sort_name} (Data)', s=20)

plt.title('Временные затраты на сортировку (квадратичные алгоритмы)')
plt.xlabel('Размер массива')
plt.ylabel('Время (сек)')
plt.legend()
plt.grid(True)
plt.show()

# Второй график для быстрых алгоритмов с линейной регрессией
plt.figure(figsize=(12, 8))

# Линейная регрессия для алгоритмов O(n log n)
for sort_name in ['Quick Sort', 'Heap Sort', 'Merge Sort', 'Shell Sort Hibbard']:
    times = np.array(results[sort_name]).reshape(-1, 1)

    # Линейная регрессия
    model = LinearRegression()
    model.fit(sizes, times)

    # Предсказания
    predicted_times = model.predict(sizes)

    # Построение графика
    plt.plot(sizes, predicted_times, label=f'{sort_name} (Regression)')
    plt.scatter(sizes, times, label=f'{sort_name} (Data)', s=20)

# Полиномиальная регрессия для Shell Sort с O(n^(3/2))
poly_3_2 = PolynomialFeatures(degree=2)  # Используем квадратичную регрессию для приближения
times_shell = np.array(results['Shell Sort']).reshape(-1, 1)
sizes_poly_3_2 = poly_3_2.fit_transform(sizes)
model_shell = LinearRegression()
model_shell.fit(sizes_poly_3_2, times_shell)
predicted_times_shell = model_shell.predict(sizes_poly_3_2)
plt.plot(sizes, predicted_times_shell, label='Shell Sort (Approximation)')
plt.scatter(sizes, times_shell, label='Shell Sort (Data)', s=20)

# Полиномиальная регрессия для Shell Sort Pratt с O((log n)^2)
log_sizes = np.log(sizes)  # Логарифмическая трансформация размеров
times_pratt = np.array(results['Shell Sort Pratt']).reshape(-1, 1)
poly_log = PolynomialFeatures(degree=2)
sizes_poly_log = poly_log.fit_transform(log_sizes)
model_pratt = LinearRegression()
model_pratt.fit(sizes_poly_log, times_pratt)
predicted_times_pratt = model_pratt.predict(sizes_poly_log)
plt.plot(sizes, predicted_times_pratt, label='Shell Sort Pratt (Approximation)')
plt.scatter(sizes, times_pratt, label='Shell Sort Pratt (Data)', s=20)

plt.title('Временные затраты на сортировку (неквадратичные алгоритмы)')
plt.xlabel('Размер массива')
plt.ylabel('Время (сек)')
plt.legend()
plt.grid(True)
plt.show()