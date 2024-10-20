import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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


def time_sort(sort_func, arr):
    start_time = time.time()
    sort_func(arr.copy())
    return time.time() - start_time


# Генерация массивов для тестирования
sizes = np.array([i for i in range(1000, 10001, 1000)]).reshape(-1, 1)
results = {'Selection Sort': [],'Insertion Sort': [],'Bubble Sort': [],'Quick Sort': [], 'Heap Sort': [], 'Merge Sort': [], 'Shell Sort': [], 'Shell Sort Hibbard': [],
           'Shell Sort Pratt': []}

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

# Построение регрессионных линий
plt.figure(figsize=(12, 8))

# Модели линейной регрессии для каждой сортировки
for sort_name, times in results.items():
    times = np.array(times).reshape(-1, 1)

    # Создаем и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(sizes, times)

    # Предсказываем значения на основе размеров массивов
    predicted_times = model.predict(sizes)

    # Отображаем исходные данные как точки и регрессионную линию
    plt.plot(sizes, predicted_times, label=f'{sort_name} (Regression)')
    plt.scatter(sizes, times, label=f'{sort_name} (Data)', s=20)

plt.title('Временные затраты на сортировку')
plt.xlabel('Размер массива')
plt.ylabel('Время (сек)')
plt.legend()
plt.grid(True)
plt.show()