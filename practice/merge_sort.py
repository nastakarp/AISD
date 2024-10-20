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

sizes = [i for i in range (1000, 100001, 10000)]

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

# Все предсказания теперь будут основаны на n*log(n)
X_nlogn = np.array([n * math.log(n) for n in results['Size']]).reshape(-1, 1)

model_sorted_nlogn = LinearRegression()
model_almost_sorted_nlogn = LinearRegression()
model_desc_sorted_nlogn = LinearRegression()
model_random_nlogn = LinearRegression()

model_sorted_nlogn.fit(X_nlogn, np.array(results['Sorted']))
model_almost_sorted_nlogn.fit(X_nlogn, np.array(results['Almost Sorted']))
model_desc_sorted_nlogn.fit(X_nlogn, np.array(results['Descending']))
model_random_nlogn.fit(X_nlogn, np.array(results['Random']))

y_pred_sorted_nlogn = model_sorted_nlogn.predict(X_nlogn)
y_pred_almost_sorted_nlogn = model_almost_sorted_nlogn.predict(X_nlogn)
y_pred_desc_sorted_nlogn = model_desc_sorted_nlogn.predict(X_nlogn)
y_pred_random_nlogn = model_random_nlogn.predict(X_nlogn)

# График для отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Sorted'], label='Sorted Array', marker='o', linestyle='None')
plt.plot(results['Size'], y_pred_sorted_nlogn, label='Regression (Sorted)', linestyle='-')
plt.title('Merge Sort Performance on Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# График для почти отсортированного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Almost Sorted'], label='Almost Sorted Array', marker='o', linestyle='None')
plt.plot(results['Size'], y_pred_almost_sorted_nlogn, label='Regression (Almost Sorted)', linestyle='-')
plt.title('Merge Sort Performance on Almost Sorted Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# График для случайного массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Random'], label='Random Array', marker='o', linestyle='None')
plt.plot(results['Size'], y_pred_random_nlogn, label='Regression (Random)', linestyle='-')
plt.title('Merge Sort Performance on Random Array')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()

# График для убывающего массива
plt.figure(figsize=(10, 6))
plt.plot(results['Size'], results['Descending'], label='Descending Array', marker='o', linestyle='None')
plt.plot(results['Size'], y_pred_desc_sorted_nlogn, label='Regression (Descending)', linestyle='-')
plt.title('Merge Sort Performance on Descending Array')
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
plt.plot(results['Size'], y_pred_desc_sorted_nlogn, label='Regression (Descending)', linestyle='-')
plt.plot(results['Size'], results['Random'], 'o', label='Random', linestyle='None')
plt.plot(results['Size'], y_pred_random_nlogn, label='Regression (Random)', linestyle='-')

plt.title('Merge Sort Performance: All Cases')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()