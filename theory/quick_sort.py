import numpy as np
import matplotlib.pyplot as plt

# Функции сложности для разных случаев Quick Sort
def quick_sort_best_case(n):
    return n * np.log2(n)  # O(n log n) для лучшего случая

def quick_sort_avg_case(n):
    return n * np.log2(n)  # O(n log n) для среднего случая

def quick_sort_worst_case(n):
    return n ** 2  # O(n^2) для худшего случая

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = [i for i in range(0, 10001, 1000)]

# Вычисляем значения сложности для каждого случая
best_case_values = quick_sort_best_case(np.array(sizes))
avg_case_values = quick_sort_avg_case(np.array(sizes))
worst_case_values = quick_sort_worst_case(np.array(sizes))

# Отрисовка одного графика для всех случаев
plt.figure(figsize=(10, 6))
plt.plot(sizes, best_case_values, label='Best Case (O(n log n))', color='green')
plt.plot(sizes, avg_case_values, label='Average Case (O(n log n))', color='blue')

# Настройки графика
plt.title('Quick Sort Time Complexity')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
# Отрисовка одного графика для всех случаев
plt.figure(figsize=(10, 6))
plt.plot(sizes, worst_case_values, label='Worst Case (O(n^2))', color='red')

# Настройки графика
plt.title('Quick Sort Time Complexity')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
