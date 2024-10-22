import numpy as np
import matplotlib.pyplot as plt

# Функции сложности для всех случаев Heap Sort
def heap_sort_complexity(n):
    return n * np.log2(n)  # O(n log n) для всех случаев

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = [i for i in range(1000, 10001, 1000)]

# Вычисляем значения сложности для всех случаев
best_case_values = heap_sort_complexity(np.array(sizes))
avg_case_values = heap_sort_complexity(np.array(sizes))
worst_case_values = heap_sort_complexity(np.array(sizes))

# Отрисовка одного графика для всех случаев
plt.figure(figsize=(10, 6))
plt.plot(sizes, best_case_values, label='Best Case (O(n log n))', color='green')
plt.plot(sizes, avg_case_values, label='Average Case (O(n log n))', color='blue')
plt.plot(sizes, worst_case_values, label='Worst Case (O(n log n))', color='red')

# Настройки графика
plt.title('Heap Sort Time Complexity (All Cases)')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
