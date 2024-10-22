import numpy as np
import matplotlib.pyplot as plt

# Функции сложности для всех случаев Merge Sort
def merge_sort_complexity(n):
    return n * np.log2(n)  # O(n log n) для всех случаев

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = [i for i in range(0, 11000, 1000)]

# Вычисляем значения сложности для всех случаев
best_case_values = merge_sort_complexity(np.array(sizes))
avg_case_values = merge_sort_complexity(np.array(sizes))
worst_case_values = merge_sort_complexity(np.array(sizes))

# Отрисовка одного графика для всех случаев
plt.figure(figsize=(10, 6))
plt.plot(sizes, best_case_values, label='Best Case', color='green')
plt.plot(sizes, avg_case_values, label='Average Case', color='blue')
plt.plot(sizes, worst_case_values, label='Worst Case', color='red')

# Настройки графика
plt.title('Merge Sort Time Complexity (All Cases)')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
