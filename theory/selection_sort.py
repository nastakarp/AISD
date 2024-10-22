import numpy as np
import matplotlib.pyplot as plt

# Функции сложности для лучшего, среднего и худшего случая
def selection_sort_complexity(n):
    return n * (n - 1) / 2

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = [i for i in range(1000, 10001, 1000)]

# Вычисляем значения сложности для каждого случая
best_case_values = selection_sort_complexity(np.array(sizes))
avg_case_values = selection_sort_complexity(np.array(sizes))
worst_case_values = selection_sort_complexity(np.array(sizes))

# Отрисовка графика
plt.figure(figsize=(10, 6))
plt.plot(sizes, best_case_values, label='Best Case (O(n^2))', color='green')
plt.plot(sizes, avg_case_values, label='Average Case (O(n^2))', color='blue')
plt.plot(sizes, worst_case_values, label='Worst Case (O(n^2))', color='red')

# Настройки графика
plt.title('Selection Sort Time Complexity')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
