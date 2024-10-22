import numpy as np
import matplotlib.pyplot as plt

# Функции сложности для лучшего, среднего и худшего случая
def bubble_sort_best_case(n):
    return n * (n - 1) / 2  # T_B(n) ~ n(n-1)/2

def bubble_sort_avg_case(n):
    return n * (n - 1) / 4  # T_A(n) ~ n(n-1)/4

def bubble_sort_worst_case(n):
    return n * (n - 1) / 2  # T_W(n) ~ n(n-1)/2

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = [i for i in range(1000, 10001, 1000)]

# Вычисляем значения сложности для каждого случая
best_case_values = bubble_sort_best_case(np.array(sizes))
avg_case_values = bubble_sort_avg_case(np.array(sizes))
worst_case_values = bubble_sort_worst_case(np.array(sizes))

# Отрисовка графика для всех случаев
plt.figure(figsize=(10, 6))
plt.plot(sizes, best_case_values, label='Best Case', color='green')
plt.plot(sizes, avg_case_values, label='Average Case', color='blue')
plt.plot(sizes, worst_case_values, label='Worst Case', color='red')

# Настройки графика
plt.title('Bubble Sort Time Complexity')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()
