import numpy as np
import matplotlib.pyplot as plt

# Функции сложности для разных случаев Insertion Sort
def insertion_sort_best_case(n):
    return n  # T_B(n) ~ O(n)

def insertion_sort_avg_case(n):
    return n * (n - 1) / 4 # T_A(n) ~ O(n^2)

def insertion_sort_worst_case(n):
    return n * (n - 1) / 2  # T_W(n) ~ O(n^2)

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = [i for i in range(1000, 10001, 1000)]

# Вычисляем значения сложности для каждого случая
best_case_values = insertion_sort_best_case(np.array(sizes))
avg_case_values = insertion_sort_avg_case(np.array(sizes))
worst_case_values = insertion_sort_worst_case(np.array(sizes))

# Отрисовка первого графика для лучшего случая (O(n))
plt.figure(figsize=(10, 6))
plt.plot(sizes, best_case_values, label='Best Case (O(n))', color='green')

# Настройки первого графика
plt.title('Insertion Sort Time Complexity (Best Case)')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение первого графика
plt.show()

# Отрисовка второго графика для среднего и худшего случаев (O(n^2))
plt.figure(figsize=(10, 6))
plt.plot(sizes, avg_case_values, label='Average Case (O(n^2))', color='blue')
plt.plot(sizes, worst_case_values, label='Worst Case (O(n^2))', color='red')

# Настройки второго графика
plt.title('Insertion Sort Time Complexity (Average and Worst Case)')
plt.xlabel('Input size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)

# Отображение второго графика
plt.show()
