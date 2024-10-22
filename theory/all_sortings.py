import numpy as np
import matplotlib.pyplot as plt

# Функции для различных алгоритмов сортировки
def selection_sort(n):
    return n * (n - 1) / 2

def insertion_sort(n):
    return n * (n - 1) / 4

def bubble_sort(n):
    return n * (n - 1) / 4

def merge_sort(n):
    return n * np.log2(n)

def shell_sort(n):
    return n ** 1.5

def shell_sort_hibbard(n):
    return n * np.log2(n)

def shell_sort_pratt(n):
    return n * (np.log2(n) ** 2)

def quick_sort(n):
    return n * np.log2(n)

def heap_sort(n):
    return n * np.log2(n)

# Создаем массив размеров входных данных от 1000 до 10000 с шагом 1000
sizes = np.arange(1000, 10001, 1000)

# Вычисление значений сложности для первых трёх алгоритмов
selection_sort_values = selection_sort(sizes)
insertion_sort_values = insertion_sort(sizes)
bubble_sort_values = bubble_sort(sizes)

# Отрисовка первого графика
plt.figure(figsize=(10, 6))
plt.plot(sizes, selection_sort_values, label='Selection Sort', color='green')
plt.plot(sizes, insertion_sort_values, label='Insertion Sort', color='blue')
plt.plot(sizes, bubble_sort_values, label='Bubble Sort', color='red')

# Настройки графика
plt.title('Time Complexity')
plt.xlabel('Input Size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)
plt.show()
sizes = np.arange(1000, 100001, 10000)
# Вычисление значений сложности для остальных алгоритмов
merge_sort_values = merge_sort(sizes)
shell_sort_values = shell_sort(sizes)
shell_sort_hibbard_values = shell_sort_hibbard(sizes)
shell_sort_pratt_values = shell_sort_pratt(sizes)
quick_sort_values = quick_sort(sizes)
heap_sort_values = heap_sort(sizes)

# Отрисовка второго графика
plt.figure(figsize=(10, 6))
plt.plot(sizes, merge_sort_values, label='Merge Sort', color='purple')
plt.plot(sizes, shell_sort_values, label='Shell Sort', color='orange')
plt.plot(sizes, shell_sort_hibbard_values, label='Shell Sort (Hibbard)', color='brown')
plt.plot(sizes, shell_sort_pratt_values, label='Shell Sort (Pratt)', color='cyan')
plt.plot(sizes, quick_sort_values, label='Quick Sort', color='pink')
plt.plot(sizes, heap_sort_values, label='Heap Sort', color='magenta')

# Настройки графика
plt.title('Time Complexity')
plt.xlabel('Input Size (n)')
plt.ylabel('Operations (T(n))')
plt.legend()
plt.grid(True)
plt.show()
