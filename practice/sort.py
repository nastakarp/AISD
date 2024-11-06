arr = [10,9,8,7,6,5,4,3,2,1]
print(arr)
def selection_sort(arr):
    n = len(arr)

    # Проходим по всем элементам массива
    for i in range(n):
        # Считаем, что текущий элемент i — это минимальный
        min_idx = i

        # Ищем минимальный элемент в оставшейся части массива
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Меняем местами найденный минимальный элемент с первым элементом
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr

sorted_arr = selection_sort(arr)
print("selection_sort:", sorted_arr)

def insertion_sort(arr):
    # Проходим по всем элементам массива, начиная со второго
    for i in range(1, len(arr)):
        key = arr[i]
        # Сравниваем элемент с предыдущими и перемещаем его в нужное место
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr

sorted_arr = insertion_sort(arr)
print("insertion_sort:", sorted_arr)

def bubble_sort(arr):
    n = len(arr)
    # Проходим по массиву
    for i in range(n):
        # Последние i элементов уже на своих местах
        for j in range(0, n-i-1):
            # Меняем элементы местами, если они стоят в неправильном порядке
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

sorted_arr = bubble_sort(arr)
print("bubble_sort:", sorted_arr)


def merge(arr, left, mid, right):
    # Создаем временные массивы
    n1 = mid - left + 1
    n2 = right - mid

    # Временные массивы для левой и правой частей
    L = [0] * n1
    R = [0] * n2

    # Копируем данные в временные массивы L[] и R[]
    for i in range(0, n1):
        L[i] = arr[left + i]
    for j in range(0, n2):
        R[j] = arr[mid + 1 + j]

    # Инициализируем индексы для L[], R[] и основного массива
    i = 0  # Начальный индекс первого подмассива
    j = 0  # Начальный индекс второго подмассива
    k = left  # Начальный индекс объединенного подмассива

    # Сливаем временные массивы обратно в arr[left..right]
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Копируем оставшиеся элементы L[], если они есть
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Копируем оставшиеся элементы R[], если они есть
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

# Основная функция сортировки слиянием
def merge_sort(arr, left, right):
    if left < right:
        # Находим середину
        mid = (left + right) // 2

        # Рекурсивно сортируем первую и вторую половины
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)

        # Сливаем отсортированные половины
        merge(arr, left, mid, right)

merge_sort(arr, 0, len(arr) - 1)
print("merge_sort:", arr)

def shell_sort(arr):
    n = len(arr)
    gap = n // 2  # Инициализируем шаг

    # Уменьшаем шаг с каждым проходом
    while gap > 0:
        # Сортировка вставками с заданным шагом
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp

        # Уменьшаем шаг в два раза
        gap //= 2

    return arr

sorted_arr = shell_sort(arr)
print("shell_sort:", sorted_arr)

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

sorted_arr = shell_sort_hibbard(arr)
print("shell_sort_hibbard:", sorted_arr)

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

sorted_arr = shell_sort_pratt(arr)
print("shell_sort_pratt:", sorted_arr)


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        # Выбираем опорный элемент (например, последний элемент)
        pivot = arr[len(arr) // 2]

        # Массивы меньших, равных и больших элементов
        less = [x for x in arr if x < pivot]
        equal = [x for x in arr if x == pivot]
        greater = [x for x in arr if x > pivot]

        # Рекурсивная сортировка левой и правой части
        return quick_sort(less) + equal + quick_sort(greater)



sorted_arr = quick_sort(arr)
print("quick_sort:", sorted_arr)

# Функция для построения кучи
def heapify(arr, n, i):
    largest = i  # Инициализируем наибольший элемент как корень
    left = 2 * i + 1  # Левый дочерний элемент
    right = 2 * i + 2  # Правый дочерний элемент

    # Если левый дочерний элемент больше корня
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Если правый дочерний элемент больше текущего наибольшего
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Если наибольший элемент не корень
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Меняем местами

        # Рекурсивно перестраиваем поддерево
        heapify(arr, n, largest)

# Основная функция пирамидальной сортировки
def heap_sort(arr):
    n = len(arr)

    # Построение макс-кучи
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Извлечение элементов из кучи по одному
    for i in range(n - 1, 0, -1):
        # Меняем текущий корень с последним элементом
        arr[i], arr[0] = arr[0], arr[i]

        # Восстанавливаем кучу для оставшегося подмножества элементов
        heapify(arr, i, 0)

heap_sort(arr)
print("heap_sort: ", arr)
