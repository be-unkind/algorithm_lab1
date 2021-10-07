
import time
from numpy.random import randint
import matplotlib.pyplot as plt
import numpy as np

def selection_sort( array ):
    counter = 0
    n = len( array )
    for i in range( n - 1 ): 
        min_idx = i
        for j in range( i + 1, n ):
            if array[j] < array[min_idx] :
                min_idx = j
            counter += 1
        if min_idx != i :
            temp = array[i]
            array[i] = array[min_idx]
            array[min_idx] = temp

    return [array, counter]


def insertion_sort(array):
    counter = 0
    for step in range(1, len(array)):
        key = array[step]
        j = step - 1   
        counter += 1   
        while j >= 0 and key < array[j]:
            array[j + 1] = array[j]
            j = j - 1
        array[j + 1] = key
    return [array, counter]


def merge_sort(array):
    counter = 0
    if len(array) > 1:
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        merge_sort(L)
        merge_sort(M)

        i = j = k = 0

        while i < len(L) and j < len(M):
            counter += 1
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1
        counter += 1

        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1
            counter += 1
        counter += 1    

        while j < len(M):
            counter += 1
            array[k] = M[j]
            j += 1
            k += 1
        counter += 1

    return [array, counter]


def shell_sort(array):
    counter = 1
    n = len(array)
    interval = n // 2
    while interval > 0:
        for i in range(interval, n):
            temp = array[i]
            j = i
            counter += 1
            while j >= interval and array[j - interval] > temp:
                array[j] = array[j - interval]
                j -= interval
            array[j] = temp
        interval //= 2
    return [array, counter]


list_values = [7, 8, 9, 10, 11, 12, 13, 14, 15]

selection_count = []
insertion_count = []
merge_count = []
shell_count = []

selection_time = []
insertion_time = []
merge_time = []
shell_time = []

for value in list_values:

    array = randint(0, 100000, 2**value)

    # array = []
    # for val in range(0, 2**value):
    #     array.append(value)
    # array.reverse()

    # list_count = np.array([0, 100000])
    # list_x = [0, 7, 7, 9, 10, 11, 12, 13, 14, 15]

    # print("Selection sort")
    start_time = time.time()
    new_array = selection_sort(array)
    end_time = time.time() - start_time
    selection_time.append(end_time)
    # selection_count.append(int(new_array[1]))
    # print(end_time)
    # print(new_array[0])
    # print(new_array[1])

    # print("Insertion sort")
    start_time = time.time()
    new_array = insertion_sort(array)
    end_time = time.time() - start_time
    insertion_time.append(end_time)
    # insertion_count.append(int(new_array[1]))
    # print(end_time)
    # print(new_array[0])
    # print(new_array[1])

    # print("Merge sort")
    start_time = time.time()
    new_array = merge_sort(array)
    end_time = time.time() - start_time
    merge_time.append(end_time)
    # merge_count.append(int(new_array[1]))
    # print(end_time)
    # print(new_array[0])
    # print(new_array[1])

    # print("Shell sort")
    start_time = time.time()
    new_array = shell_sort(array)
    end_time = time.time() - start_time
    shell_time.append(end_time)
    # shell_count.append(int(new_array[1]))
    # print(end_time)
    # print(new_array[0])
    # print(new_array[1])

# print(selection_count)
# print(insertion_count)
# print(merge_count)
# print(shell_count)

plt.xlabel("Array size")
# plt.ylabel("Number of comparisons")
plt.ylabel("Running time")

plt.yscale('log',base=2)

# plt.plot(list_values, selection_count, marker='.', color='darkturquoise')
# plt.plot(list_values, insertion_count, marker='.', color='darkslateblue')
# plt.plot(list_values, merge_count, marker='.', color='wheat')
# plt.plot(list_values, shell_count, marker='.', color = 'orchid')

plt.plot(list_values, selection_time, marker='.', color='darkturquoise')
plt.plot(list_values, insertion_time, marker='.', color='darkslateblue')
plt.plot(list_values, merge_time, marker='.', color='wheat')
plt.plot(list_values, shell_time, marker='.', color = 'orchid')


plt.show()

