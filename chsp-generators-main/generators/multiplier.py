import numpy as np

a = [0, 0, 143, 133, 125, 111, 100, 90, 80, 64, 52, 41, 30, 0, 143, 0, 10, 18, 32, 43, 53, 64, 79, 91, 102, 12, 0, 133,    10, 0, 8, 22, 33, 43, 53, 69, 81, 92, 11, 0, 125, 18, 8, 0, 14, 25, 35, 45, 61, 73, 84, 10, 0, 111, 32, 22, 14, 0,    11, 21, 31, 47, 59, 70, 88, 0, 100, 43, 33, 25, 11, 0, 10, 20, 36, 48, 59, 77, 0, 90, 53, 43, 35, 21, 10, 0, 10,    26, 38, 49, 67, 0, 80, 64, 53, 45, 31, 20, 11, 0, 15, 28, 39, 57, 0, 64, 79, 69, 61, 47, 36, 26, 15, 0, 12, 23, 41,    0, 52, 91, 81, 73, 59, 48, 38, 28, 12, 0, 11, 29, 0, 41, 102, 92, 84, 70, 59, 49, 39, 23, 11, 0, 18, 0, 30, 120,    110, 102, 88, 77, 67, 57, 41, 29, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

a = np.array(a)
a.resize(int(np.sqrt(a.size)), int(np.sqrt(a.size)))

a = np.roll(a, shift=1, axis=0)
# print(a)

a12 = np.copy(a[1:, 1:]) + 5
a3 = np.copy(a[1:, 1:])

a_new = np.zeros(shape=(int(np.sqrt(a.size) - 1) * 2 + 1, int(np.sqrt(a.size) - 1) * 2 + 1), dtype=np.int64)

a_new[:a.shape[0], :a.shape[0]] = a
a_new[a.shape[0]:, 1:a.shape[0]] = a12
a_new[1:a.shape[0], a.shape[0]:] = a12
a_new[a.shape[0]:, a.shape[0]:] = a3

# still need to take first row to last row
a_new = np.roll(a_new, shift=-1, axis=0)

a_list = [str(list(x)) for x in list(a_new)]
a_print = "\n".join(a_list)
print(a_print)







