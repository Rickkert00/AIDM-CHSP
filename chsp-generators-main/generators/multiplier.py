import numpy as np

old_e = [0, 0, 143, 133, 125, 111, 100, 90, 80, 64, 52, 41, 30, 0, 143, 0, 10, 18, 32, 43, 53, 64, 79, 91, 102, 12, 0, 133, 10, 0, 8, 22, 33, 43, 53, 69, 81, 92, 11, 0, 125, 18, 8, 0, 14, 25, 35, 45, 61, 73, 84, 10, 0, 111, 32, 22, 14, 0, 11, 21, 31, 47, 59, 70, 88, 0, 100, 43, 33, 25, 11, 0, 10, 20, 36, 48, 59, 77, 0, 90, 53, 43, 35, 21, 10, 0, 10, 26, 38, 49, 67, 0, 80, 64, 53, 45, 31, 20, 11, 0, 15, 28, 39, 57, 0, 64, 79, 69, 61, 47, 36, 26, 15, 0, 12, 23, 41, 0, 52, 91, 81, 73, 59, 48, 38, 28, 12, 0, 11, 29, 0, 41, 102, 92, 84, 70, 59, 49, 39, 23, 11, 0, 18, 0, 30, 120, 110, 102, 88, 77, 67, 57, 41, 29, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

old_e = np.array(old_e)
old_e.resize(int(np.sqrt(old_e.size)), int(np.sqrt(old_e.size)))

old_e = np.roll(old_e, shift=1, axis=0)

# new_e12 is the part of the new matrix which is top right and bottom left
new_e12 = np.copy(old_e[1:, 1:]) + 5
# new_e3 is the part of the new matrix which is bottom right (this is an exact copy of the original e without the 0's in the first row and column)
new_e3 = np.copy(old_e[1:, 1:])

# e is of size (Ninner + 1)^2 so we make a new one corresponding to the new size
e_new = np.zeros(shape=(int(np.sqrt(old_e.size) - 1) * 2 + 1, int(np.sqrt(old_e.size) - 1) * 2 + 1), dtype=np.int64)

# top left (including zero row/column)
e_new[:old_e.shape[0], :old_e.shape[0]] = old_e
# bottom left and top right (exlcuding the zero row/column as these are only for row=0 and column=0)
e_new[old_e.shape[0]:, 1:old_e.shape[0]] = new_e12
e_new[1:old_e.shape[0], old_e.shape[0]:] = new_e12
# bottom right (exlcuding the zero row/column as these are only for row=0 and column=0)
e_new[old_e.shape[0]:, old_e.shape[0]:] = new_e3

# In the proper representation what we would interpret as the first row need to be shifted to the last row
e_new = np.roll(e_new, shift=-1, axis=0)

# Printing in such a way that inputting it in JSON is easy
a_list = [str(list(x)) for x in list(e_new)]
a_print = "\n".join(a_list)
print(f'[{a_print}]')







