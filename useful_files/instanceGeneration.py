import os
import random

class ProblemInstance:
    def __init__(self, ninner_variable, j_variable, processing_times, empty_movement_times, next_tank_movement_times,
               m_variable, h_variable, c_variable):
        self.ninner_variable = ninner_variable
        self.j_variable = j_variable
        self.processing_times = processing_times
        self.empty_movement_times = empty_movement_times
        self.next_tank_movement_times = next_tank_movement_times
        self.m_variable = m_variable
        self.h_variable = h_variable
        self.c_variable = c_variable

    def write_to_datafile(self, datafile_location):
        # Create or change file
        f = open(datafile_location, "w")

        # Write the number of tanks
        f.write("Ninner = " + str(self.ninner_variable) + ";\n")

        # Write the number of max items in process
        f.write("J = " + str(self.j_variable) + ";\n")

        # Write min and max processing times for each tank
        min_times = []
        max_times = []
        for i in range(self.ninner_variable):
            min_times.append(self.processing_times[i][0])
            max_times.append(self.processing_times[i][1])

        f.write("tmin = " + str(min_times) + ";\n")
        f.write("tmax = " + str(max_times) + ";\n")

        # Write emptry movements matrix
        f.write("e = array2d(1..Tinner,0..Ninner,\n")
        string_to_write = ""
        for i in range(self.ninner_variable + 1):
            string_to_write += "    "
            if i == 0:
                string_to_write += "["
            else:
                string_to_write += " "
            string_to_write += "|"
            string_to_write += str(self.empty_movement_times[i])[1:-1]
            if i != self.ninner_variable:
                string_to_write += ",\n"
            else:
                string_to_write += "|]);\n"
            f.write(string_to_write)
            string_to_write = ""

        # Write time to move an item to the next tank
        f.write("f = array1d(0..Ninner,\n")
        f.write("     " + str(self.next_tank_movement_times) + ");\n")

        # Write: Multiplier, Hoists, Capacity
        f.write("Multiplier = " + str(self.m_variable) + ";\n")
        f.write("Hoists = " + str(self.h_variable) + ";\n")
        f.write("Capacity = " + str(self.c_variable) + ";")

        # Close writer
        f.close()

def generate_data(ninner_variable, j_variable, processing_time_type, empty_movements_type,
                  next_tank_movements_type, m_variable, h_variable, c_variable):

    # Calculate instances attributes
    processing_times = processing_times_options(processing_time_type, ninner_variable)
    empty_movement_times = empty_movements_options(empty_movements_type, ninner_variable)
    next_tank_movement_times = next_tank_movements_options(next_tank_movements_type, ninner_variable)

    return ProblemInstance(ninner_variable, j_variable, processing_times, empty_movement_times, next_tank_movement_times,
                           m_variable, h_variable, c_variable)

# Method currently not used
def generate_additional_data(data_file_name, m_variable, h_variable, c_variable):

    # Create or change file
    f = open(data_file_name, "w")

    # Write: Multiplier, Hoists, Capacity
    f.write("Multiplier = " + str(m_variable) + ";\n")
    f.write("Hoists = " + str(h_variable) + ";\n")
    f.write("Capacity = " + str(c_variable) + ";")

    # Close writer
    f.close()

def processing_times_options(option_number, number_of_tanks):

    if option_number == 1:
        return simple_processing_times(number_of_tanks)
    if option_number == "exp3_0":
        return exp3_processing_times(number_of_tanks, 0.0)
    if option_number == "exp3_1":
        return exp3_processing_times(number_of_tanks, 0.1)
    if option_number == "exp3_2":
        return exp3_processing_times(number_of_tanks, 0.2)
    if option_number == "exp3_4":
        return exp3_processing_times(number_of_tanks, 0.4)
    if option_number == "exp3_6":
        return exp3_processing_times(number_of_tanks, 0.6)
    if option_number == "exp3_8":
        return exp3_processing_times(number_of_tanks, 0.8)
    if option_number == "exp3_10":
        return exp3_processing_times(number_of_tanks, 1.0)

    return []
    # return {
    #     1: simple_processing_times(number_of_tanks),
    #     "exp3_0": exp3_processing_times(number_of_tanks, 0.0),
    #     "exp3_2": exp3_processing_times(number_of_tanks, 0.2),
    #     "exp3_4": exp3_processing_times(number_of_tanks, 0.4),
    #     "exp3_6": exp3_processing_times(number_of_tanks, 0.6),
    #     "exp3_8": exp3_processing_times(number_of_tanks, 0.8),
    #     "exp3_10": exp3_processing_times(number_of_tanks, 1.0)
    # }[option_number]

def empty_movements_options(option_number, number_of_tanks):

    if option_number == 1:
        # return simple_empty_movement_times(number_of_tanks)
        return symmetric_empty_movement_times(number_of_tanks)
    if option_number == "exp6_1":
        return exp6_empty_movement_times(number_of_tanks, 1)
    if option_number == "exp6_2":
        return exp6_empty_movement_times(number_of_tanks, 2)
    if option_number == "exp6_4":
        return exp6_empty_movement_times(number_of_tanks, 4)
    if option_number == "exp6_6":
        return exp6_empty_movement_times(number_of_tanks, 6)
    if option_number == "exp6_8":
        return exp6_empty_movement_times(number_of_tanks, 8)
    if option_number == "exp6_10":
        return exp6_empty_movement_times(number_of_tanks, 10)

    return []

    # return {
    #     1: simple_empty_movement_times(number_of_tanks)
    # }[option_number]

def next_tank_movements_options(option_number, number_of_tanks):

    if option_number == 1:
        return simple_next_tank_movement_times(number_of_tanks)
    if option_number == "exp7_4":
        return exp7_next_tank_movement_times(number_of_tanks, 4)
    if option_number == "exp7_9":
        return exp7_next_tank_movement_times(number_of_tanks, 9)
    if option_number == "exp7_14":
        return exp7_next_tank_movement_times(number_of_tanks, 14)
    if option_number == "exp7_19":
        return exp7_next_tank_movement_times(number_of_tanks, 19)
    if option_number == "exp7_24":
        return exp7_next_tank_movement_times(number_of_tanks, 24)
    if option_number == "exp8_0":
        return exp8_next_tank_movement_times(number_of_tanks, 0)
    if option_number == "exp8_1":
        return exp8_next_tank_movement_times(number_of_tanks, 1)
    if option_number == "exp8_2":
        return exp8_next_tank_movement_times(number_of_tanks, 2)
    if option_number == "exp8_5":
        return exp8_next_tank_movement_times(number_of_tanks, 5)
    if option_number == "exp8_10":
        return exp8_next_tank_movement_times(number_of_tanks, 10)
    if option_number == "exp8_15":
        return exp8_next_tank_movement_times(number_of_tanks, 15)

    return []
    # return {
    #     1: simple_next_tank_movement_times(number_of_tanks)
    # }[option_number]

def simple_processing_times(number_of_tanks):
    list_of_min_max_times = []

    for i in range(number_of_tanks):
        min_time = random.randint(30, 35)
        max_time = random.randint(55, 100)
        list_of_min_max_times.append((min_time, max_time))
    return list_of_min_max_times

def exp3_processing_times(number_of_tanks, relative_window):
    list_of_min_max_times = []

    for i in range(number_of_tanks):
        average_processing_time = random.randint(60, 70)
        min_time = average_processing_time - int((0.5 * relative_window * 65))
        max_time = average_processing_time + int((0.5 * relative_window * 65))
        list_of_min_max_times.append((min_time, max_time))

    # print("Test relative window:", relative_window, ", Test generated windows:", list_of_min_max_times)
    return list_of_min_max_times

def simple_empty_movement_times(number_of_tanks):
    movement_times = [[1 for i in range(number_of_tanks+1)] for j in range(number_of_tanks+1)]

    for i in range(number_of_tanks+1):
        for j in range(number_of_tanks+1):
            if i+1 == j:
                movement_times[i][j] = 0
            else:
                movement_times[i][j] = random.randint(3, 10)

    return movement_times

def symmetric_empty_movement_times(number_of_tanks):
    movement_times = [[1 for i in range(number_of_tanks + 1)] for j in range(number_of_tanks + 1)]

    for i in range(number_of_tanks+1):
        for j in range(number_of_tanks+1):
            if i+1 == j:
                movement_times[i][j] = 0
            elif i >= j:
                movement_times[i][j] = random.randint(3, 10)

    for i in range(number_of_tanks+1):
        for j in range(number_of_tanks+1):
            if i + 1 < j:
                movement_times[i][j] = movement_times[j-1][i+1]

    return movement_times

def exp6_empty_movement_times(number_of_tanks, upper_bound):
    movement_times = [[1 for i in range(number_of_tanks+1)] for j in range(number_of_tanks+1)]

    for i in range(number_of_tanks+1):
        for j in range(number_of_tanks+1):
            if i+1 == j:
                movement_times[i][j] = 0
            # else:
            #     movement_times[i][j] = random.randint(1, upper_bound)
            elif i >= j:
                movement_times[i][j] = random.randint(1, upper_bound)

    # Comment out if non-changed required
    for i in range(number_of_tanks+1):
        for j in range(number_of_tanks+1):
            if i + 1 < j:
                movement_times[i][j] = movement_times[j-1][i+1]

    return movement_times

def simple_next_tank_movement_times(number_of_tanks):
    movement_times = [1 for i in range(number_of_tanks+1)]

    for i in range(number_of_tanks+1):
        movement_times[i] = random.randint(11,29)

    return movement_times

def exp7_next_tank_movement_times(number_of_tanks, lower_bound):
    movement_times = [1 for i in range(number_of_tanks + 1)]

    for i in range(number_of_tanks + 1):
        movement_times[i] = random.randint(lower_bound, lower_bound+5)

    return movement_times

def exp8_next_tank_movement_times(number_of_tanks, variability):
    movement_times = [1 for i in range(number_of_tanks + 1)]

    for i in range(number_of_tanks + 1):
        movement_times[i] = random.randint(19 - variability, 19 + variability)

    return movement_times