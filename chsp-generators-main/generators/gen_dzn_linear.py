import itertools
import random
from debugpy import debug_this_thread

import numpy as np


def emptyTime(emptys, i, j):
    sum = 0
    start = min(i, j)
    stop = max(i, j)
    for k in range(start, stop):
        sum += emptys[k]
    return sum

def save_problem_to_file(filename, tanks, lowers, uppers, fulls, e, emptys):
  with open(filename, 'w') as f:
    f.write("Ninner = %d;\n" % tanks)
    # f.write("J = 9;\n")
    f.write("tmin = %s;\n" % lowers)
    f.write("tmax = %s;\n" % uppers)
    f.write("f = array1d(0..Ninner, %s);\n" % fulls)
    f.write("e = array2d(1..Tinner,0..Ninner, [\n")
    arr_str = np.array2string(e, separator=',')
    arr_str = arr_str.replace('[', '').replace(']', '')
    f.write(arr_str)
    f.write("\n]);\n")
    # Only for calculating statistics after the solution - not required by the model
    f.write("% emptys = " + str(emptys))

def generate_problem(tanks, processingTimeBase, lowerVsUpperMean, timeMinVariability, emptyTimeRange, loadedTimeVariability, save_np=True):
    t = 0
    lowers = []
    uppers = []
    emptys = []
    fulls = []
    emptyToNextSmall = random.uniform(3, 7)
    emptyToNextMedium = random.uniform(3, 20)
    emptyToNextLarge = random.uniform(5, 50)
    while t <= tanks:
        # define the base of processing time
        lowerInitial = random.uniform(processingTimeBase * 0.5, processingTimeBase * 1.5)

        # Define tank's tmin value
        # for small tmin variability
        if timeMinVariability == "small":
            lowerMax = lowerInitial * random.uniform(4, 10)
        # for large tmin variability
        else:
            # limit the lowerMax for large initial values
            if processingTimeBase < 100:
                lowerMax = lowerInitial * random.uniform(4, 250)
            else:
                lowerMax = lowerInitial * random.uniform(4, 25)
        # each tank's tmin value
        lower = random.uniform(lowerInitial, lowerMax)
        lowers.append(int(lower))

        # Define tank's tmax value
        # for large variability in processing times (tmin diverse from tmax)
        if lowerVsUpperMean == "diverse":
            if lower > 1000:
                upperInitial = random.uniform(lower * 1.01, lower * 3)
            else:
                upperInitial = random.uniform(lower * 1.1, lower * 10)
        # for small variability in processing times (tmin similar to tmax)
        else:
            upperInitial = random.uniform(lower * 1.01, lower * 1.5)
        # each tank's tmax value
        upperMax = upperInitial * random.uniform(1, 1.2)
        upper = random.uniform(upperInitial, upperMax)
        uppers.append(int(upper))

        # define the empty time
        if processingTimeBase < 100:
            if emptyTimeRange == "small":
                emptyToNext = random.uniform(3, 7)
            elif emptyTimeRange == "large":
                emptyToNext = random.uniform(3, 12)
            else:
                emptyToNext = emptyToNextSmall
        elif processingTimeBase < 450:
            if emptyTimeRange == "small":
                emptyToNext = random.uniform(3, 10)
            elif emptyTimeRange == "large":
                emptyToNext = random.uniform(3, 20)
            else:
                emptyToNext = emptyToNextMedium
        else:
            if emptyTimeRange == "small":
                emptyToNext = random.uniform(5, 20)
            elif emptyTimeRange == "large":
                emptyToNext = random.uniform(5, 50)
            else:
                emptyToNext = emptyToNextLarge
        emptys.append(int(emptyToNext))

        # define loaded time
        if processingTimeBase < 60:
            if loadedTimeVariability == "small":
                fullToNext = random.uniform(emptyToNext * 1.5, emptyToNext * 2)
            else:
                fullToNext = random.uniform(emptyToNext * 1.5, emptyToNext * 4)
        else:
            if loadedTimeVariability == "small":
                fullToNext = random.uniform(emptyToNext * 1.5, emptyToNext * 3)
            else:
                fullToNext = random.uniform(emptyToNext * 3, emptyToNext * 15)
        fulls.append(int(fullToNext))

        # Update Tank counter for the While
        t += 1

    del lowers[-1]
    del uppers[-1]

    n = tanks + 1

    def get_row(emptys, i):
        row = []
        for j in range(n):
            row.append(emptyTime(emptys, i, j))
        return row

    e = np.zeros((n, n)).astype(int)
    for i in range(1, n):
        e[i - 1] = get_row(emptys, i)
    # Write the first row as the last one (see Wallace instructions)
    e[-1] = get_row(emptys, 0)
    return {'Ninner': tanks, 'tmin': lowers, 'tmax': uppers, 'f': fulls, 'e': e, 'emptys': emptys}


def generateRandomProblems(save_np=False, seed=0, max=999999):
    random.seed(seed)
    debug=False
    processingTimeScale = [30, 60, 100, 150, 250, 450, 600]
    similarity = ["similar", "diverse"]
    variability = ["small", "large"]
    timeRange = ["fixed", "small", "large"]
    inst = 0
    instances = {}
    base_path = "instances/linear"
    for rounds, tanks, processingTimeBase, lowerVsUpperMean, timeMinVariability, emptyTimeRange, loadedTimeVariability in itertools.product(
            range(1, 4), range(3, 25), processingTimeScale, similarity, variability, timeRange, variability):
        inst += 1
        vars = tanks, processingTimeBase, lowerVsUpperMean, timeMinVariability, emptyTimeRange, loadedTimeVariability
        problem = generate_problem(*vars, save_np)
        if save_np:
            instances[str(vars)] = problem
            if inst % 10 == 1 and not debug:
                print(f"Saving with instances {inst}")
                np.save(base_path + f"problems_{seed}.npy", instances)
        elif not debug:
            filename = f"{base_path}/{inst}.dzn"
            save_problem_to_file(filename, *list(problem.values()))
        if inst >= max:
            break
    if save_np and not debug:
        print("save np")
        np.save(base_path + "problems.npy", instances)

if __name__ == "__main__":
    generateRandomProblems(save_np=True, seed=1)