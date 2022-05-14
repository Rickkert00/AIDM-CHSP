import random


def emptyTime(sumEmptys, emptys, i, j):
    if i == j:
        totalTime = 0
    elif i == 0 or j == 0:
        totalTime = sumEmptys
        start = min(i, j)
        stop = max(i, j)
        for k in range(start, stop):
            totalTime -= emptys[k]
    else:
        totalTime = 0
        start = min(i, j)
        stop = max(i, j)
        for k in range(start, stop):
            totalTime += emptys[k]
    return totalTime


def generateRandomProblem():
    processingTimeScale = [30, 60, 100, 150, 250, 450, 600]
    similarity = ["similar", "diverse"]
    variability = ["small", "large"]
    timeRange = ["fixed", "small", "large"]
    inst = 0
    for rounds in range(1, 4):
        for tanks in range(3, 25):
            for processingTimeBase in processingTimeScale:
                for lowerVsUpperMean in similarity:
                    for timeMinVariability in variability:
                        for emptyTimeRange in timeRange:
                            for loadedTimeVariability in variability:
                                t = 0
                                lowers = []
                                uppers = []
                                emptys = []
                                fulls = []
                                sumEmptys = 0
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
                                    sumEmptys = sumEmptys + int(emptyToNext)

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

                                    if t == 0:
                                        fulls.append(int(fullToNext)*tanks)
                                    else:
                                        fulls.append(int(fullToNext))

                                    # Update Tank counter for the While
                                    t += 1

                                del lowers[-1]
                                del uppers[-1]

                                # output dzn
                                # Update instance name counter
                                inst += 1
                                filename = "../instances/reverse/" + str(inst)
                                filename += "."
                                filename += "dzn"
                                with open(filename, 'w') as f:
                                    f.write("Ninner = %d;\n" % tanks)
                                    # f.write("J = 9;\n")
                                    f.write("tmin = %s;\n" % lowers)
                                    f.write("tmax = %s;\n" % uppers)
                                    f.write("f = array1d(0..Ninner, %s);\n" % fulls)
                                    f.write("e = array2d(1..Tinner,0..Ninner, [\n")
                                    for i in range(1, tanks + 1):
                                        for j in range(0, tanks + 1):
                                            f.write(str(emptyTime(sumEmptys, emptys, i, j)))
                                            if j < tanks:
                                                f.write(",")
                                        if i < tanks + 1:
                                            f.write(",\n")

                                    # Write the first row as the last one (see Wallace instructions)
                                    i = 0
                                    for j in range(0, tanks + 1):
                                        f.write(str(emptyTime(sumEmptys, emptys, i, j)))
                                        if j < tanks:
                                            f.write(",")
                                    f.write("\n]);\n")
                                    # Only for calculating statistics after the solution - not required by the model
                                    f.write("emptys = " + str(emptys))


# Main code
def main():
    random.seed()
    generateRandomProblem()


if __name__ == "__main__":
    main()
