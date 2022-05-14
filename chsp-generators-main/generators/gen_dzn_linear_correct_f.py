import json


def emptyTime(emptys, i, j):
    sum = 0
    start = min(i, j)
    stop = max(i, j)
    for k in range(start, stop):
        sum += emptys[k]
    return sum


def write_dzn(filename, num_tanks, tmin, tmax, f, emptys):
    with open(filename, 'w') as file:
        file.write("Ninner = %d;\n" % num_tanks)
        file.write("tmin = %s;\n" % tmin)
        file.write("tmax = %s;\n" % tmax)
        file.write("f = array1d(0..Ninner, %s);\n" % f)
        file.write("e = array2d(1..Tinner,0..Ninner, [\n")
        for i in range(1, num_tanks + 1):
            for j in range(0, num_tanks + 1):
                file.write(str(emptyTime(emptys, i, j)))
                if j < num_tanks:
                    file.write(",")
            if i < num_tanks + 1:
                file.write(",\n")

        # Write the first row as the last one (see Wallace instructions)
        i = 0
        for j in range(0, num_tanks + 1):
            file.write(str(emptyTime(emptys, i, j)))
            if j < num_tanks:
                file.write(",")
        file.write("\n]);\n")
        # Only for calculating statistics after the solution - not required by the model
        file.write("emptys = " + str(emptys))
        file.close()


# The upper bound of the range depends on the number of generated linear dzn files
for i in range(1, 11089):
    f = open("../instances/linear/" + str(i) + ".dzn", "r")
    d = {}
    for x in f:
        words = x.split('=')
        if len(words) == 1:
            continue
        else:
            attribute = words[0][:-1]
            value = words[1][1:]
            if attribute == 'Ninner':
                d['Ninner'] = int(value[:-2])
            elif attribute == 'tmin':
                d['tmin'] = json.loads(value[:-2])
            elif attribute == 'tmax':
                d['tmax'] = json.loads(value[:-2])
            elif attribute == 'f':
                d['f'] = json.loads(value.split('Ninner')[1][2:-3])
            elif attribute == 'emptys':
                d['emptys'] = json.loads(value)
            else:
                continue
    f.close()
    updated_value = d['f']
    updated_value[-1] = sum(d['emptys'][:-1], updated_value[-1])
    d['f'] = updated_value
    write_dzn('../instances/linear_correct_f/' + str(i) + '.dzn', num_tanks=d['Ninner'], tmin=d['tmin'], tmax=d['tmax'], f=d['f'], emptys=d['emptys'])
