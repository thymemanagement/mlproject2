import numpy as np
import csv
import re

parser = '\[\s*(\S*)\s+(\S*)\s*\]'

def parse_data(str):
    regex = re.compile(parser)
    match = regex.match(str)
    return (float(match[1]), float(match[2]))

project = "./mini-project/"
train_20 = "noise_train_data(20_percent_noise).csv"
x_train = []
y_train = []
with open(project + train_20) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            line_count += 1
            x_train.append(parse_data(row[0]))
            y_train.append(row[1])
            x1 = parse_data(row[0])[0]
            x2 = parse_data(row[0])[1]
            print('\tx = [' + str(x1) + ', ' + str(x2) + '];\ty = ' + str(row[1]))

    print(f'Processed {line_count} lines.')
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
