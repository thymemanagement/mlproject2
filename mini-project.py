import numpy as np
import csv
import re

parser = r'\[\s*(\S*)\s+(\S*)\s*\]'
project = "./mini-project/"
train_20 = "noise_train_data(20_percent_noise).csv"

def parse_data(data):
    match = re.match(parser, data)
    return (float(match[1]), float(match[2]))

def parse_file(path):
    x_array = []
    y_array = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            data = parse_data(row[0])
            x_array.append(data)
            y_array.append(row[1])
    return (np.asarray(x_array), np.asarray(y_array))

x_train, y_train = parse_file(project + train_20)
print("x_train and y_train generated successfully")
