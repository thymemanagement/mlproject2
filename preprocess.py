import csv
from enum import Enum
from itertools import *

import cv2
import numpy as np

#class to identify labels. Label.X produces the X label as does Label("x"). Label("x") can be used to verify label input.
class Label(Enum):
    AIRPLANE = "airplane"
    AUTOMOBILE = "automobile"
    BIRD = "bird"
    CAT = "cat"
    DEER = "deer"
    DOG = "dog"
    FROG = "frog"
    HORSE = "horse"
    SHIP = "ship"
    TRUCK = "truck"

#Parses a csv file and produces two arrays that contain the image data and label_data provided by the csv.
#Since the csv file and the image data can be quite large, this function only parses data from a subset of the rows.
#csv_path: path of the csv file, img_path: path to the image directory
#start: integer containing first row to parse, end: integer containing last row to parse
def parse_file(csv_path, img_path, start, end):
    png_array = []
    y_array = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for i, row in zip(count(0), csv_reader):
            if start <= i and i < end:
                png_array.append(cv2.imread(img_path + row[0]))
                y_array.append(Label(row[1]))
    return (np.asarray(png_array), np.asarray(y_array))
