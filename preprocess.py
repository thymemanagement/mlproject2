import csv
from enum import Enum
from itertools import *
import matplotlib.pyplot as plt
from numpy import asarray, save, load
import os
import random

import cv2
import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD
from numpy import load
=======
TEST_IMG_DIR = os.path.join('Test_Image', 'Test_Image')
>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf
=======
TEST_IMG_DIR = os.path.join('Test_Image', 'Test_Image')
>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf

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

#generates a random set of batches from a given number of batches and batch size.
#batches is a list where indexes represent rows in an input csv file and values represent the batch number of the row
def random_batches(num_batches, batch_size):
    batches = list(chain.from_iterable(repeat(n, batch_size) for n in range(0, num_batches)))
    random.shuffle(batches)
    return batches

#generalized parse training function. Uses a generator selectors to determine which rows to parse.
def parse_training_selected(csv_path, img_path, selectors):
    png_array = []
    y_array = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
<<<<<<< HEAD
<<<<<<< HEAD
        for row in compress(csv_reader,selectors):
            print("Processing file", row[0])
            png_array.append(cv2.imread(img_path + row[0]))
=======
        for row in compress(csv_reader, selectors):
            IMG_PATH = os.path.join(img_path, row[0])
            png_array.append(cv2.imread(IMG_PATH))
>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf
=======
        for row in compress(csv_reader, selectors):
            IMG_PATH = os.path.join(img_path, row[0])
            png_array.append(cv2.imread(IMG_PATH))
>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf
            y_array.append(Label(row[1]))
    return (np.asarray(png_array), np.asarray(y_array))

#Parses a given csv file and produces two parallel lists that contain image data and label data provided by the csv.
#The data parsed is given by batch_num and batches
#batches is a list where each index represents a row in the csv and each index has a number which is the rows batch number
#batch_num is the batch number currently being parsed. If an index in batches has this batch_number, it will be included in the output
def parse_training_batch(csv_path, img_path, batch_num, batches):
    return parse_training_selected(csv_path, img_path, map(lambda x : x == batch_num, batches))

#Parses a csv file and produces two lists that contain image data and label data provided by the csv.
#Since the csv file and the image data can be quite large, this function only parses data from a subset of the rows.
#csv_path: path of the csv file, img_path: path to the image directory
#start: integer containing first row to parse, end: integer containing last row to parse
def parse_training_range(csv_path, img_path, start, end):
    png_array = []
    y_array = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in islice(csv_reader, start, end):
            IMG_PATH = os.path.join(img_path, row[0])
            png_array.append(cv2.imread(IMG_PATH))
            y_array.append(Label(row[1]))
    return (np.asarray(png_array), np.asarray(y_array))

#Parses the entire given csv file and image path into two parallel lists
#Do not recommend using. Will probably take up too much space and take a long time
def parse_training_all(csv_path, img_path):
    return parse_training_selected(csv_path, img_path, repeat(True))

<<<<<<< HEAD
<<<<<<< HEAD
def load_np_data(image_path, labels_path):
    x_train = load(image_path)
    y_train = load(labels_path)
    return (load(image_path), load(labels_path))

=======
def main():
    # load all of the train labels and images into np arrays "images" and "labels"
    csv_path = 'Train_Labels.csv'
    img_path = TEST_IMG_DIR
    images, labels = parse_training_all(csv_path, img_path)

    # reformat Label objects into strings for saving into .npy files
    string_labels = []
    for i in range(labels.size):
        string_labels.append(labels[i].value)
    
    labels = np.asarray(string_labels)

    # save the data in .npy files if valid
    train_images_path = 'train_images.npy'
    train_labels_path = 'train_labels.npy'
    # save(train_images_path, images)
    # save(train_labels_path, labels)

    # load it back in to test
    x_train = load(train_images_path)
    y_train = load(train_labels_path)

=======
def main():
    # load all of the train labels and images into np arrays "images" and "labels"
    csv_path = 'Train_Labels.csv'
    img_path = TEST_IMG_DIR
    images, labels = parse_training_all(csv_path, img_path)

    # reformat Label objects into strings for saving into .npy files
    string_labels = []
    for i in range(labels.size):
        string_labels.append(labels[i].value)
    
    labels = np.asarray(string_labels)

    # save the data in .npy files if valid
    train_images_path = 'train_images.npy'
    train_labels_path = 'train_labels.npy'
    # save(train_images_path, images)
    # save(train_labels_path, labels)

    # load it back in to test
    x_train = load(train_images_path)
    y_train = load(train_labels_path)

>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf
    # plot first image with its label to check if valid
    plt.imshow(x_train[0])
    
if (__name__ == '__main__'):
<<<<<<< HEAD
    main()
>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf
=======
    main()
>>>>>>> 690c98a601b041bff6c2c6b5d74b432fdff319bf
