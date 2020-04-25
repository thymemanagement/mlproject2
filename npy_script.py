from preprocess import *

import numpy as np

from numpy import asarray
from numpy import save

# load all of the train labels and images into np arrays "images" and "labels"
csv_path = './Train_Labels.csv'
img_path = './Train_Image/Train_Image/'
images, labels = parse_training_all(csv_path, img_path)

# reformat Label objects into strings for saving into .npy files
string_labels = []
for i in range(labels.size):
  string_labels.append(labels[i].value)
labels = np.asarray(string_labels)

# save the data in .npy files if valid
train_images_path = './train_images.npy'
train_labels_path = './train_labels.npy'
save(train_images_path, images)
save(train_labels_path, labels)

x_train, y_train = load_np_data(train_images_path, train_labels_path)

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

# plot first image with its label to check if valid
plt.imshow(x_train[0])
print(y_train[0])
