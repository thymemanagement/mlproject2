from preprocess import *
from data_augmentation import *
  
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from numpy import asarray
from numpy import save

train_images_path = './train_images.npy'
train_labels_path = './train_labels.npy'
images, labels = load_np_data(train_images_path, train_labels_path)

num_augments = 5

print("augmenting " + str(len(labels)) + " images")
aug_images, aug_labels = augment_data(images, labels, seq, num_augments)
print("shape of aug_images =", aug_images.shape)

augment_images_path = 'augment_images.npy'
augment_labels_path = 'augment_labels.npy'

print("writing augmented images to " + augment_images_path)
save(augment_images_path, aug_images)
print("writing augmented labels to " + augment_labels_path)
save(augment_labels_path, aug_labels)

