import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load

aug_images_path = 'augment_images.npy'
aug_labels_path = 'augment_labels.npy'

aug_x_train = load(aug_images_path, allow_pickle=True)
aug_y_train = load(aug_labels_path, allow_pickle=True)

print('aug_x_train.shape =', aug_x_train.shape)
print('aug_y_train.shape =', aug_y_train.shape)

# plot first image with its label to check if valid
plt.imshow(aug_x_train[0]);
print(aug_y_train[0]);
