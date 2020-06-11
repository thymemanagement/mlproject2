# load the created model and output a csv file for kaggle submission
from preprocess import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import models
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv2D
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import asarray
from numpy import save
from keras.models import load_model

# load the model
model = load_model('cnn(+aug).h5')
model.summary()

# load test images
test_images_path = './test_images.npy'

test_images = load(test_images_path, allow_pickle=True)

print('test_images.shape =', test_images.shape)
print('test_images[0].shape =', test_images[0].shape)

# plot first image with its label to check if valid
plt.imshow(test_images[0]);

train_images_path = './train_images.npy'
train_labels_path = './train_labels.npy'
x_train, y_train = load_np_data(train_images_path, train_labels_path)

# append augmented data to existing x_train and y_train
aug_images_path = 'augment_images.npy'
aug_labels_path = 'augment_labels.npy'

print('Before adding aug images...')
print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

aug_x_train, aug_y_train = load_np_data(aug_images_path, aug_labels_path)
print('aug_y_train[0]=', aug_y_train[0])
# convert Label.ANIMAL values to string
print('Converting animal labels to strings')
for i in range(aug_y_train.size):
   aug_y_train[i] = aug_y_train[i].value
print('aug_y_train[0]=', aug_y_train[0])

print('aug_x_train.shape =', aug_x_train.shape)
print('aug_y_train.shape =', aug_y_train.shape)

x_train = np.append(x_train, aug_x_train, axis=0)
y_train = np.append(y_train, aug_y_train, axis=0)

print('After adding aug images...')
print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)


# one hot encoding example from https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# need to convert labels from strings to integers and integers to "binary class matrices"
# define example
data = y_train
values = array(data)
print('values =', values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print('integer_encoded =', integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print('onehot_encoded =', onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print('inverted =',inverted)
y_train = onehot_encoded
print('y_train.shape =', y_train.shape)


# create predictions for test images
prediction = model.predict(test_images, batch_size=128, verbose=1)
predicted = []
files = []
for i in range(1000):
  temp = label_encoder.inverse_transform([argmax(prediction[i, :])])
  predicted.append(temp[0])
  files.append(str(i) + '.png')

print(files[0])
print(predicted[0])

# create submission.csv file using these predicted labels
field_names = ['Image File Name', 'Test Label']
submission = zip(files, predicted)
with open('./submission.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(field_names)
    writer.writerows(submission)

