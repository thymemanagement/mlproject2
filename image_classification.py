from preprocess import *
  
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import models
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
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
from classification_models.keras import Classifiers

train_images_path = './train_images.npy'
train_labels_path = './train_labels.npy'
x_train, y_train = load_np_data(train_images_path, train_labels_path)

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

# initialize training and test data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.20, random_state = 4)
print('Shape of x_train:', x_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of x_test:', x_test.shape)
print('Shape of y_test:', y_test.shape)

# define parameters
INPUT_SHAPE = x_train[0].shape
print(INPUT_SHAPE)

# define hyper parameters
NUM_FILTERS = 8
FILTER_SIZE = 3
POOL_SIZE = 2
EPOCHS = 100
BATCH_SIZE = 32
CONSTANT = 64 

## initialize a sequential model
#model = Sequential()
## add convolution layers to this model
#model.add(Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SHAPE)))
##model.add(keras.layers.Dropout(0.01, noise_shape=None, seed=None))
#model.add(MaxPooling2D(pool_size=POOL_SIZE))
#model.add(Conv2D(32, (3, 3), activation='relu', activity_regularizer=keras.regularizers.l1(0.1)))
#model.add(MaxPooling2D(pool_size=POOL_SIZE))
#model.add(Conv2D(64, (3, 3), activation='relu', activity_regularizer=keras.regularizers.l1(0.1)))
#model.add(MaxPooling2D(pool_size=POOL_SIZE))
#model.add(Flatten()) # convert 3D feature map into 1D feature vectors for input into fully connected
## add fully connected layers
##model.add(Dense(CONSTANT*2, activation = 'relu'))
##model.add(Dense(CONSTANT//2, activation = 'sigmoid'))
## use softmax as we have multiple classifications
#model.add(Dense(10, activation='softmax'))
#model.summary

n_classes = 10
X = x_train
ResNet34, preprocess_input = Classifiers.get('resnet34')

# build model
base_model = ResNet34(input_shape=(32,32,3), weights='imagenet', include_top=False)
x = layers.GlobalAveragePooling2D()(base_model.output)
output = layers.Dense(n_classes, activation='softmax')(x)
model = models.Model(inputs=[base_model.input], outputs=[output])

# train
#model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X, y)

model.compile(loss='categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=2,
    validation_data=(x_test, y_test))

# plot the results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()


# load test images
test_images_path = './train_labels.npy'
test_images = load(test_images_path, allow_pickle=True)

print('test_images.shape =', test_images.shape)
print('test_images[0].shape =', test_images[0].shape)

prediction = model.predict(test_images, batch_size=BATCH_SIZE, verbose=1)
predicted = []
files = []
for i in range(1000):
  temp = label_encoder.inverse_transform([argmax(prediction[i, :])])
  predicted.append(temp[0])
  files.append(str(i) + '.png')

print(files[0])
print(predicted[0])

field_names = ['Image File Name', 'Test Label']
submission = zip(files, predicted)
with open('./submission.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(field_names)
    writer.writerows(submission)

