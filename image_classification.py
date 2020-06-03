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

# define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 

model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

# setup parameters for compiling the model
# use early stopping to stop the training if valudation accuracy does not change after 10 epochs
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, verbose=1)
model.compile(loss='categorical_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
history = model.fit(
    x_train,
    y_train,
    epochs=100, 
    batch_size=128, 
    callbacks=[callback],
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

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss') 
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

# load test images
test_images_path = './test_images.npy'

test_images = load(test_images_path, allow_pickle=True)

print('test_images.shape =', test_images.shape)
print('test_images[0].shape =', test_images[0].shape)

# plot first image with its label to check if valid
plt.imshow(test_images[0]);

# create predictions for test images
prediction = model.predict(test_images, batch_size=BATCH_SIZE, verbose=1)
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
