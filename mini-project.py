import numpy as np
import csv
import re

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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



# initialize training and test data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.20, random_state = 4)
print('Shape of x_train:', x_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of x_test:', x_test.shape)
print('Shape of y_test:', y_test.shape)

# define parameters
INPUT_DIM = len(x_train[0])
EPOCHS = 50
BATCH_SIZE = 4
ACTIVATION = 'sigmoid'

# create model
model = Sequential()
model.add(Dense(INPUT_DIM*20, activation = ACTIVATION, input_dim = INPUT_DIM))
model.add(Dense(INPUT_DIM*40, activation = ACTIVATION))
model.add(Dense(INPUT_DIM*80, activation = ACTIVATION))
model.add(Dense(INPUT_DIM*160, activation = ACTIVATION))
model.add(Dense(1, activation=ACTIVATION))

model.summary()

# actually train the data
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(
    x_train, 
    y_train,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=2,
    validation_data=(x_test, y_test))

#train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc= model.evaluate(x_test,y_test)
#print('Train set accuracy:', train_acc)
#print('Train set loss:', train_loss)
print('Test set accuracy:', test_acc)
print('Test set loss:', test_loss)

# analyze the results
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
