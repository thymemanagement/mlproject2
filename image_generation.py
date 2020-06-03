from preprocess import *
from data_augmentation import *
  
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
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.50, random_state = 5)
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

# create a prediction array for test data set
prediction = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
print('prediction.shape =', prediction.shape)

highest = 0
highest_idx = 0
second = 0
confidence_threshold = 0.99
difference_threshold = 0.01
candidates = []
curr_img = 0
# for each prediction array
for predict in prediction:
  # if a single prediction value is higher than 99% && if there are no other predictions values higher than 1%
  # loop through the predict array and set the highest and second highest values
  highest = max(predict)
  highest_idx = np.argmax(predict)
  temp = np.delete(predict, highest_idx) 
  second = max(temp)
  if highest > confidence_threshold and second < difference_threshold:
    # append the image name to a list
    candidates.append(curr_img)
    print(curr_img, 'is a candidate with confidence =', highest, 'and second highest =', second)
  curr_img += 1

# candidates has all image file names ('0.png') that meet the thresholds for duplication
candidates = np.asarray(candidates)
print('Candidates.shape =', candidates.shape)

# if you can use this np array and create new .npy files that have more data that would be awesome

def candidate_gen(cands):
  i = 0
  while True:
    if i in cands:
      yield True
    else:
      yield False
    i += 1

csv_path = 'Train_Labels.csv'
img_path = 'Train_Image/Train_Image/'

num_augments = 5
images, labels = parse_training_selected(csv_path, img_path, candidate_gen(candidates))
aug_images, aug_labels = augment_data(images, labels, seq, num_augments)

augment_images_path = 'augment_images.npy'
augment_labels_path = 'augment_labels.npy'

save(augment_images_path, aug_images)
save(augment_labels_path, aug_labels)

