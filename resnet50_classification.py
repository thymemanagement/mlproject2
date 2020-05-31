# reference: https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Network_Keras.ipynb

from preprocess import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, applications
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import asarray
from numpy import save

train_images_path = './train_images.npy'
train_labels_path = './train_labels.npy'
test_x_path = './test_images.npy'
test_x = np.load(test_x_path, allow_pickle=True)
x_train, y_train = load_np_data(train_images_path, train_labels_path)

# one hot encoding example from https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# need to convert labels from strings to integers and integers to "binary class matrices"
# define example
data = y_train
values = array(data)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
y_train = onehot_encoded

# initialize training and test data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.20, random_state = 4)

# define hyper parameters
EPOCHS = 50
BATCH_SIZE = 64

img_height, img_width = 32, 32
num_classes = 10
# If imagenet weights are being loaded, 
# input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(x_test, y_test)
)

# save model into a pickle file
with open('cnn.pickle', 'wb') as file:
    pickle.dump(model, file)

prediction = model.predict(test_x)
print(prediction)

# plot the results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()