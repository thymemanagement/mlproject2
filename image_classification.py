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


def test_labels_equal(a, b):
    max_index_a = 0
    max_index_b = 0
    for i in range(10):
        if a[i] > a[max_index_a]:
            max_index_a = i
        if b[i] > b[max_index_b]:
            max_index_b = i
    if max_index_a == max_index_b:
        return 0.0
    else:
        if b[max_index_b] > 0.80:
            return 1.0
        else:
            return 0.0

def relabel(predicted_correct):
    max_index = 0
    for i in range(10):
        if predicted_correct[i] > predicted_correct[max_index]:
            max_index = i

    ret = []
    for i in range(10):
        if i == max_index:
            ret.append(1.0)
        else:
            ret.append(0.0)
    return np.asarray(ret)


def main():
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
    NUM_FILTERS = 32
    FILTER_SIZE = 3
    POOL_SIZE = 2
    EPOCHS = 2
    BATCH_SIZE = 64 #64 or 128
    CONSTANT = 64 

    # initialize a sequential model
    model = Sequential()
    # add convolution layers to this model
    model.add(Conv2D(NUM_FILTERS*2, FILTER_SIZE, activation='relu', input_shape=(INPUT_SHAPE)))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Conv2D(NUM_FILTERS*4, FILTER_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Conv2D(NUM_FILTERS*8, FILTER_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Flatten()) # convert 3D feature map into 1D feature vectors for input into fully connected
    # add fully connected layers
    model.add(Dense(CONSTANT*4, activation = 'relu'))
    # use softmax as we have multiple classifications
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model_save_path = "best_model"
    best_x = x_train
    best_y = y_train
    current_x = best_x.copy()
    current_y = best_y.copy()
    best_accuracy = 0.0
    while True:
        sgd = optimizers.SGD(lr=0.001, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        history = model.fit(
            current_x,
            current_y,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            verbose=2,
            validation_split=0.2)
        current_accuracy = history.history['val_accuracy'][-1]
        print("Current accuracy = " + str(current_accuracy))

        # If current data is worse than best, revery back to best
        if current_accuracy > best_accuracy:
            print("New best accuracy: " + str(current_accuracy) + " VS " + str(best_accuracy))
            best_accuracy = current_accuracy
            best_x = current_x.copy()
            best_y = current_y.copy()
            model.save(model_save_path)
        else:
            current_x = best_x.copy()
            current_y = best_y.copy()
            # Have to re-build the old model since it was better.
            model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            history = model.fit(
                current_x,
                current_y,
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=2,
                validation_split=0.2)
            

        # go through best training data, and re-label anything bad
        bad_training_indices = []
        num_printed = 0
        for i in range(len(current_x)):
            c_img = current_x[i]
            c_label = current_y[i]
            predicted_label = model.predict(np.asarray([c_img]))[0]
            predicted_str = label_encoder.inverse_transform([argmax([predicted_label])])
            given_str = label_encoder.inverse_transform([argmax([c_label])])
            if test_labels_equal(c_label, predicted_label) != 0.0:
                if num_printed < 5:
                    cv2.imshow('given ' + str(given_str) + ", predicted " + str(predicted_str),c_img)
                    cv2.waitKey(0)
                    num_printed += 1
                bad_training_indices.append((test_labels_equal(c_label, predicted_label), i))

        # From the list of bad training data, select a random set of 20 items to re-label.
        num_bad_items = len(bad_training_indices)
        bad_training_indices.sort()
        print("Num bad training items: " + str(num_bad_items))
        bad_training_indices = bad_training_indices[:20]

        # Re-train the selected data
        for (err, i) in bad_training_indices:
            c_img = current_x[i]
            predicted_label = model.predict(np.asarray([c_img]))[0]
            current_y[i] = relabel(predicted_label)




    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of epochs')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.show()


if (__name__ == '__main__'):
    main()