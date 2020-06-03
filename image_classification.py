from preprocess import *
  
import math
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

# a is the given label, b is predicted
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
        if b[max_index_b] > 0.60:
            return 1.0 - b[max_index_a]
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


# If a labels score is predicted correct, but we want to try changing it anyway,
# this function will return a random "next best guess" of sorts
def relabel_random(predicted_correct):
    scores = []
    max_index = 0
    for i in range(10):
        if predicted_correct[i] > predicted_correct[max_index]:
            max_index = i
        scores.append((predicted_correct[i], i))

    # Find a random index to try, ideally from the top few best
    relabel_idx = 0
    while relabel_idx == max_index: # Don't guess the same label were already given!
        (score, idx) = scores[random.randrange(0, 3)]
        relabel_idx = idx

    ret = []
    for i in range(10):
        if i == relabel_idx:
            ret.append(1.0)
        else:
            ret.append(0.0)
    return np.asarray(ret)

def relabel_round(given_labels, pictures, best_model, iters_since_last_improvement):
     # don't trust prediction entirely, gives some randomness
    if iters_since_last_improvement < 3:
        salt_ratio = 0.00
    elif iters_since_last_improvement < 10:
        salt_ratio = 0.05
    elif iters_since_last_improvement < 20:
        salt_ratio = 0.25
    elif iters_since_last_improvement < 30:
        salt_ratio = 0.5
    else:
        salt_ratio = 0.75

    out_labels = given_labels.copy()
    n_labels = len(out_labels)
    n_incorrect = int(n_labels * 0.2) # ~20% of labels are incorrect?

    bad_label_indices = [] # (how_rong_given_label_is, index)
    for i in range(n_labels):
        c_img = pictures[i]
        c_label = given_labels[i]
        predicted_label = best_model.predict(np.asarray([c_img]))[0]
        score = test_labels_equal(c_label, predicted_label)
        if score != 0.0:
            bad_label_indices.append((score, i))
    bad_label_indices.sort()
    bad_label_indices.reverse()

    # How many of the bad_labels will we relabel?
    if len(bad_label_indices) > n_incorrect:
        n_to_relabel = int(math.floor(n_incorrect * (1.0 - salt_ratio)))
    else:
        n_to_relabel = int(len(bad_label_indices))

    # How many random "so called correct" labels will we relabel?
    n_to_salt = n_incorrect - n_to_relabel

    print("Num relabel = " + str(n_to_relabel))
    print("Num salt = " + str(n_to_salt))

    # relabel the bad scored data
    for i in range(n_to_relabel):
        (score, relabel_idx) = bad_label_indices[i]
        out_labels[relabel_idx] = relabel(out_labels[relabel_idx])

    # re-label some random labels
    while n_to_salt > 0:
        rand_idx = random.randrange(0, n_labels)
        # Check if we already relabeld rand_idx
        index_alrealy_relabeld = False
        for i in range(n_to_relabel):
            (score, idx) = bad_label_indices[i]
            if rand_idx == idx:
                index_alrealy_relabeld = True
                break
        if not index_alrealy_relabeld:
            out_labels[rand_idx] = relabel_random(out_labels[rand_idx])
            n_to_salt -= 1

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
    EPOCHS = 15
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

    current_y = y_train.copy()
    best_accuracy = 0.0
    iters_since_last_improvement = 0
    while True:
        sgd = optimizers.SGD(lr=0.001, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        history = model.fit(
            x_train,
            current_y,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            verbose=2,
            validation_data=(x_test, y_test))
        current_accuracy = history.history['val_accuracy'][-1]
        print("Current accuracy = " + str(current_accuracy))

        # If current data is worse than best, revery back to best
        if current_accuracy > best_accuracy:
            print("New best accuracy: " + str(current_accuracy) + " VS " + str(best_accuracy))
            best_accuracy = current_accuracy
            iters_since_last_improvement += 1
        else:
            current_y = y_train.copy()
            iters_since_last_improvement = 0
            # Have to re-build the old model since it was better.
            model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            history = model.fit(
                x_train,
                current_y,
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=2,
                validation_data=(x_test, y_test))
        
        # Re-label step
        relabel_round(current_y, x_train, model, iters_since_last_improvement)





    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of epochs')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.show()


if (__name__ == '__main__'):
    main()