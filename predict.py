import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from preprocess import load_np_data

train_images_path = './train_images.npy'
train_labels_path = './train_labels.npy'
x_train, y_train = load_np_data(train_images_path, train_labels_path)

# one hot encoding example from https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# need to convert labels from strings to integers and integers to "binary class matrices"
# define example
data = y_train
values = np.array(data)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# load test images
test_x_path = './test_images.npy'
test_x = np.load(test_x_path, allow_pickle=True)

# save model into a pickle file
with open('cnn.pickle', 'rb') as file:
	history = pickle.load(file)

prediction = history.predict(test_x)

predict_labels = []
for i in range(1000):
	print(i)
	label = label_encoder.inverse_transform([np.argmax(prediction[i])])
	predict_labels.append(label)

print(predict_labels)