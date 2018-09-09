import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
hdf5_path = 'DrivFace.h5'
num_classes = 3

def load_dataset():
    print("Reading h5py file..")
    with h5py.File(hdf5_path, 'r') as hf:
        dataset_X = np.array(hf["dataset_X"][:])
        dataset_y = np.array(hf["dataset_y"][:])

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(dataset_y)
    dataset_y = pd.get_dummies(y1).values
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=33)

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))
    return X_train, X_test, y_train, y_test

def train():
    my_new_model = Sequential()
    my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    my_new_model.add(Dense(num_classes, activation='softmax'))
    my_new_model.layers[0].trainable = False
    my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    my_new_model.fit(X_train, y_train, epochs=2, batch_size=128)
    preds = my_new_model.evaluate(X_test, y_test, batch_size=10, verbose=1, sample_weight=None)
    my_new_model.summary()
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

X_train, X_test, y_train, y_test = load_dataset()
train()
