import numpy as np

from keras.utils import np_utils
from keras.models import Sequential

from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D


BATCH_SIZE = 32
NB_EPOCH = 50


class FashionMNISTModel:

    def __init__(self):

        mnist_model = Sequential()

        mnist_model.add(Conv2D(32, (3, 3),
                               padding='same',
                               input_shape=(28, 28, 1), activation='relu'))

        mnist_model.add(MaxPooling2D(pool_size=(2, 2)))

        mnist_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

        mnist_model.add(MaxPooling2D(pool_size=(2, 2)))

        mnist_model.add(Flatten())
        mnist_model.add(Dense(128, activation='relu'))

        mnist_model.add(Dense(4, activation='softmax'))

        mnist_model.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])

        self.model = mnist_model

    def preprocess_training_data(self, X, y):

        # Put our input data in the range 0-1
        X /= 255
        (n, w, h) = X.shape
        X = X.reshape(n, w, h, 1)

        # Convert class vectors to binary class matrices
        y = np_utils.to_categorical(y, 4)

        return X, y

    def fit(self, X, y):

        self.model.fit(X, y, batch_size=BATCH_SIZE, epochs=NB_EPOCH)

    def preprocess_unseen_data(self, X):

        X /= 255
        (n, w, h) = X.shape
        X = X.reshape(n, w, h, 1)
        return X

    def predict(self, X):

        pred_pros = self.model.predict(X)
        pred_idxs = [np.argmax(pred) for pred in pred_pros]
        return pred_idxs
