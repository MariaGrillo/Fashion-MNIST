import argparse
import os
import pickle
import urllib.request

import numpy as np

from model import FashionMNISTModel as Model

DATASET_URL = "https://s3-eu-west-1.amazonaws.com/kate-datasets/fashion/"
X_TRAIN_NAME = "X_train.npy"
Y_TRAIN_NAME = "y_train.npy"
X_TEST_NAME = "X_test.npy"

DATA_DIR = "data"
PICKLE_NAME = 'model.pickle'


def setup_data():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    for filename in [X_TRAIN_NAME, Y_TRAIN_NAME, X_TEST_NAME]:
        print("Downloading {}...".format(filename))
        req = urllib.request.urlopen(DATASET_URL + filename)
        data = req.read()

        with open(os.path.join(DATA_DIR, filename), "wb") as f:
            f.write(data)


def train_model():
    X = np.load(os.sep.join([DATA_DIR, X_TRAIN_NAME]))
    y = np.load(os.sep.join([DATA_DIR, Y_TRAIN_NAME]))

    my_model = Model()
    X_train, y_train = my_model.preprocess_training_data(X, y)
    my_model.fit(X_train, y_train)

    # Save to pickle
    with open(PICKLE_NAME, 'wb') as f:
        pickle.dump(my_model, f)


def test_model():
    X = np.load(os.sep.join([DATA_DIR, X_TEST_NAME]))

    # Load pickle
    with open(PICKLE_NAME, 'rb') as f:
        my_model = pickle.load(f)

    X_test = my_model.preprocess_unseen_data(X)
    preds = my_model.predict(X_test)
    print("### Your predictions ###")
    print(preds)


def main():
    parser = argparse.ArgumentParser(
        description="A command line-tool to manage the project.")
    parser.add_argument(
        'stage',
        metavar='stage',
        type=str,
        choices=['setup', 'train', 'test'],
        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == "setup":
        setup_data()
        print("\nSetup was successful!")

    elif stage == "train":
        print("Training model...")
        train_model()

    elif stage == "test":
        print("Testing model...")
        test_model()


if __name__ == "__main__":
    main()
