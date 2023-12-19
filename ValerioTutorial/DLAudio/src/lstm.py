import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

DATASET_PATH = "datasets/GTZAN/data_10.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # conver lists into numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    print("Data succesfully loaded!")

    return X, y

def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    # Generates RNN-LSTM model
    # build network topology
    model = keras.Sequential()
    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


def predict(model, X, y):
    X = X[np.newaxis, ...]
    # predicrtions = [ [0.1, 0.2, ..., ]]
    predictions = model.predict(X)  # X -> (1,130,13,1)
    # extract index with max value
    predicted_index = np.argmax(predictions, axis=1)  # [3]
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


def plot_history(history):
    fig, axs = plt.subplots(2)
    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create accuracy subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("error")
    axs[1].set_xlabel("epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.savefig("tmp.png")


if __name__ == "__main__":
    # get train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2
    )  # 0.25 = 1/4をテストに使う、0.2 = 残りの75%の内、20%をvalidation に使用する

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])  # 130, 13 / timestep, mfccbin
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    # train the CNN
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        batch_size=32,
        epochs=30,
    )

    # plot accuracy and error over the epochs
    plot_history(history)
    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make prediction on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)
