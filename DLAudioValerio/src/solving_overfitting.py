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
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    print("Data succesfully loaded!")

    return inputs, targets


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
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3
    )
    # build the network architecture
    model = keras.Sequential(
        [
            # input layer
            keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
            # 1st hidden layer
            keras.layers.Dense(
                512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # 2nd hidden layer
            keras.layers.Dense(
                256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # 3rd hidden layer
            keras.layers.Dense(
                64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # output layer
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # train network
    history = model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        epochs=50,
        batch_size=32,
    )
    # plot accuracy and error over the epochs
    plot_history(history)
