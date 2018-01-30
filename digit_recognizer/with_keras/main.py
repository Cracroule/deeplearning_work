import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)

DISPLAY = False


def main():

    # Load data
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1)

    # free some space
    del train

    pre_analyze_data(X_train, Y_train, test)
    X_train, Y_train, X_val, Y_val, test = prepare_data(X_train, Y_train, test)
    post_analyze_data(X_train, Y_train)

    # define and configure model
    model = prepare_model()

    # Define the optimizer
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Its better to have a decreasing learning rate during the training to reach efficiently the global
    # minimum of the loss function.
    # To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically
    # every X steps (epochs) depending if it is necessary (when accuracy is not improved).
    # With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    epochs = 20  # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 128

    # # With data augmentation to prevent overfitting (accuracy 0.99286)
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    #     zoom_range=0.1,  # Randomly zoom image
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=False,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    #
    # datagen.fit(X_train)
    #
    # # Fit the model
    # history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
    #                               epochs = epochs, validation_data=(X_val, Y_val),
    #                               verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
    #                               , callbacks=[learning_rate_reduction])

    history = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val),
                        verbose=2, callbacks=[learning_rate_reduction])

    # predict results
    results = model.predict(test)

    # select the indix with the maximum probability
    results = np.argmax(results, axis=1)

    results = pd.Series(results, name="Label")

    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

    submission.to_csv("cnn_mnist_datagen.csv", index=False)

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)


def pre_analyze_data(X_train, Y_train, test):
    g = sns.countplot(Y_train)
    if DISPLAY: plt.show()
    print(Y_train.value_counts())

    # Check the data
    print(X_train.isnull().any().describe())
    print(test.isnull().any().describe())


def prepare_data(X_train, Y_train, test):

    # Normalize data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes=10)

    random_seed = 2

    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

    return X_train, Y_train, X_val, Y_val, test


def post_analyze_data(X_train, Y_train):

    # Some examples
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(9):
        plt.subplot(3, 3, i+1)
        g = plt.imshow(X_train[i][:, :, 0], cmap='gray', vmin=0, vmax=1)
    if DISPLAY: plt.show()


def prepare_model():
    # Set the CNN model
    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    return model

if __name__ == '__main__':
    main()
