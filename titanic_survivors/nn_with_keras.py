import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers

DISPLAY = True


def main():
    # Load in the train and test datasets
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']

    random_seed = 0
    np.random.seed(random_seed)  # used implicitly in feature_engineering with age feature

    full_data = [train, test]
    x_train, y_train, x_test, feature_names = feature_engineering(full_data)

    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=random_seed)

    # define and configure model
    model = prepare_model(x_train.shape[1])

    # Define the optimizer
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    epochs = 300
    batch_size = 1024  # bigger than test size, on purpose (so basically no batches, input is small enough)
    history = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                        verbose=2)

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    if DISPLAY: plt.show()

    # a bit cheated, we continue training our model on full input instead of part of it
    # CARE: X_val does not have any validation sens now
    epochs_then = 100
    model.fit(x=x_train, y=y_train, epochs=epochs_then, batch_size=batch_size, validation_data=(X_val, Y_val),
              verbose=2)

    # predict results
    # predictions = model.predict(x_test)  # to get percentages
    predictions = model.predict_classes(x_test)  # to get actual predictions

    print(x_test.shape)
    print(predictions.shape)

    # Generate Submission File
    # performs 0.80382 on Kaggle -> We are happy !
    submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions.reshape((-1,))})
    submission.to_csv("MyDenseNNSubmission.csv", index=False)


def feature_engineering(full_data):
    train, test = full_data

    # Some features of my own that I have added in
    # Gives the length of the name
    train['Name_length'] = train['Name'].apply(len)
    test['Name_length'] = test['Name'].apply(len)
    # Feature that tells whether a passenger had a cabin on the Titanic
    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Feature engineering steps taken from Sina (see classifier analysis)
    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

    # Create a New feature CategoricalAge
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    train['CategoricalAge'] = pd.cut(train['Age'], 5)

    # Define function to extract titles from passenger names
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    # Create a new feature Title, containing the titles of passenger names
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    # Feature selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train = train.drop(drop_elements, axis=1)
    # train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1) # original choice
    train = train.drop(['CategoricalAge', 'CategoricalFare', 'Name_length'], axis=1)  # rpil test
    test = test.drop(['Name_length'], axis=1)  # rpil test
    test = test.drop(drop_elements, axis=1)

    # visualisations
    print(train.head(10))

    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    if DISPLAY: plt.show()

    # below pairplot not working here (singular matrix)
    # g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
    #                         u'FamilySize', u'Title']], hue='Survived', palette='seismic', size=1.2, diag_kind='kde',
    #                  diag_kws=dict(shade=True), plot_kws=dict(s=10))
    # g.set(xticklabels=[])
    # if DISPLAY: plt.show()

    # on the below: vectorization
    feature_names = train.columns.values[1:]
    train = train.values
    test = test.values
    y_train = train[0:, 0]
    train = train[0:, 1:]

    print(train.shape)
    print(y_train.shape)
    print(test.shape)

    return train, y_train, test, feature_names


def prepare_model(n_features):
    n_activations = 256
    base_dropout = 0.2
    l2_regularization_factor = 0.005
    model = Sequential()

    model.add(Dense(n_activations, activation="relu", input_dim=n_features,
                    kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout))

    model.add(Dense(n_activations, activation="relu", kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout))

    model.add(Dense(n_activations, activation="relu", kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout))

    model.add(Dense(n_activations, activation="relu", kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout*1.3))
    model.add(Dense(2, activation="softmax"))
    return model


if __name__ == '__main__':
    main()
