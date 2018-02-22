import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


DISPLAY = False


def main():
    # Load in the train and test datasets
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']

    np.random.seed(2)

    full_data = [train, test]
    x_train, y_train, x_test, feature_names = feature_engineering(full_data)
    predictions = classifier_stacking(x_train, y_train, x_test, feature_names)

    # on the below: we reduce the input set, in order to try to validate our results
    reduced_x_train, reduced_y_train, cross_val_x, cross_val_y = split_train_set(x_train, y_train, 0.8)
    cross_val_pred = classifier_stacking(reduced_x_train, reduced_y_train, cross_val_x, feature_names)
    print("ACCURACY ON CROSS VAL :", (cross_val_pred == cross_val_y).astype(int).mean())

    # Generate Submission File
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv("MyStackingSubmission.csv", index=False)


def split_train_set(x, y, split_ratio):
    assert(x.shape[0] == y.shape[0])

    # shuffle inputs
    p = np.random.permutation(x.shape[0])
    shuffled_x = x[p]
    shuffled_y = y[p]

    # split our training data into 2 sets
    n = int(shuffled_x.shape[0] * split_ratio)
    cross_val_x = shuffled_x[n:]
    reduced_x = shuffled_x[:n]
    cross_val_y = shuffled_y[n:]
    reduced_y = shuffled_y[:n]

    return reduced_x, reduced_y, cross_val_x, cross_val_y


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


def classifier_stacking(x_train, y_train, x_test, cols):
    SEED = 0  # for reproducibility

    # Class to extend the Sklearn classifier
    class SklearnHelper(object):

        def __init__(self, clf, seed=None, params=None):
            if seed:
                params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

        def fit(self, x, y):
            return self.clf.fit(x, y)

        def feature_importances(self, x, y):
            res_fit = self.clf.fit(x, y)
            try:
                return res_fit.feature_importances_
            except AttributeError:
                return None

    # Put in our parameters for said classifiers
    # Random Forest parameters
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 800,
        'warm_start': True,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }

    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.025
    }

    # KNN params
    knn_params = {
        'algorithm': 'auto',
        'leaf_size': 26,
        'metric': 'minkowski',
        'metric_params': None,
        'n_jobs': -1,
        'n_neighbors': 6,
        'p': 2,
        'weights': 'uniform'}

    # Create 5 objects that represent our 4 models
    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
    knn = SklearnHelper(clf=KNeighborsClassifier, params=knn_params)

    # One the below; We train each model with Kfold method
    # ##############################################
    classifiers = [rf, et, ada, gb, svc, knn]

    n_splits = 5
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1./n_splits, random_state=0)
    X = x_train
    y = y_train

    acc_dict = {}
    my_train_predictions = dict()
    my_test_predictions = dict()
    i = 0
    for train_index, cross_val_index in sss.split(X, y):  # k_fold core computations
        X_train, X_cross_val = X[train_index], X[cross_val_index]
        Y_train, Y_cross_val = y[train_index], y[cross_val_index]

        for clf in classifiers:
            name = clf.clf.__class__.__name__
            clf.fit(X_train, Y_train)
            if name not in my_train_predictions.keys():
                my_train_predictions[name] = np.zeros((x_train.shape[0], ))
            my_train_predictions[name][cross_val_index] = clf.predict(X_cross_val)
            if name not in my_test_predictions.keys():
                my_test_predictions[name] = np.zeros((x_test.shape[0], n_splits))
            my_test_predictions[name][:, i] = clf.predict(x_test)

            acc = accuracy_score(Y_cross_val, my_train_predictions[name][cross_val_index])
            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc
        i += 1
    print("Training is complete")

    # we aggregate results into helpful dictionaries
    log_cols = ["Classifier", "Accuracy"]
    logs = pd.DataFrame(columns=log_cols)
    avg_test_prediction, majority_test_prediction = dict(), dict()
    print("--- accuracy (obtained by kfold method on train set) ---")
    flatten_train_predictions = dict()
    for clf in classifiers:
        name = clf.clf.__class__.__name__
        avg_test_prediction[name] = my_test_predictions[name].mean(axis=1, keepdims=True)
        majority_test_prediction[name] = line_by_line_most_frequent(my_test_predictions[name]).reshape((-1, 1))
        acc_dict[name] /= n_splits
        log_entry = pd.DataFrame([[name, acc_dict[name]]], columns=log_cols)
        logs = logs.append(log_entry)
        # we save flatten version to display heatmap between models a bit later
        flatten_train_predictions[name] = my_train_predictions[name]
        my_train_predictions[name] = my_train_predictions[name].reshape((-1, 1))
        print(name, round(acc_dict[name], 4))

    # two following lines: size of display
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=logs, color="b")
    if DISPLAY: plt.show()

    # below: heatmap of results with different models (better if not too similar results)
    base_predictions_train = pd.DataFrame(flatten_train_predictions)
    print(base_predictions_train.head())
    g = sns.heatmap(base_predictions_train.astype(float).corr(), annot=True, fmt=".2f", cmap="coolwarm")
    if DISPLAY: plt.show()

    # below: work on importance of features according to those models
    feature_importance = {'features': cols}
    for clf in classifiers:
        clf_feature_importance = clf.feature_importances(x_train, y_train)
        if clf_feature_importance is not None:  # does not work with SVC for example
            feature_importance[clf.clf.__class__.__name__] = clf_feature_importance
    feature_dataframe = pd.DataFrame(feature_importance)  # Create a dataframe with features
    feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
    print(feature_dataframe.head(10))

    # below: level 2 data generation and predictions
    x_train_level_2 = np.concatenate(tuple([my_train_predictions[c.clf.__class__.__name__] for c in classifiers]),
                                     axis=1)
    x_test_level_2 = np.concatenate(tuple([avg_test_prediction[c.clf.__class__.__name__] for c in classifiers]), axis=1)
    x_test_level_2_bis = np.concatenate(
        tuple([majority_test_prediction[c.clf.__class__.__name__] for c in classifiers]),
        axis=1)

    # res = list()
    # rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    # et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    # ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    # gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    # svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
    # meta_classifiers = [rf, et, ada, gb, svc]
    # for clf in meta_classifiers:
    #     print(clf.clf.__class__.__name__)
    #     clf.train(x_train_level_2, y_train)
    #     res.append(clf.predict(x_test_level_2))
    #     res.append(clf.predict(x_test_level_2_bis))
    #return res

    rf_level_2 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    rf_level_2.train(x_train_level_2, y_train)
    level_2_prediction = rf_level_2.predict(x_test_level_2)

    return level_2_prediction


def line_by_line_most_frequent(input_matrix):
    r = list()
    for line in input_matrix:
        d = dict()
        for el in line:
            if el in d.keys():
                d[el] += 1
            else:
                d[el] = 1
        v = list(d.values())
        k = list(d.keys())
        r.append([k[v.index(max(v))]])
    return np.array(r).reshape((-1,))

if __name__ == '__main__':
    main()
