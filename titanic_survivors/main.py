# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
#import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

DISPLAY = False


def main():
    # Load in the train and test datasets
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']

    print(train.head(3))

    np.random.seed(1)

    full_data = [train, test]
    x_train, y_train, x_test, feature_names = feature_engineering(full_data)
    predictions = classifier_stacking(x_train, y_train, x_test, feature_names)

    # on the below: we reduce the input set, in order to try to validate our results
    reduced_x_train, reduced_y_train, cross_val_x, cross_val_y = split_train_set(x_train, y_train, 0.9)
    cross_val_predictions = classifier_stacking(reduced_x_train, reduced_y_train, cross_val_x, feature_names)
    print("ACCURACY ON CROSS VAL :", (cross_val_predictions == cross_val_y).mean())

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
    # Some useful parameters which will come in handy later on
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

    # Class to extend the Sklearn classifier
    class SklearnHelper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

        def fit(self, x, y):
            return self.clf.fit(x, y)

        def feature_importances(self, x, y):
            return self.clf.fit(x, y).feature_importances_

            # Class to extend XGboost classifer

    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        # returns the predictions made on artificially created test samples (from train set)
        # and actual test predictions (as mean of 5 obtained params set)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

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

    # Create 5 objects that represent our 4 models
    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

    # Create our OOF train and test predictions. These base results will be used as new features
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
    ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
    gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
    svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

    print("Training is complete")
    rf_feature = rf.feature_importances(x_train, y_train)
    et_feature = et.feature_importances(x_train, y_train)
    ada_feature = ada.feature_importances(x_train, y_train)
    gb_feature = gb.feature_importances(x_train, y_train)

    # Create a dataframe with features
    feature_dataframe = pd.DataFrame({'features': cols,
                                      'Random Forest feature importances': rf_feature,
                                      'Extra Trees  feature importances': et_feature,
                                      'AdaBoost feature importances': ada_feature,
                                      'Gradient Boost feature importances': gb_feature
                                      })

    # Create the new column containing the average of values

    feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
    print(feature_dataframe.head(10))

    base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                           'ExtraTrees': et_oof_train.ravel(),
                                           'AdaBoost': ada_oof_train.ravel(),
                                           'GradientBoost': gb_oof_train.ravel()
                                           })
    base_predictions_train.head()

    g = sns.heatmap(base_predictions_train.astype(float).corr(), annot=True, fmt=".2f", cmap="coolwarm")
    if DISPLAY: plt.show()

    x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
    x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

    rf2 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    rf2.train(x_train, y_train)
    res = rf2.predict(x_test)

    # The below works as a vote, the majority choose the final class
    # results are the same on test i made
    # res2 = line_by_line_most_frequent(x_test)

    return res


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
