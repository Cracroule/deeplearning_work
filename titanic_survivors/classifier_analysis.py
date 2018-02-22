import numpy as np
import pandas as pd
import re as re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

DISPLAY = True


def main():
    train = pd.read_csv('./input/train.csv', header=0, dtype={'Age': np.float64})
    test = pd.read_csv('./input/test.csv', header=0, dtype={'Age': np.float64})

    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']

    full_data = [train, test]

    print(train.info())

    np.random.seed(0)
    train, test = feature_engineering(full_data)
    best_classifier = classifier_comparison(train, test)
    predictions = best_classifier.predict(test)

    # Generate Submission File
    best_classifier_submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    best_classifier_submission.to_csv("BestClassifierSubmission.csv", index=False)


def feature_engineering(full_data):
    train, test = full_data

    print()
    print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

    print()
    print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

    print()
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

    print()
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

    print()
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

    print()
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4) # categorization in 4 ranges
    print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

    print()
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()

        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    train['CategoricalAge'] = pd.cut(train['Age'], 5)
    print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

    print()
    def get_title(name):
        title_search = re.search(" ([A-Za-z]+)\.", name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    print(pd.crosstab(train['Title'], train['Sex']))
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

    # Data cleaning
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

    # Feature Selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
    train = train.drop(drop_elements, axis=1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
    test = test.drop(drop_elements, axis=1)
    print(train.head(10))

    return train.values, test.values


def classifier_comparison(train, test):

    classifiers = [
        KNeighborsClassifier(3),
        SVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression()]

    log_cols = ["Classifier", "Accuracy"]
    logs = pd.DataFrame(columns=log_cols)

    n_splits = 10
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=0)

    X = train[0::, 1::]
    y = train[0::, 0]

    acc_dict = {}

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, train_predictions)
            # acc = f1_score(y_test, train_predictions)
            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / n_splits
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        logs = logs.append(log_entry)

    # two folloowing lines: size of display
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)

    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=logs, color="b")
    if DISPLAY: plt.show()

    print()
    best_acc, best_clf = 0., None
    for clf in classifiers:
        name = clf.__class__.__name__
        print(name, round(acc_dict[name], 4))
        if acc_dict[name] > best_acc:
            best_acc = acc_dict[name]
            best_clf = clf
    print("\nbest classifier for this problem is: ", best_clf.__class__.__name__)
    best_clf.fit(train[0::, 1::], train[0::, 0])  # train best classifier on full training
    return best_clf

if __name__ == '__main__':
    main()
