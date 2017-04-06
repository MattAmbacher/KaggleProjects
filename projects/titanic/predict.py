import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn.ensemble
import sklearn.svm
from sklearn.model_selection import GridSearchCV

sys.path.append('../')
from Network import Network

#Read in that sweet, sweet data
uncleaned_test_data = pd.read_csv('../../DataSets/TitanicIntro/test.csv')
training_data = pd.read_csv('../../DataSets/TitanicIntro/train_cleaned.csv')
cv_data = pd.read_csv('../../DataSets/TitanicIntro/cv_cleaned.csv')
test_data = pd.read_csv('../../DataSets/TitanicIntro/test_cleaned.csv')

training_data = training_data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
    'Embarked', 'Survived']]
cv_data = cv_data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
    'Embarked', 'Survived']]
test_data = test_data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
    'Embarked']]

training_data = training_data.values
cv_data = cv_data.values
test_data = test_data.values

X = training_data[:,:-1]
y = training_data[:,-1]

Xc = cv_data[:,:-1]
yc = cv_data[:,-1]

tX = test_data

training_data = list(zip(X,y))
cv_data = list(zip(Xc, yc))

#Support Vector Classifier
svm = sklearn.svm.SVC()
C_list = np.logspace(-1, 1, 8)
g_list = np.logspace(-5, -1, 8)
params = {'kernel': 'rbf', 'C': C_list, 'gamma': g_list}
GridSearchCV(svm, params)

svm.fit(X, y)
print(svm.score(Xc,yc))
'''
#RFC classifier
forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, bootstrap=True)
forest = forest.fit(X, y)

predictions = forest.predict(tX).astype(int)
'''
'''
print("Test: {0}\t|\tTrain: {1}\t|\t(lambda, eta) = ({2},{3})".format(
    test_accuracy, train_accuracy, lmbda, eta))
'''
#Build the output for submission
output = uncleaned_test_data['PassengerId'].to_frame()
output = output.assign(Survived=predictions)

output.to_csv('prediction.csv', index=False)
