import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


sys.path.append('../')
from Network import Network

#Read in that sweet, sweet data

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

tX = test_data[:,:-1]
ty = test_data[:,:-1]

training_data = list(zip(X,y))
cv_data = list(zip(Xc, yc))
test_data = list(zip(tX, ty))



network = Network([len(X[0]), 25, 10, 1])
lmbda = 0.667
eta = 2.6

test_accuracy, train_accuracy =  network.SGD(training_data, cv_data, 50, 20, eta, lmbda)
print("Test: {0}\t|\tTrain: {1}\t|\t(lambda, eta) = ({2},{3})".format(
    test_accuracy, train_accuracy, lmbda, eta))
