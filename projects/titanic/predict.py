import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


sys.path.append('../')
import NN

#Read in that sweet, sweet data

training_data = pd.read_csv('../../DataSets/TitanicIntro/train_cleaned.csv')
cv_data = pd.read_csv('../../DataSets/TitanicIntro/cv_cleaned.csv')
training_data = training_data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
    'Embarked', 'Survived']]
cv_data = cv_data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin',
    'Embarked', 'Survived']]

training_data = training_data.values
cv_data = cv_data.values
network = NN.Network([training_data.shape[1]-1, 5, 1])

lambdas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
cost = []
cv_cost = []
for l in lambdas:
    print("lambda = {}".format(l))
    print('='*10)
    c = network.SGD(training_data, 200, 20, 1, l)
plt.figure()
plt.plot(cost)
plt.hold('on')
plt.plot(cv_cost)
plt.show()
