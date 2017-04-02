import pandas as pd
import numpy as np
import math
import re

from matplotlib import pyplot as plt


# Read in the train and test data
# I might take ~200 training examples to use as a cross validation set, will revisit
train = pd.read_csv('../../DataSets/TitanicIntro/train.csv')
test = pd.read_csv('../../DataSets/TitanicIntro/test.csv')

'''
The scrubbing plan:
    Variables not used for prediction purposes:
        PassengerID
        Survived
        
    Variables that will be used, with slight modifications:
        Sex           male -> 1, female -> 0 
        Cabin         change from cabin number(s) to number of cabins purchased (owned? I don't know) 
        Embarked      C = 0, Queenstown = 1, Southampton = 2
        Name          Name doesn't matter, but title might (Mr, Miss, Mrs, Dr.)

    Variables that will need to cleaned:
        Age           lots of missing ages 
                       -don't want to throw out entire example due to relatively small training set.
                       -maybe sample from a distribution

    Variables I'm not sure of right away:
        Ticket Number. Is the number ordered or related too the layout of the ship or ticket class or order sold?
                       It's a number, sometimes there's letters though. I don't know what they mean.

All titles (taken from Name field)                              Value Mapped To
---------------------------------------------------------------------------------------
                                                           |       
Military: Major, Col, Capt.                                |   4
Nobility: the (countess), Lady, Sir, Jonkheer, Don, Dona   |   3
Educated and Spiritual: Dr, Rev                            |   2
Married: Mr, Mrs, Mme                                      |   1
Unmarried: Master, Miss, Miss                              |   0
---------------------------------------------------------------------------------------

Header: 

PassengerId | Survived | Pclass | Name | Sex | Age | SibSpouse | ParChild | Ticket # | Fare | Cabin | Embarked

'''

#--------------------------
# Collect survival indicator
y = train['PassengerId'][:-200]
yc = train['PassengerId'][-200:]
ytest = test['PassengerId']

#--------------------------

train = train[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
test = test[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------
Helper functions for parsing through the data in some way
-----------------------------------------------------------------------
'''
def get_all_titles(df):
    names = df['Name']
    r = r', \w+' #regex to find titles from the name
    titles = []
    for name in names:
        title = re.search(r, name) 
        title = (title.group())[2:] #remove trailing space
        if title not in titles:
            titles.append(title)
    return titles

def generate_age_CDF(data):
    '''
    Generates an age sample by drawing a uniform random number from the 
    empirical CDF of the known ages from the data
    '''
    n = len(data) - sum(data.isnull())
    num_bins = (math.ceil(max(data)) - math.floor(min(data))) + 1
    cdf = np.zeros((num_bins,1))
    for t in range(num_bins):
        for x in data:
            if x != x:
                continue
            cdf[t] += 1/n * (x <= min(data)+ t)
    return cdf

def generate_age_sample(cdf):
    r = np.random.uniform()
    ind = 0
    while (cdf[ind] < r):
        ind += 1
    return ind

#Use to visualize age distribution before and after replacement
def plot_ages(df):
    ages_no_nans = []
    ages = []
    age_list = df['Age']
    cdf = generate_age_CDF(age_list);
    for age in age_list:
        if (age != age):
            age = generate_age_sample(cdf)
            ages.append(age)
            continue
        ages_no_nans.append(age)
        ages.append(age)
    plt.figure(1)
    plt.hist(ages_no_nans)
    plt.figure(2)
    plt.hist(ages)
    plt.show()


def normalize(df):
    normalizable_columns = ['Age', 'Fare']
    for c in normalizable_columns:
        x = df[c]
        mu = df[c].mean()
        range = df[c].max() - df[c].min()
        df[c] = (x - mu)/range
    return df
'''
-----------------------------------------------------------------------
'''
#Get the set of all titles
all_titles_train = get_all_titles(train)
all_titles_test = get_all_titles(test)

'''
-----------------------------------------------------------------------
-----------------------------------------------------------------------

DataFrame helper functions for cleaning up the data
-----------------------------------------------------------------------
'''
def replace_gender(x):
   gender = x['Sex']
   if (gender == 'male'):
       return 1
   elif (gender == 'female'):
       return 0
   else:
       return gender

def count_cabins(x):
    cabin = x['Cabin']
    if (cabin != cabin): #test for NaN
        return 0
    cabin = cabin.split()
    count = 0
    for c in cabin:
        count += 1
    return count

def replace_name(x):
    name = x['Name']
    r = r', \w+'
    m = re.search(r, name)
    name = (m.group())[2:] #throw away comma in regex 

    military = ['Col', 'Capt', 'Major']
    nobility = ['Don', 'Dona', 'Sir', 'the', 'Jonkheer']
    educated = ['Dr', 'Rev']
    married = ['Mr', 'Mrs', 'Mme']
    unmarried = ['Master', 'Mlle', 'Miss', 'Ms']
    if name in military:
        return 4
    elif name in nobility:
        return 3
    elif name in educated:
        return 2
    elif name in married:
        return 1
    elif name in unmarried:
        return 0
    else:
        return float('nan')

def replace_embark(x):
    embark = x['Embarked']
    if embark == 'C':
        return 0
    elif embark == 'Q':
        return 1
    elif embark == 'S':
        return 2
    else:
        return float('nan')

def replace_age(x, cdf):
    age = x['Age']
    if age != age:
        age = generate_age_sample(cdf)
    return age
'''
-----------------------------------------------------------------------
'''


#--------------------------------------------------
#Clean up the data
cdf = generate_age_CDF(train['Age']) # Need this to take samples from later
train['Sex'] = train.apply(replace_gender, axis=1)
train['Cabin'] = train.apply(count_cabins, axis=1)
train['Name'] = train.apply(replace_name, axis=1)
train['Embarked'] = train.apply(replace_embark, axis=1)
train['Age'] = train.apply((lambda x: replace_age(x, cdf)), axis=1)

test['Sex'] = test.apply(replace_gender, axis=1)
test['Cabin'] = test.apply(count_cabins, axis=1)
test['Name'] = test.apply(replace_name, axis=1)
test['Embarked'] = test.apply(replace_embark, axis=1)
test['Age'] = test.apply((lambda x: replace_age(x, cdf)), axis=1)


#Normalize data
train = normalize(train)
test = normalize(test)

#create cross validation set
cv = train[-200:]
train = train[:-200]

print(train.values)
#Write new csv files
train.to_csv('../../DataSets/TitanicIntro/train_cleaned.csv')
cv.to_csv('../../DataSets/TitanicIntro/cv_cleaned.csv')
test.to_csv('../../DataSets/TitanicIntro/test_cleaned.csv')
