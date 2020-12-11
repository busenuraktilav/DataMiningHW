import pandas as pd
from sklearn import preprocessing
import numpy as np
import math

#read the file
bank_cust = pd.read_csv('bank_customer.csv',  delimiter=',')

#a) combine similar jobs
bank_cust.loc[bank_cust.job == 'admin.', 'job'] = "white-collar"
bank_cust.loc[bank_cust.job == 'management', 'job'] = "pink-collar"
bank_cust.loc[bank_cust.job == 'services', 'job'] = "pink-collar"
bank_cust.loc[bank_cust.job == 'housemaid', 'job'] = "pink-collar"
bank_cust.loc[bank_cust.job == 'retired', 'job'] = "other"
bank_cust.loc[bank_cust.job == 'student', 'job'] = "other"
bank_cust.loc[bank_cust.job == 'unemployed', 'job'] = "other"

#a) combine "poutcome" values
bank_cust.loc[bank_cust.poutcome == 'other', 'poutcome'] = "unknown"

#a) print all types with counts
print("\n########## JOB COUNTS ##########")
print(bank_cust['job'].value_counts())
print("\n########## POUTCOME COUNTS ##########")
print(bank_cust['poutcome'].value_counts())

#b) convert categorical values into numerical values
le = preprocessing.LabelEncoder()
bank_cust['job'] = le.fit_transform(bank_cust['job'])
bank_cust['marital'] = le.fit_transform(bank_cust['marital'])
bank_cust['education'] = le.fit_transform(bank_cust['education'])
bank_cust['default'] = le.fit_transform(bank_cust['default'])
bank_cust['housing'] = le.fit_transform(bank_cust['housing'])
bank_cust['loan'] = le.fit_transform(bank_cust['loan'])
bank_cust['contact'] = le.fit_transform(bank_cust['contact'])
bank_cust['month'] = le.fit_transform(bank_cust['month'])
bank_cust['poutcome'] = le.fit_transform(bank_cust['poutcome'])
bank_cust['deposit'] = le.fit_transform(bank_cust['deposit'])

#c) split the data into subsets
msk = np.random.rand(len(bank_cust)) < 0.7
train_set = bank_cust[msk]
test_set = bank_cust[~msk]

#d) create 2-dataset with different attributes
data_1_train = train_set[['age', 'job', 'marital', 'education', 'balance', 'housing', 'duration', 'poutcome']]
data_1_test = test_set[['age', 'job', 'marital', 'education', 'balance', 'housing', 'duration', 'poutcome']]
data_2_train = train_set[['job', 'marital', 'education', 'housing']]
data_2_test = test_set[['job', 'marital', 'education', 'housing']]


#e) calculates the entropy of a given attribute
def entropy_measure(dataset, attribute_name):
    c = dataset[attribute_name].nunique()
    t = len(dataset)
    sum = 0
    for j in range(c):
        i = len(dataset[dataset[attribute_name] == j])
        if(i != 0):
            sum += ((i/t)*(math.log(i/t, 2)))

    return(-sum)


#f) calculates the gini index of a given attribute
def gini_index(dataset, attribute_name):
    t = dataset[attribute_name].nunique()
    l1 = [[], []]
    l2 = []
    for i in range(t):
        temp = len((bank_cust[bank_cust[attribute_name] == i])[bank_cust['deposit'] == 0])
        l1[0].append(temp)
        temp = len((bank_cust[bank_cust[attribute_name] == i])[bank_cust['deposit'] == 1])
        l1[1].append(temp)

    for i in range(t):
        temp = 1 - np.square(l1[0][i]/(l1[0][i] + l1[1][i])) - np.square(l1[1][i]/(l1[0][i] + l1[1][i]))
        l2.append(temp)

    sum = 0
    for i in range(t):
        sum += ((l1[0][i]+l1[1][i])/len(dataset))*l2[i]

    return(sum)

a = entropy_measure(data_2_train, "marital")
print(a)

#NOTE:
#I did not quite understand how to build a decision tree so I could not implement that part.

