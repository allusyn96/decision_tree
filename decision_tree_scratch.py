from sklearn import datasets
import pandas as pd
import random
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

'''
PROCEDURE
load_data(filename)
train_test_split(df, test_size)-->cross_validate(df, test_size, folds)
decision_tree(train_data)
accuracy(train_data, tree)
'''

# PART 1: Load Data
'''
This is a helper function that converts a csv file into a 
pandas data frame and returns this data frame
'''
def load_data(filename, cols, label):

    x_df = pd.DataFrame(filename.data)
    y_df = pd.DataFrame(filename.target)

    x_df = x_df.rename(columns=cols)
    d_f = pd.concat([x_df, y_df], axis=1)
    d_f = d_f.rename(columns={0: label})
    d_f.Label.replace([0, 1, 2], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], inplace=True)

    return d_f

# PART 2A: Split the data using basic train test split method
def train_test_split(daf, test_size):

    if isinstance(test_size, float):
        test_size = test_size*len(daf)

    # generate random indices
    rand_nums = np.random.randint(0, len(daf), test_size)

    # choose a subset of the training & testing data based on the random indices
    test_df = daf.loc[rand_nums]
    train_df = daf.drop(rand_nums, axis=0)

    return train_df, test_df

# PART 2B: Split the data using k-fold validation
def cross_validate(df, test_size, kfolds):

    train_data_set, test_data_set = [], []
    for i in range(0, kfolds):
        # generate random indices
        rand_nums = np.random.randint(0, len(df), test_size)

        test_data_frame = df.loc[rand_nums]
        train_data_frame = df.drop(rand_nums, axis=0)

        train_data_set.append(train_data_frame)
        test_data_set.append(test_data_frame)

    return train_data_set, test_data_set

# PART 3A: Check purity function (Helper function)
'''
This will return true if a dataset has only one label, else it
will return false
'''
def check_purity(df):

    label = df[:, -1]
    unique_classes = np.unique(label)
    if len(unique_classes) == 1:
        return True
    else:
        return False

# PART 3B: Classify data
'''
<<<<<<< HEAD
This function returns the class label containing the most predominant values
by identifying the greatest number of flowers within a class 
=======
If data is pure we can classify the data
>>>>>>> 56ae0f550b88f477098002758a748298d082b433
'''
def classify(data):

    label = data[:, -1]
    unique_classes, counts = np.unique(label, return_counts=True)
    idx = counts.argmax()
    classification = unique_classes[idx]

    return classification

#### PART 3X: Build Decision tree Algorithm
def decision_tree():

    return True

#### PART 4: Determine Accuracy


f_name = datasets.load_iris()
m_columns = {0: 'Sepal length', 1: 'Sepal width', 2: 'Petal length', 3: 'Petal width'}
m_label = 'Label'
df = load_data(f_name, m_columns, m_label)

train_df, test_df = train_test_split(df, 20)

'''
train_set, test_set = cross_validate(df, 20, 4)
for i in range(0, len(test_set)):
    print 'test set #{0}'.format(i+1)
    print test_set[i]
'''

# convert training dataframe into a numpy 2-D array
train_data = train_df.values
train_data2 = train_df[train_df['Petal width'] < 0.8].values
x = check_purity(train_data2)

y = classify(train_data)
print y

