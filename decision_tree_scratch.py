from sklearn import datasets

import matplotlib
matplotlib.use('TkAgg')

import seaborn as sns
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

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
This function returns the class label containing the most predominant values
by identifying the greatest number of flowers within a class 
'''
def classify(data):

    label = data[:, -1]
    unique_classes, counts = np.unique(label, return_counts=True)
    idx = counts.argmax()
    classification = unique_classes[idx]

    return classification

# PART 3C: Determine Potential Splits
'''
This function will return a dictionary containing potential splits w.r.t. each feature
value (column) of the data frame
'''
def potential_splits(data):
    # initialize a dictionary object to store lists containing unique values for each feature
    potential_splits = {0: [], 1: [], 2: [], 3: []}

    # populate a temporary list with values containing the
    temp_list = []
    num_cols = data.shape[1]-1
    for column in range(0, num_cols):
        unique_value_col = np.unique(data[:, column])
        length = len(unique_value_col)
        for idx in range(0, length):
            if idx != 0:
                x_value = (unique_value_col[idx] + unique_value_col[idx-1])/2
                temp_list.append(x_value)
        potential_splits[column] = temp_list
        # empty temp_list for next iteration
        temp_list = []

    return potential_splits

# PART 3D: Split data function
'''
This function will split the data and return two numPy arrays based on 1. The column or 
feature and 2. A split value
'''
def split_value(data, feature, split_value):

    col = data[:, feature]

    data_below = data[col <= split_value]
    data_above = data[col > split_value]

    return data_below, data_above

# PART 3X: Build Decision tree Algorithm
def decision_tree():

    return True

# PART 4: Determine Accuracy


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
splits = potential_splits(train_data)

# plot the data
'''
sns.lmplot(x='Petal width', y='Petal length', data=train_df, hue='Label', fit_reg=False)
plt.vlines(x=splits[3], ymin=0, ymax=7)
plt.show()
'''

# plotting using the split value function
split_col = 3
threshold = 0.8
data_below, data_above = split_value(train_data, split_col, threshold)

data_below_df = pd.DataFrame(data_below, columns=df.columns)
sns.lmplot(x='Petal width', y='Petal length', data=data_below_df, hue='Label', fit_reg=False, size=6, aspect=1.5)
plt.vlines(x=threshold, ymin=0, ymax=7)
plt.xlim(0, 2.6)
plt.show()





