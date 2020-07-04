#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:57:37 2020

@author: praveen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:09:14 2020

@author: praveenvudumu
"""

# import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# load data and label columns

# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
df = pd.read_csv("/Users/praveenvudumu/desktop/datascience/datasets/heatdisease_uciml.data", header = None)
df.columns = [
    'age',
    'sex',
    'cp',
    'restbp',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
    'hd' # heart disease
    ]
df.head()
len(df)


# missing data

df.dtypes # data types for each column
df['ca'].unique() # find unique values
df['thal'].unique()

len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')]) # find no.of missing values
df.loc[(df['ca'] == '?') | (df['thal'] == '?')] # print missing value rows

df = df.loc[(df['ca'] != '?') & (df['thal'] != '?')] #since only 6 are missing we can delete


# split dataset to train and test

X = df.drop('hd', axis = 1)
y = df['hd']

X = pd.get_dummies(X, columns = ['cp', 'restecg', 'slope', 'thal']) # one-hot encoding for categorical variables

y.unique() # 0 - no hd; 1-4 different levels of hd
y_index = y > 0 # convert to 0 if hd > 0
y[y_index] = 1 # 1 - has hd

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42) # default split

print("Training features/target:", X_train.shape, y_train.shape)
print("Testing features/target:", X_test.shape, y_test.shape)


# build classification tree

df_dtc = DecisionTreeClassifier(random_state = 42)
df_dtc = df_dtc.fit(X_train, y_train)
plot_tree(df_dtc,
         filled = True,
         rounded = True,
         class_names = ["No Heart Disease", "Yes Heart Disease"],
         feature_names = X.columns
         ) # plot initial tree

plot_confusion_matrix(df_dtc, X_test, y_test, cmap=plt.cm.Blues, display_labels = ["No HD", "Yes HD"]) # confusion matrix


# Post pruning decision trees with cost complexity pruning
## https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

#path = df_dtc.cost_complexity_pruning_path(X_train, y_train)
#ccp_alphas, impurities = path.ccp_alphas, path.impurities

path = df_dtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1] #remove max value of alpha

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = "train", drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker = 'o', label = "test", drawstyle = "steps-post")
ax.legend()
plt.show()


# Cross validation to find the best value of 'alpha'

df_dtc = DecisionTreeClassifier(random_state = 42, ccp_alpha = 0.016)
scores = cross_val_score(df_dtc, X_train, y_train, cv = 5)
df = pd.DataFrame(data = {'tree': range(5), 'accuracy': scores})
df.plot(x = 'tree', y = 'accuracy', marker = 'o', linestyle = '--')

alpha_loop = []
for ccp_alpha in ccp_alphas:
    dt_dtc = DecisionTreeClassifier(random_state = 42, ccp_alpha = ccp_alpha)
    scores = cross_val_score(df_dtc, X_train, y_train, cv = 5)
    alpha_loop.append([ccp_alpha, np.mean(scores), np.std(scores)])

df_alpha = pd.DataFrame(alpha_loop, columns = ['alpha', 'mean_accuracy', 'std'])
df_alpha.plot(x = 'alpha', y = 'mean_accuracy',
              yerr = 'std', marker = 'o', linestyle = '--')

df_alpha[(df_alpha['alpha'] > 0.016) & (df_alpha['alpha'] < 0.015)] # no change in accuracy

ccp_alpha = 0.016 #ideal alpha value

df_pruned = DecisionTreeClassifier(random_state = 42, ccp_alpha = ccp_alpha)
df_pruned = df_pruned.fit(X_train, y_train)

plot_confusion_matrix(df_pruned, X_test, y_test,
                      cmap=plt.cm.Blues, display_labels = ["No HD", "Yes HD"])

# final pruned tree
plot_tree(df_pruned,
          filled = True,
          rounded = True,
          class_names = ["No HD", "Yes HD"],
          feature_names = X.columns)


#######################################################################################

# Using RandomizedSearchCV()

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv = 10)
tree_cv.fit(X_train, y_train)

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
