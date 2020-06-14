#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:09:14 2020

@author: praveen
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
df = pd.read_csv("/Users/praveen/desktop/datascience/datasets/heatdisease_uciml.data", header = None)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) #split 80/20

print("Training features/target:", X_train.shape, y_train.shape)
print("Testing features/target:", X_test.shape, y_test.shape)


# build classification tree

df_dtc = DecisionTreeClassifier(random_state = 11)
df_dtc = df_dtc.fit(X_train, y_train)
plot_tree(df_dtc,
         filled = True,
         rounded = True,
         class_names = ["No Heart Disease", "Yes Heart Disease"],
         feature_names = X.columns
         ) # plot initial tree

plot_confusion_matrix(df_dtc, X_test, y_test, display_labels = ["No HD", "Yes HD"]) # confusion matrix


# Post pruning decision trees with cost complexity pruning
## https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

path = df_dtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total impurity vs effective alpha for training set")

df_dtcs = []

for ccp_alpha in ccp_alphas:
    df_dtc = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    df_dtc.fit(X_train, y_train)
    df_dtcs.append(df_dtc)

print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(df_dtcs[-1].tree_.node_count, ccp_alphas[-1]))

train_scores = [df_dtc.score(X_train, y_train) for df_dtc in df_dtcs]
test_scores = [df_dtc.score(X_test, y_test) for df_dtc in df_dtcs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = "train", drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker = 'o', label = "test", drawstyle = "steps-post")
ax.legend()
plt.show()


# Cross validation to find the best value of 'alpha'

df_dtc = DecisionTreeClassifier(random_state = 42, ccp_alpha = 0.02)
scores = cross_val_score(df_dtc, X_train, y_train, cv = 4) # cv = 4 because the df is pretty small; cv > 4 and cv < 4 gave lower accuracy scores
df = pd.DataFrame(data = {'tree': range(4), 'accuracy': scores})
df.plot(x = 'tree', y = 'accuracy', marker = 'o', linestyle = '--')

alpha_loop = []
for ccp_alpha in ccp_alphas:
    dt_dtc = DecisionTreeClassifier(random_state = 42, ccp_alpha = 0.02)
    scores = cross_val_score(df_dtc, X_train, y_train, cv = 5)
    alpha_loop.append([ccp_alpha, np.mean(scores), np.std(scores)])

df_alpha = pd.DataFrame(alpha_loop, columns = ['alpha', 'mean_accuracy', 'std'])

df_alpha.plot(x = 'alpha', y = 'mean_accuracy', yerr = 'std', marker = 'o', linestyle = '--') # from this grap alpha = 0.014

df_alpha[(df_alpha['alpha'] > 0.014) & df_alpha['alpha'] < 0.015] # no change in accuracy

ccp_alpha = 0.014 #ideal alpha value

df_pruned = DecisionTreeClassifier(random_state = 42, ccp_alpha = ccp_alpha)
df_pruned = df_pruned.fit(X_train, y_train)

plot_confusion_matrix(df_pruned, X_test, y_test,
                      display_labels = ["No HD", "Yes HD"])

# final pruned tree
plot_tree(df_pruned,
          filled = True,
          rounded = True,
          class_names = ["No HD", "Yes HD"],
          feature_names = X.columns)
