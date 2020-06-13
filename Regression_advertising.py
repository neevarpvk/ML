#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:17:45 2020

@author: praveenvudumu
"""
# Import modules
import pandas as pd
pd.set_option('precision', 2) #2 digit precision for float
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from sklearn import preprocessing #data scaling - normalization and standardization
#from sklearn.metrics import confusion_matrix

# Import dataset
df = pd.read_csv("/Users/praveen/desktop/datascience/datasets/advertising.csv")

# Exploring the df

print(df)
df = df.copy().drop(['Unnamed: 0'],axis=1) # dropping column since its essentially an index
print(df.describe())

# Visualizing the Features (sales vs features)
features = df.drop(['sales'], axis = 1)
for feature in features:
    plt.figure(figsize = (8,5)) # 8x5 plot
    sns.scatterplot(data = df, x = feature,
                    y = 'sales', hue = 'sales',
                    palette = 'cool', legend = False)

# Splitting the df for Training and Testing
from sklearn.model_selection import train_test_split

X = df.drop('sales', axis=1)
y = df['sales']

# X = preprocessing.normalize(X) # Data Normalization
X = preprocessing.scale(X) #Data Standardization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11) #split 80/20 and

print("Training features/target:", X_train.shape, y_train.shape)
print("Testing features/target:", X_test.shape, y_test.shape)

# Comparing with other Regressors
from time import time

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR, LinearSVR

from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score

estimators = [
    LinearRegression(),
    RANSACRegressor(),
    KNeighborsRegressor(),
    KNeighborsRegressor(n_neighbors=9, metric='manhattan'),
    LinearSVR(),
    GaussianProcessRegressor(),
    SVR(kernel = 'linear'),
    SVR(),
    ]

head = 7
for model in estimators [:head]:
    start = time()
    model.fit (X_train, y_train)
    train_time = time() - start
    start = time()
    predictions = model.predict (X_test)
    predict_time = time() - start
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(y_test, predictions))
    print("\tMean absolute error:", mean_absolute_error(y_test, predictions))
    print("\tR2 score:", r2_score(y_test, predictions))
    print()

