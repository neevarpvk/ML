#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:47:11 2020

@author: praveenvudumu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(precision = 3) #round off decimal values

df_diabetes_cleaned
#using cleaned df from DataCleaning.py

df_diabetes_cleaned.isnull().sum() #no missing values

X = df_diabetes_cleaned.drop('Outcome', axis = 1) #extract features
y = df_diabetes_cleaned['Outcome'] #target

# -------------------------------------------------------------------------------------------------------------- #

# MinMax Scaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1)) #very sensitive to outliners
X_scaled = scaler.fit_transform(X)
#X_scaled[0:5] #array of all records rescaled between 0 & 1
X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns) #convert from array to pd df
X_scaled_df.boxplot() #box plot confirms all records rescaled between 0 & 1

# -------------------------------------------------------------------------------------------------------------- #

# Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #centers all records with mean = 0 and variance = 1
scaler = scaler.fit(X)
X_std = scaler.transform(X)
X_std_df = pd.DataFrame(X_std, columns = X.columns)
#X_std_df.describe()
X_std_df.boxplot()

# -------------------------------------------------------------------------------------------------------------- #

# Normalization

from sklearn.preprocessing import Normalizer

scaler = Normalizer(norm = 'l1') # sum of the absolute values of individual features is 1
X_norm = scaler.fit_transform(X)
X_norm_df = pd.DataFrame(X_norm, columns = X.columns)

scaler = Normalizer(norm = 'l2') # sum of the squares of the individual features is 1
X_norm = scaler.fit_transform(X)
X_norm_df = pd.DataFrame(X_norm, columns = X.columns)

scaler = Normalizer(norm = 'max') # max value in a particular vector/feature = 1 and other vectors/features are represented in term of this max value
X_max = scaler.fit_transform(X)
X_max_df = pd.DataFrame(X_max, columns = X.columns)
X_max_df.head() #one feature in every record will be one (max feature value) and the remaining values are expressed in terms of this max value

# -------------------------------------------------------------------------------------------------------------- #

# Robust Scaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
X_robust_df = pd.DataFrame(X_robust, columns = X.columns)
X_robust_df.boxplot()

# -------------------------------------------------------------------------------------------------------------- #

# Binarization

from sklearn.preprocessing import Binarizer #to discretize a continuous numeric feature to be converted into a categorical form

binarizer = Binarizer(threshold = float((X['Pregnancies']).mean())) #values > mean = 1 and < mean = 0
X_bin = binarizer.fit_transform(X['Pregnancies'])

for i in range(1, X.shape[1]):
    binarizer = Binarizer(threshold = float((X[X.columns[i]]).mean())).fit(X[X.columns[i]])
    X_bin_feature = binarizer.transform(X[X.columns[i]])
    X_bin = np.concatenate((X_bin, X_bin_feature), axis=1) #for loop to binarize all features
#X_bin[0:5]

# -------------------------------------------------------------------------------------------------------------- #

# Build Logistic Regression model on these transformed features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def LR_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = LogisticRegression (solver = 'liblinear').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy score : ", accuracy_score(y_test, y_pred))

LR_model(X_scaled, y) #MinMaxScaler
LR_model(X_std, y) #StandardScaler
LR_model(X_norm, y) #Normalizer
LR_model(X_bin, y) #Binarizer

# -------------------------------------------------------------------------------------------------------------- #

# Convert Categorical data to Numeric data - Label encoding and One-hot encoding

# Label encoding - Encode target labels with value between 0 and n_classes-1

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
df_bcancer = read_csv(url, header = None)
df_bcancer.columns = [
    'age',
    'mp',
    'tsize',
    'invnodes',
    'nodecaps',
    'degmalig',
    'breast',
    'bquad',
    'irradiat',
    'class'
    ]

df_bcancer['bquad'].unique() #clean df
df_bcancer['bquad'] = np.where(df_bcancer['bquad'].str.contains('nan'), 'central', df_bcancer['bquad'])
df_bcancer.isnull().sum()

encoder = LabelEncoder() #recommended when no.of categories < 2 (0,1); if label encoded 0-9, scikit lib may give more weight to 9
encoder = encoder.fit(df_bcancer['irradiat'])
df_bcancer['irradiat'] = encoder.transform(df_bcancer['irradiat'].astype(str))
encoder.classes_ #categories that have been encoded

# -------------------------------------------------------------------------------------------------------------- #

# On-hot encoding

df_bcancer_encoded = pd.get_dummies(df_bcancer['bquad'])
df_bcancer = pd.concat([df_bcancer, df_bcancer_encoded], axis = 1)

# -------------------------------------------------------------------------------------------------------------- #

# Discretization

df_auto_cleaned
#using cleaned df from DataCleaning.py

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import KBinsDiscretizer

X = df_auto_cleaned[['Horsepower']]
y = df_auto_cleaned['MPG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
linear_model = LinearRegression(normalize = True).fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
r2_score(y_test, y_pred) #0.567

enc = KBinsDiscretizer(n_bins = 20, encode = 'ordinal') #Binning to compare r2 score; encode{‘onehot’, ‘onehot-dense’, ‘ordinal’}, (default=’onehot’)
X_binned = enc.fit_transform(X_train)
X_test_binned = enc.transform(X_test)
model = LinearRegression().fit(X_binned, y_train)
y_pred = model.predict(X_test_binned)
r2_score(y_test, y_pred) #0.596

enc = KBinsDiscretizer(n_bins = 20, encode = 'onehot') #Binning - onehot encode
X_binned = enc.fit_transform(X_train)
X_test_binned = enc.transform(X_test)
model = LinearRegression().fit(X_binned, y_train)
y_pred = model.predict(X_test_binned)
r2_score(y_test, y_pred) #0.642; this is best so far and hence binning did help in this ML model

# -------------------------------------------------------------------------------------------------------------- #
