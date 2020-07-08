#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:26:49 2020

@author: praveen
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------------------------------------------- #

df_auto = pd.read_csv("auto-mpg.csv")

#get data types and array size
df_auto.dtypes
df_auto.shape

#replace '?' with NaN
df_auto = df_auto.replace('?', np.nan)

#find NaN sum
df_auto.isna().sum()

#fill NaN with mean()
df_auto['MPG'] = df_auto['MPG'].fillna(df_auto['MPG'].astype(float).mean())
df_auto['Displacement'] = df_auto['Displacement'].fillna(df_auto['Displacement'].astype(float).mean())
df_auto['Weight'] = df_auto['Weight'].fillna(df_auto['Weight'].astype(float).mean())
#df['MPG'] = df['MPG'].fillna(df['MPG'].mean())

#drop all NaN since number of missing values are less now
df_auto = df_auto.dropna()

#drop 'car name' irrelevant column
df_auto.drop(['Model','bore','stroke','compression-ratio'], axis = 1, inplace = True)

#find values numeric in nature
df_auto['Year'].str.isnumeric().value_counts()
df_auto['Year'].loc[df_auto['Year'].str.isnumeric() == False]

#extract first four characters from 'Year'
df_auto_ext = df_auto['Year'].str.extract(r'^(\d{4})', expand=False)

df_auto['Year'].isnull().values.any() #confirm no missing values
df_auto['Year'] = pd.to_numeric(df_auto_ext) #change datatype to numberic and assign to df_auto

#Caluclate 'Age'
df_auto['Age'] = dt.datetime.now().year - df_auto['Year']
df_auto.drop(['Year'], axis = 1, inplace = True) # drop 'Year'

#convert datatypes
df_auto.dtypes

df_auto['MPG'] = pd.to_numeric(df_auto['MPG'], errors='coerce') #coerce-invalid parsing NaN
df_auto['Displacement'] = pd.to_numeric(df_auto['Displacement'], errors='coerce')
df_auto['Weight'] = pd.to_numeric(df_auto['Weight'], errors='coerce')

df_auto['Acceleration'].loc[df_auto['Acceleration'].str.isnumeric() == False] #no no-numeric values
df_auto['Acceleration'] = pd.to_numeric(df_auto['Acceleration'], errors='coerce')
df_auto['Horsepower'].loc[df_auto['Horsepower'].str.isnumeric() == False] #no no-numeric values
df_auto['Horsepower'] = pd.to_numeric(df_auto['Horsepower'], errors='coerce')

df_auto['Cylinders'].loc[df_auto['Cylinders'].str.isnumeric() == False] #find what records are not numeric
df_auto_cylinders = df_auto['Cylinders'].loc[df_auto['Cylinders'] != '-'] #extract all numeric values
cmean = df_auto_cylinders.astype(int).mean() #convert to integer and calc mean
df_auto['Cylinders'] = df_auto['Cylinders'].replace('-', cmean).astype(int) #replace missing values with mean

df_auto['Origin'].unique()
df_auto['Origin'] = np.where(df_auto['Origin'].str.contains('US'), 'US', df_auto['Origin']) #convert and replace if records contains 'US'
df_auto['Origin'] = np.where(df_auto['Origin'].str.contains('Europe'), 'Europe', df_auto['Origin'])
df_auto['Origin'] = np.where(df_auto['Origin'].str.contains('Japan'), 'Japan', df_auto['Origin'])

df_auto_cleaned = df_auto #cleaned df_auto

df_auto.drop(['Cylinders', 'Origin'], axis = 1, inplace = True) #drop these features because they are discrete and not continuos; optional use one-hot encoding?

# EDA

plt.bar(df_auto['Age'], df_auto['MPG']) #bar plot
plt.xlabel('Age')
plt.ylabel('Miles per gallon')

df_auto.plot.scatter(x='Weight',
                y='Acceleration',
                c='Horsepower',
                colormap='viridis',
                ); #scatter plot

df_auto_corr = df_auto.corr() #correlation matrix and heatmap
df_auto_corr
sns.heatmap(df_auto_corr, annot=True, cmap = plt.cm.nipy_spectral_r)

#box plot to detect outliners
df_auto.boxplot(column = ['Age'])

# Standardize the 'Age' feature; mean = 0 and variance = 1
scalar = StandardScaler()
age_scaled = scalar.fit_transform(df_auto["Age"].values.reshape(-1,1))
df_auto["Age_Scaled"] = age_scaled #add to df_auto
df_auto.boxplot(column = ['Age_Scaled'])
age_outliner = df_auto.loc[df_auto["Age_Scaled"] > 3] #outliners can be records whose StandDev is more than 3
age_outliner #view Age records with StdDev > 3
#set outliners to mean
df_auto[((df_auto['Age'] >= 120) & (df_auto['Age'] <= -1))] = df_auto["Age"].mean()
df_auto.boxplot(column = ["Age"])

# Regression - Normalize
X = df_auto.drop('MPG', axis=1)
y = df_auto['MPG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
linear_model = LinearRegression(normalize = True).fit(X_train, y_train)
print('Training score: ', linear_model.score(X_train, y_train))
y_pred = linear_model.predict(X_test)
#print('Testing score: ', r2_score(y_test, y_pred)) # Convert all to numeric (one-hot encoding)

# -------------------------------------------------------------------------------------------------------------- #

# Univarient Feature Imputation (Impute/Inferring missing values from current data - constant, mean, median, mode)

from sklearn.impute import SimpleImputer

df_diabetes = pd.read_csv("diabetes.csv")
#get basic stats of df
df_diabetes.shape
df_diabetes.info()
df_diabetes.describe().transpose() #Min value of 0 for features in this df_diabetes indicates missing values; working with data in real world, we should understand how missing values is represented in df_diabetes

#replace 0 with NaN
df_diabetes['Glucose'].replace(0, np.nan, inplace= True)
df_diabetes['BloodPressure'].replace(0, np.nan, inplace= True)
df_diabetes['SkinThickness'].replace(0, np.nan, inplace= True)
df_diabetes['Insulin'].replace(0, np.nan, inplace= True)
df_diabetes['BMI'].replace(0, np.nan, inplace= True)

df_diabetes.isnull().sum() #find missing values

#df_diabetes_diabetes = df_diabetes

df_diabetes_skinthickness = df_diabetes['SkinThickness'].values.reshape(-1,1) #reshape feature in 2D array
df_diabetes_skinthickness.shape

imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent') #mode to fill missing values
imp.fit(df_diabetes['SkinThickness'].values.reshape(-1,1)) #fit on existing values
df_diabetes['SkinThickness'] = imp.transform(df_diabetes['SkinThickness'].values.reshape(-1,1)) #transform will fill missing values

df_diabetes['SkinThickness'].describe() #stats change

imp = SimpleImputer(missing_values = np.nan, strategy='median') #median to fill missing values; 50th percentile of data
imp.fit(df_diabetes['Glucose'].values.reshape(-1,1))
df_diabetes['Glucose'] = imp.transform(df_diabetes['Glucose'].values.reshape(-1,1))

imp = SimpleImputer(missing_values = np.nan, strategy='mean') #mean/avg to fill missing values
imp.fit(df_diabetes['BloodPressure'].values.reshape(-1,1))
df_diabetes['BloodPressure'] = imp.transform(df_diabetes['BloodPressure'].values.reshape(-1,1))

imp = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value = 32) #constant 32 to fill missing values
imp.fit(df_diabetes['BMI'].values.reshape(-1,1))
df_diabetes['BMI'] = imp.transform(df_diabetes['BMI'].values.reshape(-1,1))


# -------------------------------------------------------------------------------------------------------------- #

# Multivariant Feature Imputation (Use entire set of available features to estimate the missing values and not just that missing value feature alone)
# Models each feature with missing calues as a function of other features in an iterative round-robbin fashion and fits a regressor to find missing values

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_diabetes_features = df_diabetes.drop('Outcome', axis = 1) #drop Y lable to calc the missing values in Insulin feature using all other features
df_diabetes_label = df_diabetes['Outcome']

imp = IterativeImputer(max_iter = 1000, random_state = 42)
imp.fit(df_diabetes_features)

df_diabetes_features_arr = imp.transform(df_diabetes_features) #array with all missing values filled in
df_diabetes_features = pd.DataFrame(df_diabetes_features_arr, columns = df_diabetes_features.columns) #convert array to pd df_diabetes to have same columns

df_diabetes = pd.concat([df_diabetes_features, df_diabetes_label], axis = 1)#concat the features and labels to one df_diabetes

df_diabetes_cleaned = df_diabetes

# -------------------------------------------------------------------------------------------------------------- #
